import os
import sys
import json
import time
import logging
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.base import clone
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.evaluation import (
    compute_metrics, plot_confusion_matrix, plot_roc_curves,
    plot_pr_curves, plot_feature_importance, plot_model_comparison,
    create_comparison_table, evaluate_temporal_holdout, analyze_cluster_errors,
    SEVERITY_CLASSES, SEVERITY_NAMES
)
from src.feature_engineering import (
    build_preprocessing_pipeline, get_feature_lists,
    create_temporal_features, create_binary_features, create_interaction_features,
    frequency_encode_local_authority
)
from src.data_cleaning import asymmetric_undersample
from src.clustering import prepare_clustering_data, fit_kmeans
from src.models import create_histgbt_pipeline

# 日志配置
log_path = os.path.join(config.RESULTS_DIR, 'phase12_log.txt')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode='a'),
    ]
)
logger = logging.getLogger(__name__)


# 重建 Phase 1-4 的特征数据（用于时序评估）
def replay_phase_1_to_4():
    logger.info("  Replaying Phase 1-4 to reconstruct featured_df...")
    from src.data_loading import (
        load_accidents, load_vehicles, parse_datetime,
        aggregate_vehicles, merge_accidents_vehicles, load_context_tables
    )
    from src.data_cleaning import (
        identify_missing_patterns, drop_high_missing_columns,
        remove_leakage_columns, clip_outliers
    )

    t0 = time.time()

    # 阶段1：加载
    accidents_df, _ = load_accidents(config.ACCIDENTS_FILE)
    vehicles_df = load_vehicles(config.VEHICLES_FILE)
    accidents_df = parse_datetime(accidents_df)
    context_tables = load_context_tables(config.CONTEXT_DIR)

    # 阶段3后阶段2：先合并再清洗（与 main.py 一致）
    vehicles_agg = aggregate_vehicles(vehicles_df)
    merged_df = merge_accidents_vehicles(accidents_df, vehicles_agg)
    _ = identify_missing_patterns(merged_df)
    cleaned_df, _ = drop_high_missing_columns(merged_df)
    cleaned_df = remove_leakage_columns(cleaned_df)
    cleaned_df = clip_outliers(cleaned_df)

    # 阶段4：特征工程
    featured_df = create_temporal_features(cleaned_df)
    featured_df = create_binary_features(featured_df, context_tables)
    featured_df = create_interaction_features(featured_df)

    logger.info(f"  Phase 1-4 replay complete: {featured_df.shape} ({time.time()-t0:.1f}s)")
    return featured_df


# 读取调参最佳参数
def get_best_params():
    """从调参检查点读取第11阶段的最佳参数。"""
    ckpt_path = os.path.join(config.RESULTS_DIR, 'tuning_checkpoint.json')
    with open(ckpt_path) as f:
        ckpt = json.load(f)

    best_params = {}
    for name, info in ckpt['completed'].items():
        params = {}
        for k, v in info['best_params'].items():
            param_name = k.replace('model__', '')
            if v == 'None':
                params[param_name] = None
            elif v in ('l1', 'l2', 'liblinear', 'saga'):
                params[param_name] = v
            else:
                try:
                    params[param_name] = int(v)
                except ValueError:
                    try:
                        params[param_name] = float(v)
                    except ValueError:
                        params[param_name] = v
        best_params[name] = params
    return best_params


# 主入口：完成 L2/L3 评估与全部图表
def main():
    logger.info("=" * 60)
    logger.info("PHASE 12-13: Final Evaluation + All Visualizations")
    logger.info("=" * 60)
    t_total = time.time()

    # 加载第7阶段数据
    pkl_path = os.path.join(config.RESULTS_DIR, 'phase7_data.pkl')
    logger.info(f"Loading Phase 7 data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    X_train_sampled = data['X_train_sampled']
    y_train_sampled = data['y_train_sampled']
    X_test = data['X_test']
    y_test = data['y_test']
    logger.info(f"  Train: {X_train_sampled.shape}, Test: {X_test.shape}")

    best_params = get_best_params()
    logger.info(f"  Best params loaded: {list(best_params.keys())}")

    # L2 测试集评估
    logger.info("\n" + "=" * 60)
    logger.info("PART 1: L2 Test Set Evaluation (tuned models)")
    logger.info("=" * 60)

    test_results = {}
    y_proba_dict = {}
    fitted_pipelines = {}

    for name in ['RandomForest', 'HistGBT', 'LogisticRegression']:
        pkl_file = os.path.join(config.RESULTS_DIR, f'tuning_{name}_best_pipeline.pkl')
        logger.info(f"\n  Loading {name} from {pkl_file}...")
        with open(pkl_file, 'rb') as f:
            pipeline = pickle.load(f)
        fitted_pipelines[name] = pipeline

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)

        metrics = compute_metrics(y_test, y_pred, y_proba, class_names=SEVERITY_NAMES)
        test_results[name] = metrics
        y_proba_dict[name] = y_proba

        logger.info(f"  {name}: Macro F1={metrics['macro_f1']:.4f}, "
                     f"Fatal Recall={metrics['Fatal_recall']:.4f}, "
                     f"Fatal Precision={metrics['Fatal_precision']:.4f}, "
                     f"Balanced Acc={metrics['balanced_accuracy']:.4f}")

    # 关键结果：RF 的致死召回
    rf_fatal_recall = test_results['RandomForest']['Fatal_recall']
    logger.info(f"\n  *** RF TUNED FATAL RECALL = {rf_fatal_recall:.4f} ***")
    logger.info(f"  (Phase 9 default was 0.005, ablation all-schemes <0.01)")

    # 保存测试集结果
    test_df = create_comparison_table(test_results,
        save_path=os.path.join(config.RESULTS_DIR, 'Table_test_results.csv'))
    logger.info(f"\n  Test results summary:")
    key_cols = ['Model', 'accuracy', 'balanced_accuracy', 'macro_f1',
                'Fatal_recall', 'Fatal_precision', 'Serious_recall']
    existing_cols = [c for c in key_cols if c in test_df.columns]
    logger.info(f"\n{test_df[existing_cols].to_string(index=False)}")

    # CV 与测试集对比
    logger.info("\n  CV (Phase 11) vs Test (Phase 12) Macro F1:")
    cv_scores = {'RandomForest': 0.4482, 'HistGBT': 0.4082, 'LogisticRegression': 0.4205}
    for name in ['RandomForest', 'HistGBT', 'LogisticRegression']:
        cv = cv_scores[name]
        test = test_results[name]['macro_f1']
        delta = test - cv
        logger.info(f"    {name}: CV={cv:.4f}, Test={test:.4f}, delta={delta:+.4f}")

    # Cluster_ID 对比
    logger.info("\n" + "=" * 60)
    logger.info("PART 2: Cluster_ID Comparison (Table18)")
    logger.info("=" * 60)

    cluster_results = {}
    model_types = {'RandomForest': 'tree', 'HistGBT': 'histgbt', 'LogisticRegression': 'linear'}

    for name, mtype in model_types.items():
        logger.info(f"\n  Building {name} + Cluster_ID pipeline...")
        pp = build_preprocessing_pipeline(mtype, include_cluster=True)

        params = {**best_params[name], 'class_weight': 'balanced', 'random_state': config.RANDOM_SEED}
        if name == 'RandomForest':
            params['n_jobs'] = -1
            model = RandomForestClassifier(**params)
        elif name == 'HistGBT':
            features = get_feature_lists()
            n_num_bin = len(features['numeric']) + len(features['binary'])
            n_cat = len(features['categorical']) + 1  # 额外加入 Cluster_ID
            cat_indices = list(range(n_num_bin, n_num_bin + n_cat))
            params['categorical_features'] = cat_indices
            model = HistGradientBoostingClassifier(**params)
        else:
            model = LogisticRegression(**params)

        pipe = Pipeline([('preprocessing', pp), ('model', model)])
        pipe.fit(X_train_sampled, y_train_sampled)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba, class_names=SEVERITY_NAMES)
        cluster_results[name + '+Cluster'] = metrics

        # 相对基线的变化
        base_f1 = test_results[name]['macro_f1']
        clust_f1 = metrics['macro_f1']
        base_fr = test_results[name]['Fatal_recall']
        clust_fr = metrics['Fatal_recall']
        logger.info(f"  {name}: Macro F1 {base_f1:.4f}→{clust_f1:.4f} ({clust_f1-base_f1:+.4f}), "
                     f"Fatal Recall {base_fr:.4f}→{clust_fr:.4f} ({clust_fr-base_fr:+.4f})")

    # 保存 Table18
    cluster_rows = []
    for name in ['RandomForest', 'HistGBT', 'LogisticRegression']:
        cluster_rows.append({
            'model': name,
            'baseline_macro_f1': test_results[name]['macro_f1'],
            'cluster_macro_f1': cluster_results[name + '+Cluster']['macro_f1'],
            'delta_macro_f1': cluster_results[name + '+Cluster']['macro_f1'] - test_results[name]['macro_f1'],
            'baseline_fatal_recall': test_results[name]['Fatal_recall'],
            'cluster_fatal_recall': cluster_results[name + '+Cluster']['Fatal_recall'],
            'delta_fatal_recall': cluster_results[name + '+Cluster']['Fatal_recall'] - test_results[name]['Fatal_recall'],
        })
    pd.DataFrame(cluster_rows).to_csv(
        os.path.join(config.RESULTS_DIR, 'Table_cluster_id_comparison.csv'),
        index=False, float_format='%.6f')

    # L3 时序留出集评估
    logger.info("\n" + "=" * 60)
    logger.info("PART 3: L3 Temporal Holdout")
    logger.info("=" * 60)

    featured_df = replay_phase_1_to_4()

    target = 'Accident_Severity'
    temporal_train_full = featured_df[featured_df['Year'] <= config.TEMPORAL_TRAIN_END_YEAR].copy()
    temporal_test_full = featured_df[featured_df['Year'] >= config.TEMPORAL_TEST_START_YEAR].copy()

    logger.info(f"  Temporal train (≤{config.TEMPORAL_TRAIN_END_YEAR}): {temporal_train_full.shape}")
    logger.info(f"  Temporal test (≥{config.TEMPORAL_TEST_START_YEAR}): {temporal_test_full.shape}")

    for label, df_part in [('temporal_train', temporal_train_full), ('temporal_test', temporal_test_full)]:
        dist = df_part[target].value_counts(normalize=True).sort_index()
        logger.info(f"  {label} severity: " + ", ".join(f"{SEVERITY_NAMES[i]}={dist.get(c,0):.3f}" for i, c in enumerate(SEVERITY_CLASSES)))

    X_temp_train = temporal_train_full.drop(columns=[target])
    y_temp_train = temporal_train_full[target]
    X_temp_test = temporal_test_full.drop(columns=[target])
    y_temp_test = temporal_test_full[target]

    train_df_temp = pd.concat([X_temp_train, y_temp_train], axis=1)
    sampled_temp = asymmetric_undersample(
        train_df_temp, slight_ratio=config.SLIGHT_SAMPLE_RATIO,
        random_state=config.RANDOM_SEED)
    X_temp_train_s = sampled_temp.drop(columns=[target])
    y_temp_train_s = sampled_temp[target]

    X_temp_train_enc, X_temp_test_enc, _ = frequency_encode_local_authority(
        X_temp_train_s, X_temp_test)

    logger.info(f"  Temporal train (after Scheme C): {X_temp_train_enc.shape}")
    logger.info(f"  Temporal test (encoded): {X_temp_test_enc.shape}")

    temporal_results = {}
    for name, mtype in model_types.items():
        logger.info(f"\n  Temporal holdout: {name}...")
        pp = build_preprocessing_pipeline(mtype, include_cluster=False)
        params = {**best_params[name], 'class_weight': 'balanced', 'random_state': config.RANDOM_SEED}
        if name == 'RandomForest':
            params['n_jobs'] = -1
            model = RandomForestClassifier(**params)
        elif name == 'HistGBT':
            features = get_feature_lists()
            n_num_bin = len(features['numeric']) + len(features['binary'])
            n_cat = len(features['categorical'])
            cat_indices = list(range(n_num_bin, n_num_bin + n_cat))
            params['categorical_features'] = cat_indices
            model = HistGradientBoostingClassifier(**params)
        else:
            model = LogisticRegression(**params)

        pipe = Pipeline([('preprocessing', pp), ('model', model)])
        metrics = evaluate_temporal_holdout(pipe, X_temp_train_enc, y_temp_train_s,
                                            X_temp_test_enc, y_temp_test, name)
        temporal_results[name] = metrics

    # Table19：三层对比
    comparison_rows = []
    for name in ['RandomForest', 'HistGBT', 'LogisticRegression']:
        comparison_rows.append({
            'model': name,
            'CV_macro_f1': cv_scores[name],
            'test_macro_f1': test_results[name]['macro_f1'],
            'temporal_macro_f1': temporal_results[name]['macro_f1'],
            'CV_vs_test': test_results[name]['macro_f1'] - cv_scores[name],
            'test_vs_temporal': temporal_results[name]['macro_f1'] - test_results[name]['macro_f1'],
            'test_fatal_recall': test_results[name]['Fatal_recall'],
            'temporal_fatal_recall': temporal_results[name]['Fatal_recall'],
            'test_balanced_acc': test_results[name]['balanced_accuracy'],
            'temporal_balanced_acc': temporal_results[name]['balanced_accuracy'],
        })
    pd.DataFrame(comparison_rows).to_csv(
        os.path.join(config.RESULTS_DIR, 'Table_cv_vs_test_vs_temporal.csv'),
        index=False, float_format='%.6f')

    # 聚类错误分析
    logger.info("\n" + "=" * 60)
    logger.info("PART 4: Cluster x Error Analysis")
    logger.info("=" * 60)


    scaled_train, scaler_cluster, _ = prepare_clustering_data(
        X_train_sampled, config.CLUSTER_FEATURES, sample_size=None,
        random_state=config.RANDOM_SEED)
    km_model, _ = fit_kmeans(scaled_train, n_clusters=4, random_state=config.RANDOM_SEED)

    test_cluster_features = X_test[config.CLUSTER_FEATURES].fillna(0)
    test_scaled = scaler_cluster.transform(test_cluster_features)
    test_cluster_ids = km_model.predict(test_scaled)

    best_model_name = 'RandomForest'
    best_pipeline = fitted_pipelines[best_model_name]
    y_pred_best = best_pipeline.predict(X_test)

    error_df, fatal_pattern_df = analyze_cluster_errors(
        y_test, y_pred_best, test_cluster_ids, class_names=SEVERITY_NAMES)

    error_df.to_csv(os.path.join(config.RESULTS_DIR, 'Table_cluster_error_analysis.csv'),
                     index=False, float_format='%.4f')
    fatal_pattern_df.to_csv(os.path.join(config.RESULTS_DIR, 'Table_fatal_misclass_pattern.csv'),
                             index=False)

    logger.info(f"\n  Fatal misclassification patterns ({best_model_name}):")
    logger.info(f"\n{fatal_pattern_df.to_string(index=False)}")
    # 统一生成图表
    logger.info("\n" + "=" * 60)
    logger.info("PART 5: Generating All Figures")
    logger.info("=" * 60)
    logger.info("  Generating Fig01-03 (EDA)...")

    fig, ax = plt.subplots(figsize=(8, 5))
    severity_counts = featured_df[target].value_counts().sort_index()
    bars = ax.bar(SEVERITY_NAMES,
                  [severity_counts.get(c, 0) for c in SEVERITY_CLASSES],
                  color=['#d62728', '#ff7f0e', '#2ca02c'])
    for bar, count in zip(bars, [severity_counts.get(c, 0) for c in SEVERITY_CLASSES]):
        pct = count / len(featured_df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Accident Severity Distribution (Full Dataset)', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, 'Fig01_target_distribution.png'),
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    numeric_cols = ['Speed_limit', 'Number_of_Vehicles', 'Hour', 'Mean_Driver_Age',
                    'Is_Dark', 'Bad_Weather', 'Wet_Road', 'Is_Rural', 'At_Junction']
    valid_cols = [c for c in numeric_cols if c in featured_df.columns]
    corr = featured_df[valid_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax, vmin=-1, vmax=1, linewidths=0.5)
    ax.set_title('Feature Correlation Matrix (Pearson)', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, 'Fig02_correlation_heatmap.png'),
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, feat, title in zip(axes,
        ['Speed_limit', 'Hour', 'Is_Dark'],
        ['Speed Limit', 'Hour of Day', 'Darkness']):
        for sev_code, sev_name, color in zip(SEVERITY_CLASSES, SEVERITY_NAMES,
                                              ['#d62728', '#ff7f0e', '#2ca02c']):
            subset = featured_df[featured_df[target] == sev_code][feat].dropna()
            ax.hist(subset, bins=30, alpha=0.5, label=sev_name, color=color, density=True)
        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{title} by Severity', fontsize=13)
        ax.legend(fontsize=9)
    plt.suptitle('Key Feature Distributions by Severity', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, 'Fig03_feature_severity_dist.png'),
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    logger.info("  Generating Fig10 (Confusion matrices)...")
    for name in ['RandomForest', 'HistGBT', 'LogisticRegression']:
        y_pred = fitted_pipelines[name].predict(X_test)
        plot_confusion_matrix(y_test, y_pred, SEVERITY_NAMES, name,
            save_path=os.path.join(config.FIGURES_DIR, f'Fig10_confusion_matrix_{name}.png'))

    logger.info("  Generating Fig11 (Model comparison)...")
    plot_model_comparison(test_results,
        save_path=os.path.join(config.FIGURES_DIR, 'Fig11_model_comparison.png'))

    logger.info("  Generating Fig12 (ROC curves)...")
    plot_roc_curves(y_test, y_proba_dict, SEVERITY_NAMES,
        save_path=os.path.join(config.FIGURES_DIR, 'Fig12_roc_curves.png'))

    logger.info("  Generating Fig13 (PR curves)...")
    plot_pr_curves(y_test, y_proba_dict, SEVERITY_NAMES,
        save_path=os.path.join(config.FIGURES_DIR, 'Fig13_pr_curves.png'))

    logger.info("  Generating Fig14 (Feature importance)...")
    for name in ['RandomForest', 'HistGBT']:
        pipeline = fitted_pipelines[name]
        model = pipeline.named_steps['model']
        ct = pipeline.named_steps['preprocessing']
        try:
            feat_names = list(ct.get_feature_names_out())
        except Exception:
            feat_names = [f'F{i}' for i in range(len(model.feature_importances_))]
        plot_feature_importance(model, feat_names, name, top_n=15,
            save_path=os.path.join(config.FIGURES_DIR, f'Fig14_feature_importance_{name}.png'))

    logger.info("  Generating Fig15 (LR coefficients)...")
    lr_pipeline = fitted_pipelines['LogisticRegression']
    lr_model = lr_pipeline.named_steps['model']
    lr_ct = lr_pipeline.named_steps['preprocessing']
    try:
        lr_feat_names = list(lr_ct.get_feature_names_out())
    except Exception:
        lr_feat_names = [f'F{i}' for i in range(lr_model.coef_.shape[1])]
    plot_feature_importance(lr_model, lr_feat_names, 'LogisticRegression', top_n=20,
        save_path=os.path.join(config.FIGURES_DIR, 'Fig15_lr_coefficients.png'))

    logger.info("  Generating cluster error figure...")
    pivot = error_df.pivot_table(index='cluster', columns='class', values='error_rate')
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot_ordered = pivot[SEVERITY_NAMES] if all(c in pivot.columns for c in SEVERITY_NAMES) else pivot
    sns.heatmap(pivot_ordered, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax,
                linewidths=0.5, vmin=0, vmax=1)
    ax.set_title(f'Per-Cluster Error Rate by Severity ({best_model_name})', fontsize=14)
    ax.set_ylabel('Cluster', fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(config.FIGURES_DIR, 'Fig_cluster_error_distribution.png'),
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)

    total_time = time.time() - t_total
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE 12-13 COMPLETE — Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
