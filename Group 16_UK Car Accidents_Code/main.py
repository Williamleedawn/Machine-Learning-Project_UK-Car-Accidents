import os
import warnings
import time
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src.data_loading import (
    load_accidents, load_vehicles, parse_datetime,
    aggregate_vehicles, merge_accidents_vehicles, load_context_tables
)
from src.data_cleaning import (
    identify_missing_patterns, drop_high_missing_columns,
    remove_leakage_columns, clip_outliers, asymmetric_undersample,
    generate_cleaning_report
)
from src.feature_engineering import (
    create_temporal_features, create_binary_features,
    create_interaction_features, frequency_encode_local_authority,
    get_feature_lists, build_preprocessing_pipeline
)
from src.clustering import (
    prepare_clustering_data, find_optimal_k, fit_kmeans,
    fit_agglomerative, compute_cluster_profiles,
    cluster_severity_crosstab, compute_pca_2d,
    plot_elbow, plot_silhouette, plot_pca_clusters,
    plot_cluster_profiles, plot_cluster_severity, plot_dendrogram
)
from sklearn.metrics import adjusted_rand_score
from src.models import get_model_configs
from src.evaluation import (
    compute_metrics, plot_confusion_matrix, plot_roc_curves,
    plot_pr_curves, plot_feature_importance, plot_model_comparison,
    create_comparison_table, evaluate_temporal_holdout,
    analyze_cluster_errors
)
from src.tuning import (
    run_gridsearch, imbalance_ablation, plot_tuning_sensitivity,
    plot_tuning_cv_boxplot, plot_tuning_before_after, summarize_best_params
)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
# 过滤部分兼容性警告
warnings.filterwarnings('ignore', category=FutureWarning)


# 初始 EDA 与数据假设检查
def run_initial_eda(accidents_df, minus_one_counts, context_tables):
    logger.info("=" * 60)
    logger.info("INITIAL EDA: Verifying data assumptions")
    logger.info("=" * 60)

    import os
    from config import RESULTS_DIR

    # 1. 数据集规模
    logger.info(f"  Shape: {accidents_df.shape[0]:,} rows × {accidents_df.shape[1]} columns")

    # 2. 列类型概览
    dtype_counts = accidents_df.dtypes.value_counts()
    logger.info(f"  Dtypes: {dict(dtype_counts)}")

    # 3. 目标分布（Accident_Severity：1致死，2严重，3轻微）
    target_col = 'Accident_Severity'
    if target_col in accidents_df.columns:
        severity_counts = accidents_df[target_col].value_counts().sort_index()
        severity_pcts = accidents_df[target_col].value_counts(normalize=True).sort_index() * 100

        severity_labels = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
        if 'Accident_Severity' in context_tables:
            ct = context_tables['Accident_Severity']
            severity_labels = dict(zip(ct['code'], ct['label']))

        logger.info(f"  Target distribution (Accident_Severity):")
        for code in severity_counts.index:
            label = severity_labels.get(code, f'Code_{code}')
            count = severity_counts[code]
            pct = severity_pcts[code]
            logger.info(f"    {code} ({label}): {count:,} ({pct:.2f}%)")

    # 4. 替换前 -1 频次
    logger.info(f"  Columns with -1 values (before NaN replacement):")
    if minus_one_counts:
        for col, count in sorted(minus_one_counts.items(), key=lambda x: -x[1]):
            pct = count / len(accidents_df) * 100
            logger.info(f"    {col}: {count:,} ({pct:.2f}%)")
    else:
        logger.info("    None found")

    # 5. 替换后 NaN 缺失率排行
    nan_counts = accidents_df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
    if len(nan_counts) > 0:
        logger.info(f"  NaN rates after -1 replacement ({len(nan_counts)} columns):")
        for col, count in nan_counts.items():
            pct = count / len(accidents_df) * 100
            logger.info(f"    {col}: {count:,} ({pct:.2f}%)")

    # 6. Speed_limit 取值分布
    if 'Speed_limit' in accidents_df.columns:
        speed_dist = accidents_df['Speed_limit'].value_counts().sort_index()
        logger.info(f"  Speed_limit distribution:")
        for speed, count in speed_dist.items():
            pct = count / len(accidents_df) * 100
            logger.info(f"    {speed} mph: {count:,} ({pct:.2f}%)")

    # 7. 数值列统计
    numeric_desc = accidents_df.describe()
    logger.info(f"  Numeric columns describe():\n{numeric_desc.to_string()}")

    # 8. 保存 EDA 汇总到 CSV
    eda_summary = pd.DataFrame({
        'metric': [
            'total_rows', 'total_columns',
            'fatal_count', 'serious_count', 'slight_count',
            'fatal_pct', 'serious_pct', 'slight_pct',
            'columns_with_nan', 'year_min', 'year_max',
        ],
        'value': [
            accidents_df.shape[0], accidents_df.shape[1],
            int(severity_counts.get(1, 0)), int(severity_counts.get(2, 0)),
            int(severity_counts.get(3, 0)),
            round(severity_pcts.get(1, 0), 2), round(severity_pcts.get(2, 0), 2),
            round(severity_pcts.get(3, 0), 2),
            len(nan_counts),
            int(accidents_df['Year'].min()) if 'Year' in accidents_df.columns else 'N/A',
            int(accidents_df['Year'].max()) if 'Year' in accidents_df.columns else 'N/A',
        ]
    })

    save_path = os.path.join(RESULTS_DIR, 'Table00_eda_initial.csv')
    eda_summary.to_csv(save_path, index=False)
    logger.info(f"  EDA summary saved to {save_path}")


# 阶段 1：加载原始数据与上下文表
def phase_01_load_data():
    logger.info("=" * 60)
    logger.info("PHASE 1: Loading raw data")
    logger.info("=" * 60)

    accidents_df, minus_one_counts = load_accidents(config.ACCIDENTS_FILE)
    vehicles_df = load_vehicles(config.VEHICLES_FILE)

    accidents_df = parse_datetime(accidents_df)

    context_tables = load_context_tables(config.CONTEXT_DIR)

    logger.info(f"  Accidents: {accidents_df.shape}")
    logger.info(f"  Vehicles:  {vehicles_df.shape}")
    logger.info(f"  Context tables loaded: {len(context_tables)}")

    return accidents_df, vehicles_df, context_tables, minus_one_counts


# 阶段 2：缺失值与泄漏处理
def phase_02_clean_data(accidents_df):
    logger.info("=" * 60)
    logger.info("PHASE 2: Data cleaning")
    logger.info("=" * 60)

    df_before = accidents_df

    missing_report = identify_missing_patterns(accidents_df)
    logger.info(f"  Missing value report generated ({len(missing_report)} columns analyzed)")

    cleaned_df, dropped_cols = drop_high_missing_columns(accidents_df)
    logger.info(f"  Dropped {len(dropped_cols)} high-missing columns: {dropped_cols}")

    cleaned_df = remove_leakage_columns(cleaned_df)

    cleaned_df = clip_outliers(cleaned_df)

    cleaning_report = generate_cleaning_report(df_before, cleaned_df)

    logger.info(f"  Cleaned shape: {cleaned_df.shape}")
    return cleaned_df


# 阶段 3：车辆表聚合并与事故表合并
def phase_03_aggregate_vehicles(cleaned_df, vehicles_df):
    logger.info("=" * 60)
    logger.info("PHASE 3: Vehicle aggregation and merge")
    logger.info("=" * 60)

    vehicles_agg = aggregate_vehicles(vehicles_df)
    merged_df = merge_accidents_vehicles(cleaned_df, vehicles_agg)

    logger.info(f"  Vehicle features: {list(vehicles_agg.columns)}")
    logger.info(f"  Merged shape: {merged_df.shape}")
    return merged_df


# 阶段 4：特征工程（与标签无关）
def phase_04_feature_engineering(merged_df, context_tables):
    logger.info("=" * 60)
    logger.info("PHASE 4: Feature engineering (label-agnostic)")
    logger.info("=" * 60)

    featured_df = create_temporal_features(merged_df)
    featured_df = create_binary_features(featured_df, context_tables)
    featured_df = create_interaction_features(featured_df)

    logger.info(f"  Featured shape: {featured_df.shape}")
    return featured_df


# 阶段 5：分层训练/测试切分
def phase_05_train_test_split(featured_df):
    logger.info("=" * 60)
    logger.info("PHASE 5: Train/test split (80/20 stratified)")
    logger.info("=" * 60)

    target = 'Accident_Severity'
    X = featured_df.drop(columns=[target])
    y = featured_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED
    )

    logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"  Train class distribution:\n{y_train.value_counts(normalize=True)}")
    logger.info(f"  Test class distribution:\n{y_test.value_counts(normalize=True)}")

    return X_train, X_test, y_train, y_test


# 阶段 6：不平衡采样（仅训练集）
def phase_06_asymmetric_sampling(X_train, y_train):
    logger.info("=" * 60)
    logger.info("PHASE 6: Asymmetric undersampling (training set only)")
    logger.info("=" * 60)

    train_df = pd.concat([X_train, y_train], axis=1)
    train_sampled = asymmetric_undersample(
        train_df,
        slight_ratio=config.SLIGHT_SAMPLE_RATIO,
        random_state=config.RANDOM_SEED
    )

    target = 'Accident_Severity'
    X_train_sampled = train_sampled.drop(columns=[target])
    y_train_sampled = train_sampled[target]

    logger.info(f"  Before sampling: {len(X_train)}")
    logger.info(f"  After sampling:  {len(X_train_sampled)}")
    logger.info(f"  Sampled distribution:\n{y_train_sampled.value_counts()}")

    return X_train_sampled, y_train_sampled


# 阶段 7：地区频率编码
def phase_07_frequency_encoding(X_train, X_test):
    logger.info("=" * 60)
    logger.info("PHASE 7: Frequency encoding (Local Authority)")
    logger.info("=" * 60)

    X_train_encoded, X_test_encoded, la_freq_map = frequency_encode_local_authority(
        X_train, X_test
    )

    logger.info("  Local_Authority frequency encoding applied")
    logger.info(f"  LA_Frequency range: [{X_train_encoded['LA_Frequency'].min():.6f}, "
                f"{X_train_encoded['LA_Frequency'].max():.6f}]")
    return X_train_encoded, X_test_encoded, la_freq_map


# 阶段 8：聚类探索分析与可视化
def phase_08_clustering_eda(X_train, y_train, context_tables):
    logger.info("=" * 60)
    logger.info("PHASE 8: Clustering EDA")
    logger.info("=" * 60)

    import os
    OPTIMAL_K = 4

    # 步骤1：为 K 搜索准备子样本
    scaled_sample, _, sample_idx_sub = prepare_clustering_data(
        X_train,
        feature_cols=config.CLUSTER_FEATURES,
        sample_size=config.CLUSTER_SAMPLE_SIZE,
        random_state=config.RANDOM_SEED
    )

    # 步骤2：用肘部法和轮廓系数寻找 K（基于子样本）
    k_results = find_optimal_k(
        scaled_sample,
        k_range=config.CLUSTER_K_RANGE,
        random_state=config.RANDOM_SEED
    )

    # 生成肘部法与轮廓系数图（Fig04, Fig05）
    plot_elbow(k_results,
               save_path=os.path.join(config.FIGURES_DIR, 'Fig04_elbow_plot.png'))
    plot_silhouette(k_results,
                    save_path=os.path.join(config.FIGURES_DIR, 'Fig05_silhouette_scores.png'))

    # 步骤3：在完整训练集上重新拟合 K=4
    # 说明：子样本仅用于快速选 K，最终需覆盖全量训练样本
    # 以便进行画像分析和第10阶段的 Cluster_ID 提升实验
    full_scaled, cluster_scaler, full_idx = prepare_clustering_data(
        X_train,
        feature_cols=config.CLUSTER_FEATURES,
        sample_size=None,  # 不抽样，使用全部有效行
        random_state=config.RANDOM_SEED
    )
    kmeans_model, kmeans_labels = fit_kmeans(full_scaled, OPTIMAL_K, config.RANDOM_SEED)

    # 对齐原始数据用于画像与严重度分析
    X_full = X_train.iloc[full_idx]
    y_full = y_train.iloc[full_idx]

    # 步骤4：聚类画像与严重度交叉表（全量训练集）
    profiles = compute_cluster_profiles(X_full, kmeans_labels, config.CLUSTER_FEATURES)
    ct_counts, ct_props = cluster_severity_crosstab(y_full, kmeans_labels)

    # 保存最终表格
    profiles.to_csv(os.path.join(config.RESULTS_DIR, 'Table_cluster_profiles_K4_final.csv'))
    ct_props.to_csv(os.path.join(config.RESULTS_DIR, 'Table_cluster_severity_K4_final.csv'))

    # 步骤5：PCA 二维可视化（为速度进行子采样）
    VIZ_SAMPLE = 80000
    rng = np.random.RandomState(config.RANDOM_SEED)
    if len(full_scaled) > VIZ_SAMPLE:
        viz_idx = rng.choice(len(full_scaled), size=VIZ_SAMPLE, replace=False)
        viz_scaled = full_scaled[viz_idx]
        viz_labels = kmeans_labels[viz_idx]
    else:
        viz_scaled = full_scaled
        viz_labels = kmeans_labels

    pca_2d, var_ratio, _ = compute_pca_2d(viz_scaled, config.RANDOM_SEED)
    plot_pca_clusters(pca_2d, viz_labels, var_ratio,
                      save_path=os.path.join(config.FIGURES_DIR, 'Fig06_pca_clusters.png'))

    # 生成画像热图与严重度柱状图（Fig07, Fig08）
    plot_cluster_profiles(profiles,
                          save_path=os.path.join(config.FIGURES_DIR, 'Fig07_cluster_profiles_heatmap.png'))
    plot_cluster_severity(ct_counts, ct_props,
                          save_path=os.path.join(config.FIGURES_DIR, 'Fig08_cluster_severity_crosstab.png'))

    # 步骤6：层次聚类对比（使用较小子样本降低内存）
    AGG_SAMPLE_SIZE = 20000
    agg_scaled, _, agg_idx = prepare_clustering_data(
        X_train, config.CLUSTER_FEATURES, AGG_SAMPLE_SIZE, config.RANDOM_SEED)

    _, km_agg_labels = fit_kmeans(agg_scaled, OPTIMAL_K, config.RANDOM_SEED)
    _, agg_labels = fit_agglomerative(agg_scaled, OPTIMAL_K)
    ari = adjusted_rand_score(km_agg_labels, agg_labels)
    logger.info(f"  K-Means vs Agglomerative ARI (K={OPTIMAL_K}): {ari:.4f}")

    # 步骤7：树状图（抽样 5K 便于计算）
    plot_dendrogram(agg_scaled, max_samples=5000, random_state=config.RANDOM_SEED,
                    save_path=os.path.join(config.FIGURES_DIR, 'Fig09_dendrogram.png'))

    logger.info(f"  Phase 8 complete: K={OPTIMAL_K}, {len(kmeans_labels):,} samples clustered")
    return OPTIMAL_K, kmeans_model, kmeans_labels, cluster_scaler


# 阶段 9：基线模型训练与评估
def phase_09_baseline_models(X_train, y_train, X_test, y_test):
    logger.info("=" * 60)
    logger.info("PHASE 9: Baseline models (no Cluster_ID)")
    logger.info("=" * 60)

    model_configs = get_model_configs(include_cluster=False)
    baseline_results = {}

    for name, cfg in model_configs.items():
        logger.info(f"  Training {name}...")
        pipeline = cfg['pipeline']
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, 'predict_proba') else None
        metrics = compute_metrics(y_test, y_pred, y_proba,
                                  class_names=['Fatal', 'Serious', 'Slight'])
        baseline_results[name] = {'metrics': metrics, 'pipeline': pipeline}
        logger.info(f"    Macro F1: {metrics.get('macro_f1', 'N/A'):.4f}")

    return baseline_results


# 阶段 10：加入 Cluster_ID 的模型评估
def phase_10_cluster_enhanced_models(X_train, y_train, X_test, y_test):
    logger.info("=" * 60)
    logger.info("PHASE 10: Cluster-enhanced models (with Cluster_ID)")
    logger.info("=" * 60)

    model_configs = get_model_configs(include_cluster=True)
    enhanced_results = {}

    for name, cfg in model_configs.items():
        logger.info(f"  Training {name} + Cluster_ID...")
        pipeline = cfg['pipeline']
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, 'predict_proba') else None
        metrics = compute_metrics(y_test, y_pred, y_proba,
                                  class_names=['Fatal', 'Serious', 'Slight'])
        enhanced_results[name] = {'metrics': metrics, 'pipeline': pipeline}
        logger.info(f"    Macro F1: {metrics.get('macro_f1', 'N/A'):.4f}")

    return enhanced_results


# 阶段 11：消融与调参（含缓存判断）
def phase_11_tuning_and_ablation(X_train, y_train, X_test, y_test):
    import json

    logger.info("=" * 60)
    logger.info("PHASE 11: Hyperparameter tuning & imbalance ablation")
    logger.info("=" * 60)

    # 消融：检查缓存
    ablation_csv = os.path.join(config.RESULTS_DIR, 'Table_ablation.csv')
    ablation_df = None
    if os.path.exists(ablation_csv):
        logger.info(f"  Ablation: loading cached results from {ablation_csv}")
        ablation_df = pd.read_csv(ablation_csv)
        for scheme in sorted(ablation_df['scheme'].unique()):
            subset = ablation_df[ablation_df['scheme'] == scheme]
            for _, row in subset.iterrows():
                logger.info(f"    Scheme {scheme} | {row['model']}: "
                            f"CV Macro F1 = {row['cv_macro_f1_mean']:.4f}")
    else:
        logger.info("  Ablation: no cache found, running (this takes ~3 hours)...")
        model_configs = get_model_configs(include_cluster=False)
        ablation_results = imbalance_ablation(
            model_configs, X_train, y_train, X_test, y_test,
            slight_ratio=config.SLIGHT_SAMPLE_RATIO,
            random_state=config.RANDOM_SEED
        )
        # imbalance_ablation 会内部保存 Table_ablation.csv
        if os.path.exists(ablation_csv):
            ablation_df = pd.read_csv(ablation_csv)

    # 调参：检查缓存
    tuning_ckpt_path = os.path.join(config.RESULTS_DIR, 'tuning_checkpoint.json')
    grid_results = {}
    tuning_cached = False

    if os.path.exists(tuning_ckpt_path):
        with open(tuning_ckpt_path) as f:
            ckpt = json.load(f)
        completed = ckpt.get('completed', {})
        expected_models = {'LogisticRegression', 'HistGBT', 'RandomForest'}
        if expected_models.issubset(completed.keys()):
            tuning_cached = True
            logger.info(f"  Tuning: loading cached results from {tuning_ckpt_path}")
            for name, info in completed.items():
                logger.info(f"    {name}: best CV Macro F1 = {info['best_score']:.4f}, "
                            f"params = {info.get('best_params', 'N/A')}")

    if not tuning_cached:
        logger.info("  Tuning: no complete cache found, running GridSearchCV "
                     "(this takes ~9 hours)...")
        model_configs = get_model_configs(include_cluster=False)
        for name, cfg in model_configs.items():
            logger.info(f"    Tuning {name}...")
            gs, best_params, best_score = run_gridsearch(
                cfg['pipeline'], cfg['param_grid'],
                X_train, y_train,
                cv_folds=config.CV_FOLDS,
                random_state=config.RANDOM_SEED
            )
            grid_results[name] = gs
            logger.info(f"      Best params: {best_params}")
            logger.info(f"      Best CV Macro F1: {best_score:.4f}")

    return grid_results, ablation_df


# 阶段 12：最终评估（读取缓存或触发独立脚本）
def phase_12_final_evaluation(X_train, y_train, X_test, y_test,
                              featured_df, baseline_results, enhanced_results,
                              grid_results, kmeans_labels):
    logger.info("=" * 60)
    logger.info("PHASE 12: Final evaluation")
    logger.info("=" * 60)

    required_files = [
        'Table_test_results.csv',
        'Table_cv_vs_test_vs_temporal.csv',
        'Table_cluster_id_comparison.csv',
        'Table_cluster_error_analysis.csv',
    ]
    all_cached = all(
        os.path.exists(os.path.join(config.RESULTS_DIR, f))
        for f in required_files
    )

    if all_cached:
        logger.info("  Phase 12: Loading cached evaluation results...")

        # L2 测试集结果
        test_df = pd.read_csv(os.path.join(config.RESULTS_DIR, 'Table_test_results.csv'))
        logger.info("  L2 Test Set Results (tuned models, Scheme C):")
        for _, row in test_df.iterrows():
            # 来自 run_phase12.py 的列：Model, accuracy, macro_f1, Fatal_recall
            model = row.iloc[0]  # First column is model name
            logger.info(f"    {model}: Accuracy={row['accuracy']:.4f}, "
                        f"Macro F1={row['macro_f1']:.4f}, "
                        f"Fatal Recall={row['Fatal_recall']:.4f}")

        # L3 三层对比
        cv_test_temp = pd.read_csv(
            os.path.join(config.RESULTS_DIR, 'Table_cv_vs_test_vs_temporal.csv'))
        logger.info("  L1/L2/L3 Three-Layer Comparison (Macro F1):")
        for _, row in cv_test_temp.iterrows():
            # 列：model, CV_macro_f1, test_macro_f1, temporal_macro_f1
            model = row.iloc[0]
            logger.info(f"    {model}: CV={row['CV_macro_f1']:.4f}, "
                        f"Test={row['test_macro_f1']:.4f}, "
                        f"Temporal={row['temporal_macro_f1']:.4f}")

        # Cluster_ID 对比
        cluster_cmp = pd.read_csv(
            os.path.join(config.RESULTS_DIR, 'Table_cluster_id_comparison.csv'))
        logger.info("  Cluster_ID Uplift:")
        for _, row in cluster_cmp.iterrows():
            model = row.iloc[0]
            logger.info(f"    {model}: ΔMacro F1 = {row['delta_macro_f1']:+.4f}")

        # 聚类错误分析
        logger.info("  Cluster × Error analysis loaded from "
                     "Table_cluster_error_analysis.csv")
    else:
        missing = [f for f in required_files
                   if not os.path.exists(os.path.join(config.RESULTS_DIR, f))]
        logger.info(f"  Phase 12: Missing cached files: {missing}")
        logger.info("  Running full evaluation via run_phase12...")
        try:
            import subprocess
            result = subprocess.run(
                ['python', 'run_phase12.py'],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True, text=True, timeout=1800
            )
            if result.returncode == 0:
                logger.info("  run_phase12.py completed successfully")
            else:
                logger.error(f"  run_phase12.py failed: {result.stderr[-500:]}")
        except Exception as e:
            logger.error(f"  Failed to run Phase 12: {e}")
            logger.info("  Run `python run_phase12.py` manually to generate results")

    logger.info("  Phase 12 complete")


# 阶段 13：检查与汇总图表产物
def phase_13_generate_figures():
    logger.info("=" * 60)
    logger.info("PHASE 13: Generating all figures and tables")
    logger.info("=" * 60)

    # 预期图表清单
    expected_figures = [
        # EDA（Phase 8 / run_phase12.py）
        'Fig01_target_distribution.png',
        'Fig02_correlation_heatmap.png',
        'Fig03_feature_severity_dist.png',
        # 聚类（Phase 8）
        'Fig04_elbow_plot.png',
        'Fig05_silhouette_scores.png',
        'Fig06_pca_clusters.png',
        'Fig07_cluster_profiles_heatmap.png',
        'Fig08_cluster_severity_crosstab.png',
        'Fig09_dendrogram.png',
        # 监督模型（run_phase12.py）
        'Fig10_confusion_matrix_RandomForest.png',
        'Fig10_confusion_matrix_HistGBT.png',
        'Fig10_confusion_matrix_LogisticRegression.png',
        'Fig11_model_comparison.png',
        'Fig12_roc_curves.png',
        'Fig13_pr_curves.png',
        'Fig14_feature_importance_RandomForest.png',
        'Fig14_feature_importance_HistGBT.png',
        'Fig15_lr_coefficients.png',
        # 消融（run_ablation.py / run_phase12.py）
        'Fig16_ablation_comparison.png',
        'Fig17_ablation_tradeoff.png',
        'Fig18_ablation_cv_stability.png',
        # 调参（run_tuning.py / run_phase12.py）
        'Fig19_tuning_sensitivity_RandomForest.png',
        'Fig19_tuning_sensitivity_HistGBT.png',
        'Fig19_tuning_sensitivity_LogisticRegression.png',
        'Fig20_tuning_cv_boxplot.png',
        'Fig21_tuning_before_after.png',
        # 聚类错误分析
        'Fig_cluster_error_distribution.png',
    ]

    existing = [f for f in expected_figures
                if os.path.exists(os.path.join(config.FIGURES_DIR, f))]
    missing = [f for f in expected_figures
               if not os.path.exists(os.path.join(config.FIGURES_DIR, f))]

    if not missing:
        logger.info(f"  All {len(expected_figures)} figures already generated, "
                     "skipping regeneration")
    else:
        logger.warning(f"  {len(missing)} of {len(expected_figures)} figures missing:")
        for f in missing:
            logger.warning(f"    - {f}")
        logger.info("  To regenerate, run: python run_phase12.py")
        logger.info("  (Figure generation requires fitted model objects from "
                     "tuning checkpoints)")

    # 结果表检查
    expected_tables = [
        'Table_test_results.csv',
        'Table_ablation.csv',
        'Table_cv_vs_test_vs_temporal.csv',
        'Table_cluster_id_comparison.csv',
        'Table_cluster_error_analysis.csv',
        'Table_tuning_improvement.csv',
        'Table_tuning_best_params.csv',
        'Table_baseline_comparison.csv',
        'Table_fatal_misclass_pattern.csv',
    ]
    existing_tables = [t for t in expected_tables
                       if os.path.exists(os.path.join(config.RESULTS_DIR, t))]
    logger.info(f"  Result tables: {len(existing_tables)}/{len(expected_tables)} present")

    logger.info("  Phase 13 complete")


# 主入口：按顺序执行完整流水线
def main():
    """执行完整流水线。"""
    start_time = time.time()
    logger.info("Starting UK Car Accidents Severity Classification Pipeline")
    logger.info(f"Config: FULL_SCALE_RUN={config.FULL_SCALE_RUN}, SEED={config.RANDOM_SEED}")

    # 阶段1：数据加载（与标签无关）
    accidents_df, vehicles_df, context_tables, minus_one_counts = phase_01_load_data()

    run_initial_eda(accidents_df, minus_one_counts, context_tables)

    # 阶段2-3：车辆聚合后清洗（与标签无关）
    merged_df = phase_03_aggregate_vehicles(accidents_df, vehicles_df)
    cleaned_df = phase_02_clean_data(merged_df)

    # 阶段4：特征工程（与标签无关）
    featured_df = phase_04_feature_engineering(cleaned_df, context_tables)

    # 阶段5：训练测试切分（无关与有关的分界）
    X_train, X_test, y_train, y_test = phase_05_train_test_split(featured_df)

    X_train_full = X_train.copy()
    y_train_full = y_train.copy()

    # 阶段6-7：与标签相关的处理（仅训练集）
    X_train_sampled, y_train_sampled = phase_06_asymmetric_sampling(X_train, y_train)
    X_train_encoded, X_test_encoded, la_freq_map = phase_07_frequency_encoding(
        X_train_sampled, X_test
    )

    la_col = 'Local_Authority_(District)'
    X_train_full_enc = X_train_full.copy()
    if la_col in X_train_full_enc.columns:
        global_mean_freq = 1.0 / len(la_freq_map)
        X_train_full_enc['LA_Frequency'] = X_train_full_enc[la_col].map(
            la_freq_map).fillna(global_mean_freq)
        X_train_full_enc = X_train_full_enc.drop(columns=[la_col])
        logger.info(f"  Applied freq_map to un-sampled training data: {X_train_full_enc.shape}")

    # 序列化 Phase7 输出供 run_ablation.py 和 run_tuning.py 使用
    import pickle
    data_bundle = {
        'X_train_sampled': X_train_encoded,
        'y_train_sampled': y_train_sampled,
        'X_train_full_enc': X_train_full_enc,
        'y_train_full': y_train_full,
        'X_test': X_test_encoded,
        'y_test': y_test,
        'la_freq_map': la_freq_map,
    }
    pkl_path = os.path.join(config.RESULTS_DIR, 'phase7_data.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(data_bundle, f)
    logger.info(f"  Phase 7 data serialized to {pkl_path}")

    # 阶段8：聚类 EDA（探索性，K=4 已确认）
    optimal_k, kmeans_model, kmeans_labels, cluster_scaler = phase_08_clustering_eda(
        X_train_encoded, y_train_sampled, context_tables
    )

    # 阶段9-10：监督模型
    baseline_results = phase_09_baseline_models(
        X_train_encoded, y_train_sampled, X_test_encoded, y_test
    )
    enhanced_results = phase_10_cluster_enhanced_models(
        X_train_encoded, y_train_sampled, X_test_encoded, y_test
    )

    # 阶段11：调参与消融
    grid_results, ablation_results = phase_11_tuning_and_ablation(
        X_train_encoded, y_train_sampled, X_test_encoded, y_test
    )

    # 阶段12：最终评估
    phase_12_final_evaluation(
        X_train_encoded, y_train_sampled, X_test_encoded, y_test,
        featured_df, baseline_results, enhanced_results,
        grid_results, kmeans_labels
    )

    # 阶段13：生成全部图表
    phase_13_generate_figures()

    elapsed = time.time() - start_time
    logger.info(f"Pipeline complete in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
