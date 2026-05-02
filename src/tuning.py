"""
超参数调参模块 - 流水线第11阶段。
职责:
1. 通过 5 折 CV 做不平衡消融（A/B/C 三方案）
2. 为各模型执行 GridSearchCV 或 RandomizedSearchCV
3. 生成调参可视化（消融对比、敏感性曲线、CV 箱线图）
4. 提取并汇报最佳参数
"""
import os
import logging
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import f1_score

from src.evaluation import compute_metrics
import config

logger = logging.getLogger(__name__)


# 运行不平衡方案消融实验
def imbalance_ablation(model_configs, X_train, y_train, X_test=None, y_test=None,
                       X_train_full=None, y_train_full=None,
                       slight_ratio=0.4, random_state=42):
    logger.info("  Running imbalance ablation (3 schemes × 3 models × 5-fold CV)...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results = []

    schemes = {
        'A': {'data': (X_train_full, y_train_full), 'class_weight': 'balanced',
               'desc': 'class_weight only (no sampling)'},
        'B': {'data': (X_train, y_train), 'class_weight': None,
               'desc': 'sampling only (no class_weight)'},
        'C': {'data': (X_train, y_train), 'class_weight': 'balanced',
               'desc': 'both sampling + class_weight'},
    }

    for scheme_id, scheme_cfg in schemes.items():
        X_s, y_s = scheme_cfg['data']
        cw = scheme_cfg['class_weight']
        logger.info(f"\n  === Scheme {scheme_id}: {scheme_cfg['desc']} "
                     f"({len(X_s):,} rows) ===")

        for name, cfg in model_configs.items():
            t_start = time.time()
            logger.info(f"    {name} (class_weight={cw})...")

            # 克隆流水线并设置该方案的 class_weight
            pipeline = clone(cfg['pipeline'])
            pipeline.set_params(model__class_weight=cw)

            # 手动遍历折，单折只拟合一次
            fold_scores = []
            all_y_true, all_y_pred, all_y_proba = [], [], []

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_s, y_s)):
                X_fold_train = X_s.iloc[train_idx]
                X_fold_val = X_s.iloc[val_idx]
                y_fold_train = y_s.iloc[train_idx]
                y_fold_val = y_s.iloc[val_idx]

                # 每个折仅拟合一次
                fold_pipeline = clone(pipeline)
                fold_pipeline.fit(X_fold_train, y_fold_train)

                # 使用同一模型预测并计算概率（避免二次训练）
                y_pred_fold = fold_pipeline.predict(X_fold_val)
                try:
                    y_proba_fold = fold_pipeline.predict_proba(X_fold_val)
                except Exception:
                    y_proba_fold = None

                fold_f1 = f1_score(y_fold_val, y_pred_fold, average='macro')
                fold_scores.append(fold_f1)

                all_y_true.extend(y_fold_val.values)
                all_y_pred.extend(y_pred_fold)
                if y_proba_fold is not None:
                    all_y_proba.append(y_proba_fold)

                logger.info(f"      Fold {fold_idx+1}/5: Macro F1 = {fold_f1:.4f}")

            # 汇总折外预测指标
            y_proba_concat = np.vstack(all_y_proba) if all_y_proba else None
            metrics = compute_metrics(
                np.array(all_y_true), np.array(all_y_pred), y_proba_concat
            )
            metrics['scheme'] = scheme_id
            metrics['model'] = name
            metrics['cv_macro_f1_mean'] = np.mean(fold_scores)
            metrics['cv_macro_f1_std'] = np.std(fold_scores)
            # 将折分数保存为逗号分隔字符串便于 CSV
            metrics['fold_scores'] = ','.join(f'{s:.4f}' for s in fold_scores)

            elapsed = time.time() - t_start
            logger.info(f"      Done: CV Macro F1 = {metrics['cv_macro_f1_mean']:.4f} "
                         f"± {metrics['cv_macro_f1_std']:.4f}, "
                         f"Fatal Recall = {metrics.get('Fatal_recall', 0):.4f} ({elapsed:.1f}s)")

            results.append(metrics)

    results_df = pd.DataFrame(results)
    return results_df


# 执行 GridSearchCV 并返回最优结果
def run_gridsearch(pipeline, param_grid, X_train, y_train,
                   cv_folds=5, scoring='f1_macro', random_state=42, n_jobs=-1):
    logger.info(f"    Running GridSearchCV: {len(param_grid)} param groups, "
                f"{cv_folds}-fold CV, scoring={scoring}...")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # 统计参数组合数
    from itertools import product
    total_combos = 1
    for values in param_grid.values():
        total_combos *= len(values)
    total_fits = total_combos * cv_folds
    logger.info(f"    {total_combos} parameter combinations × {cv_folds} folds = {total_fits} fits")

    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
        verbose=1,
    )

    t_start = time.time()
    gs.fit(X_train, y_train)
    elapsed = time.time() - t_start

    logger.info(f"    Best score: {gs.best_score_:.4f}")
    logger.info(f"    Best params: {gs.best_params_}")
    logger.info(f"    Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # 检查最优值是否在边界
    for param_name, values in param_grid.items():
        best_val = gs.best_params_[param_name]
        if best_val == values[0] or best_val == values[-1]:
            if best_val is not None:  # None 是 max_depth 的合法取值
                logger.warning(f"    ⚠️ {param_name}={best_val} is at grid boundary! "
                               f"Consider expanding range.")

    return gs, gs.best_params_, gs.best_score_


# 汇总最优参数表
def summarize_best_params(grid_results_dict):
    rows = []
    for name, gs in grid_results_dict.items():
        row = {'model': name, 'best_cv_score': gs.best_score_}
        row.update(gs.best_params_)
        rows.append(row)

    return pd.DataFrame(rows)


# 绘制消融对比柱状图
def plot_ablation_comparison(ablation_df, save_path=None):
    """分组柱状图：按模型对比 A/B/C 方案（Fig16）。
    左图为 Macro F1，右图为 Fatal Recall。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    metrics_to_plot = [('cv_macro_f1_mean', 'Macro F1 (CV Mean)'),
                       ('Fatal_recall', 'Fatal Recall')]
    scheme_colors = {'A': '#2196F3', 'B': '#FF9800', 'C': '#4CAF50'}

    for ax, (metric, title) in zip(axes, metrics_to_plot):
        models = ablation_df['model'].unique()
        x = np.arange(len(models))
        width = 0.25

        for i, scheme in enumerate(['A', 'B', 'C']):
            vals = ablation_df[ablation_df['scheme'] == scheme].set_index('model')[metric]
            vals = vals.reindex(models)
            bars = ax.bar(x + i * width, vals, width, label=f'Scheme {scheme}',
                          color=scheme_colors[scheme], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x + width)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Imbalance Ablation: Scheme Comparison', fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"    Saved: {save_path}")
    plt.close(fig)
    return fig


# 绘制致死召回与精度权衡图
def plot_ablation_tradeoff(ablation_df, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 7))
    scheme_colors = {'A': '#2196F3', 'B': '#FF9800', 'C': '#4CAF50'}
    model_markers = {'RandomForest': 'o', 'HistGBT': 's', 'LogisticRegression': '^'}

    for _, row in ablation_df.iterrows():
        ax.scatter(row['Fatal_recall'], row['Fatal_precision'],
                   c=scheme_colors[row['scheme']],
                   marker=model_markers.get(row['model'], 'o'),
                   s=150, edgecolors='black', linewidth=0.5, zorder=3)
        ax.annotate(f"{row['scheme']}-{row['model'][:2]}",
                    (row['Fatal_recall'], row['Fatal_precision']),
                    fontsize=7, ha='left', va='bottom',
                    xytext=(5, 5), textcoords='offset points')

    # 图例
    for scheme, color in scheme_colors.items():
        ax.scatter([], [], c=color, s=100, label=f'Scheme {scheme}')
    for model, marker in model_markers.items():
        ax.scatter([], [], c='gray', marker=marker, s=100, label=model)
    ax.legend(fontsize=9, loc='upper right')

    ax.set_xlabel('Fatal Recall', fontsize=12)
    ax.set_ylabel('Fatal Precision', fontsize=12)
    ax.set_title('Fatal Recall vs Precision Trade-off (Analysis Thread #2)', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"    Saved: {save_path}")
    plt.close(fig)
    return fig


# 绘制 CV 稳定性箱线图
def plot_ablation_cv_stability(ablation_df, save_path=None):
    fig, ax = plt.subplots(figsize=(14, 6))

    # 将 fold_scores 字符串解析为列表
    box_data = []
    labels = []
    for _, row in ablation_df.iterrows():
        scores = [float(s) for s in row['fold_scores'].split(',')]
        box_data.append(scores)
        labels.append(f"{row['scheme']}-{row['model'][:3]}")

    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)

    # 按方案配色
    scheme_colors = {'A': '#2196F3', 'B': '#FF9800', 'C': '#4CAF50'}
    for i, label in enumerate(labels):
        scheme = label[0]
        bp['boxes'][i].set_facecolor(scheme_colors.get(scheme, 'gray'))
        bp['boxes'][i].set_alpha(0.6)

    ax.set_ylabel('Macro F1 (per fold)', fontsize=12)
    ax.set_title('CV Score Stability Across Schemes and Models', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"    Saved: {save_path}")
    plt.close(fig)
    return fig


# 绘制调参敏感性曲线
def plot_tuning_sensitivity(gs, model_name, save_path=None):
    results = pd.DataFrame(gs.cv_results_)
    param_cols = [c for c in results.columns if c.startswith('param_model__')]

    n_params = len(param_cols)
    if n_params == 0:
        return None

    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5))
    if n_params == 1:
        axes = [axes]

    for ax, param_col in zip(axes, param_cols):
        param_name = param_col.replace('param_', '')
        grouped = results.groupby(param_col)['mean_test_score'].agg(['mean', 'std'])
        grouped = grouped.reset_index()

        ax.errorbar(range(len(grouped)), grouped['mean'], yerr=grouped['std'],
                    fmt='o-', capsize=5, linewidth=2, markersize=8)
        ax.set_xticks(range(len(grouped)))
        ax.set_xticklabels([str(v) for v in grouped[param_col]], fontsize=9)
        ax.set_xlabel(param_name.replace('model__', ''), fontsize=11)
        ax.set_ylabel('Mean CV Macro F1', fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Hyperparameter Sensitivity — {model_name}', fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"    Saved: {save_path}")
    plt.close(fig)
    return fig


# 绘制调参后 CV 分布箱线图
def plot_tuning_cv_boxplot(grid_results_dict, save_path=None):
    """箱线图：调参后 CV 分数分布（Fig20）。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    box_data = []
    labels = []
    for name, gs in grid_results_dict.items():
        # 取最佳配置的各折分数
        best_idx = gs.best_index_
        fold_scores = [gs.cv_results_[f'split{i}_test_score'][best_idx]
                       for i in range(5)]
        box_data.append(fold_scores)
        labels.append(name)

    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i % len(colors)])
        box.set_alpha(0.6)

    ax.set_ylabel('Macro F1 (per CV fold)', fontsize=12)
    ax.set_title('Tuned Model CV Score Distribution', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"    Saved: {save_path}")
    plt.close(fig)
    return fig


# 绘制调参前后对比图
def plot_tuning_before_after(default_metrics, tuned_metrics, save_path=None):
    """分组柱状图：默认参数与调参后对比（Fig21）。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    metrics_to_plot = [('macro_f1', 'Macro F1'), ('Fatal_recall', 'Fatal Recall')]

    models = list(default_metrics.keys())
    x = np.arange(len(models))
    width = 0.35

    for ax, (metric, title) in zip(axes, metrics_to_plot):
        default_vals = [default_metrics[m].get(metric, 0) for m in models]
        tuned_vals = [tuned_metrics[m].get(metric, 0) for m in models]

        ax.bar(x - width / 2, default_vals, width, label='Default', color='#90CAF9', alpha=0.85)
        ax.bar(x + width / 2, tuned_vals, width, label='Tuned', color='#1565C0', alpha=0.85)

        for i, (d, t) in enumerate(zip(default_vals, tuned_vals)):
            ax.text(x[i] - width / 2, d + 0.005, f'{d:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(x[i] + width / 2, t + 0.005, f'{t:.3f}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Hyperparameter Tuning: Before vs After', fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"    Saved: {save_path}")
    plt.close(fig)
    return fig
