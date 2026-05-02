"""
评估模块 - 流水线第9至第13阶段。
职责:
1. 计算模型指标（Accuracy、Balanced Accuracy、Macro F1、Weighted F1、各类别 Precision/Recall/F1、AUC-ROC、PR-AUC）
2. 生成混淆矩阵（3x3 热图）
3. 生成 ROC 与 PR 曲线（按类别与对比）
4. 生成特征重要性图（RF、HistGBT）和 LR 系数图
5. 模型对比柱状图与汇总表
6. 聚类与预测错误交叉分析（Phase 12）
7. 时序留出评估（Phase 12，L3）
"""

# 模型评估与可视化工具

import os
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import label_binarize

import config

logger = logging.getLogger(__name__)

# 严重度编码与标签映射
SEVERITY_LABELS = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
SEVERITY_CLASSES = [1, 2, 3]
SEVERITY_NAMES = ['Fatal', 'Serious', 'Slight']


# 计算整体与分级评估指标
def compute_metrics(y_true, y_pred, y_proba=None, class_names=None):
    if class_names is None:
        class_names = SEVERITY_NAMES

    metrics = {}

    # 整体指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')

    # 分类别指标
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=SEVERITY_CLASSES, zero_division=0
    )
    for i, name in enumerate(class_names):
        metrics[f'{name}_precision'] = precision[i]
        metrics[f'{name}_recall'] = recall[i]
        metrics[f'{name}_f1'] = f1[i]
        metrics[f'{name}_support'] = int(support[i])

    # 概率指标（若提供概率）
    if y_proba is not None:
        try:
            # AUC-ROC：一对多，宏平均
            metrics['auc_roc_macro'] = roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='macro'
            )
            # 各类别 PR-AUC（不平衡下比 ROC 更敏感）
            y_bin = label_binarize(y_true, classes=SEVERITY_CLASSES)
            for i, name in enumerate(class_names):
                metrics[f'{name}_pr_auc'] = average_precision_score(
                    y_bin[:, i], y_proba[:, i]
                )
        except Exception as e:
            logger.warning(f"    Could not compute probability metrics: {e}")

    return metrics


# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=SEVERITY_CLASSES)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"    Saved: {save_path}")
    plt.close(fig)
    return fig


# 绘制 ROC 曲线
def plot_roc_curves(y_true, y_proba_dict, class_names=None, save_path=None):
    if class_names is None:
        class_names = SEVERITY_NAMES

    y_bin = label_binarize(y_true, classes=SEVERITY_CLASSES)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    for i, (cls_name, ax) in enumerate(zip(class_names, axes)):
        for j, (model_name, y_proba) in enumerate(y_proba_dict.items()):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
            ax.plot(fpr, tpr, color=colors[j % len(colors)],
                    label=f'{model_name} (AUC={auc:.3f})', linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'ROC — {cls_name}', fontsize=13)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.2)

    plt.suptitle('ROC Curves (One-vs-Rest)', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"    Saved: {save_path}")
    plt.close(fig)
    return fig


# 绘制 PR 曲线
def plot_pr_curves(y_true, y_proba_dict, class_names=None, save_path=None):
    if class_names is None:
        class_names = SEVERITY_NAMES

    y_bin = label_binarize(y_true, classes=SEVERITY_CLASSES)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    for i, (cls_name, ax) in enumerate(zip(class_names, axes)):
        for j, (model_name, y_proba) in enumerate(y_proba_dict.items()):
            prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            ap = average_precision_score(y_bin[:, i], y_proba[:, i])
            ax.plot(rec, prec, color=colors[j % len(colors)],
                    label=f'{model_name} (AP={ap:.3f})', linewidth=2)

        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title(f'PR Curve — {cls_name}', fontsize=13)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.2)

    plt.suptitle('Precision-Recall Curves (per class)', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"    Saved: {save_path}")
    plt.close(fig)
    return fig


# 绘制特征重要性或系数
def plot_feature_importance(model, feature_names, model_name, top_n=15, save_path=None):

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.4)))

    if hasattr(model, 'feature_importances_'):
        # 树模型：基尼重要性
        importances = model.feature_importances_
        if len(feature_names) != len(importances):
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        imp_df = imp_df.nlargest(top_n, 'importance')
        ax.barh(imp_df['feature'], imp_df['importance'], color='steelblue')
        ax.set_xlabel('Importance (Gini)', fontsize=12)

    elif hasattr(model, 'coef_'):
        # 逻辑回归：各类系数绝对值均值
        coefs = np.abs(model.coef_).mean(axis=0)
        if len(feature_names) != len(coefs):
            feature_names = [f'Feature_{i}' for i in range(len(coefs))]
        imp_df = pd.DataFrame({'feature': feature_names, 'importance': coefs})
        imp_df = imp_df.nlargest(top_n, 'importance')
        ax.barh(imp_df['feature'], imp_df['importance'], color='coral')
        ax.set_xlabel('Mean |Coefficient| across classes', fontsize=12)

    else:
        ax.text(0.5, 0.5, 'No importance available', ha='center', va='center')

    ax.set_title(f'Feature Importance — {model_name} (Top {top_n})', fontsize=14)
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"    Saved: {save_path}")
    plt.close(fig)
    return fig


# 绘制模型指标对比条形图
def plot_model_comparison(metrics_dict, metric_names=None, save_path=None):
    """对比多项指标的模型柱状图。

    默认指标：Accuracy、Macro F1、Balanced Accuracy、Fatal Recall。
    Fatal Recall 作为优先关注指标。

    参数:
        metrics_dict: {model_name: metrics_dict} 的字典
        metric_names: 需要对比的指标键列表（默认 4 项）
        save_path: 保存路径

    返回:
        matplotlib 图对象
    """
    if metric_names is None:
        metric_names = ['accuracy', 'balanced_accuracy', 'macro_f1', 'Fatal_recall']

    display_names = {
        'accuracy': 'Accuracy',
        'balanced_accuracy': 'Balanced Acc',
        'macro_f1': 'Macro F1',
        'Fatal_recall': 'Fatal Recall',
        'weighted_f1': 'Weighted F1',
        'auc_roc_macro': 'AUC-ROC',
    }

    model_names = list(metrics_dict.keys())
    n_metrics = len(metric_names)
    x = np.arange(n_metrics)
    width = 0.8 / len(model_names)
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336']

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, name in enumerate(model_names):
        values = [metrics_dict[name].get(m, 0) for m in metric_names]
        bars = ax.bar(x + i * width, values, width, label=name,
                      color=colors[i % len(colors)], alpha=0.85)
        # 添加柱子数值标注
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([display_names.get(m, m) for m in metric_names], fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"    Saved: {save_path}")
    plt.close(fig)
    return fig


# 生成评估汇总表
def create_comparison_table(metrics_dict, save_path=None):
    rows = []
    for name, metrics in metrics_dict.items():
        row = {'Model': name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    if save_path:
        df.to_csv(save_path, index=False, float_format='%.4f')
        logger.info(f"    Saved: {save_path}")

    return df


# 进行时序留出集评估
def evaluate_temporal_holdout(model, X_temporal_train, y_temporal_train,
                              X_temporal_test, y_temporal_test, model_name):
    logger.info(f"    Temporal holdout: fitting {model_name} on 2005-2012...")

    model.fit(X_temporal_train, y_temporal_train)

    y_pred = model.predict(X_temporal_test)
    try:
        y_proba = model.predict_proba(X_temporal_test)
    except Exception:
        y_proba = None

    metrics = compute_metrics(y_temporal_test, y_pred, y_proba,
                              class_names=SEVERITY_NAMES)
    metrics['model'] = model_name
    metrics['eval_type'] = 'temporal_holdout'

    logger.info(f"    {model_name} temporal: Macro F1={metrics['macro_f1']:.4f}, "
                f"Fatal Recall={metrics.get('Fatal_recall', 0):.4f}, "
                f"Balanced Acc={metrics['balanced_accuracy']:.4f}")

    return metrics


# 分析聚类维度的错误分布
def analyze_cluster_errors(y_true, y_pred, cluster_labels, class_names=None):
    if class_names is None:
        class_names = SEVERITY_NAMES

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cluster_labels = np.array(cluster_labels)

    # 按簇与类别的错误分析
    error_records = []
    for cluster_id in sorted(np.unique(cluster_labels)):
        c_mask = cluster_labels == cluster_id
        c_total = c_mask.sum()
        c_errors = (y_true[c_mask] != y_pred[c_mask]).sum()
        c_error_rate = c_errors / c_total if c_total > 0 else 0

        for class_idx, (class_code, class_name) in enumerate(zip(SEVERITY_CLASSES, class_names)):
            cc_mask = c_mask & (y_true == class_code)
            cc_total = cc_mask.sum()
            cc_errors = (y_pred[cc_mask] != class_code).sum()
            cc_error_rate = cc_errors / cc_total if cc_total > 0 else 0
            # 每类召回 = 1 - 错误率
            cc_recall = 1 - cc_error_rate

            error_records.append({
                'cluster': cluster_id,
                'class': class_name,
                'n_samples': int(cc_total),
                'n_errors': int(cc_errors),
                'error_rate': round(cc_error_rate, 4),
                'recall': round(cc_recall, 4),
                'cluster_total': int(c_total),
                'cluster_error_rate': round(c_error_rate, 4),
            })

    error_df = pd.DataFrame(error_records)

    # 致死样本的误判去向
    fatal_pattern_records = []
    for cluster_id in sorted(np.unique(cluster_labels)):
        c_mask = cluster_labels == cluster_id
        fatal_mask = c_mask & (y_true == 1)  # 该簇内真实致死
        n_fatal = fatal_mask.sum()

        if n_fatal > 0:
            preds_for_fatal = y_pred[fatal_mask]
            n_correct = (preds_for_fatal == 1).sum()
            n_to_serious = (preds_for_fatal == 2).sum()
            n_to_slight = (preds_for_fatal == 3).sum()  # 风险最大的误判

            fatal_pattern_records.append({
                'cluster': cluster_id,
                'n_fatal': int(n_fatal),
                'correct_pct': round(n_correct / n_fatal * 100, 1),
                'to_serious_pct': round(n_to_serious / n_fatal * 100, 1),
                'to_slight_pct': round(n_to_slight / n_fatal * 100, 1),
            })

    fatal_pattern_df = pd.DataFrame(fatal_pattern_records)

    logger.info(f"  Cluster error analysis ({len(np.unique(cluster_labels))} clusters):")
    for cid in sorted(np.unique(cluster_labels)):
        c_data = error_df[error_df['cluster'] == cid]
        overall_err = c_data.iloc[0]['cluster_error_rate']
        fatal_rec = c_data[c_data['class'] == 'Fatal']['recall'].values
        fatal_rec_str = f"{fatal_rec[0]:.4f}" if len(fatal_rec) > 0 else "N/A"
        logger.info(f"    Cluster {cid}: error_rate={overall_err:.4f}, Fatal recall={fatal_rec_str}")

    return error_df, fatal_pattern_df
