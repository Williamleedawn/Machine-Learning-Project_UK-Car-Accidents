"""
模型定义模块 - 流水线第9和第10阶段。
职责:
1. 定义随机森林、HistGradientBoosting、逻辑回归模型
2. 为每个模型封装对应的预处理管道
3. 提供统一的配置接口（get_model_configs）便于循环训练
4. 支持含或不含 Cluster_ID 的模型变体（Phase 10）
"""

# 模型与流水线构建

import logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

import config
from src.feature_engineering import build_preprocessing_pipeline, get_feature_lists

logger = logging.getLogger(__name__)


# 构建随机森林流水线
def create_rf_pipeline(preprocessing, rf_params):
    return Pipeline([
        ('preprocessing', preprocessing),
        ('model', RandomForestClassifier(**rf_params)),
    ])


# 构建 HistGBT 流水线（含类别索引）
def create_histgbt_pipeline(preprocessing, histgbt_params, include_cluster=False):
    features = get_feature_lists()
    n_num_bin = len(features['numeric']) + len(features['binary'])
    n_cat = len(features['categorical'])

    if include_cluster:
        n_cat += 1  # Cluster_ID 是额外的类别特征

    # CT 输出中的类别特征索引
    cat_indices = list(range(n_num_bin, n_num_bin + n_cat))

    params = {**histgbt_params, 'categorical_features': cat_indices}

    logger.info(f"    HistGBT categorical_features: indices {cat_indices[0]}..{cat_indices[-1]} "
                f"({n_cat} features, include_cluster={include_cluster})")

    return Pipeline([
        ('preprocessing', preprocessing),
        ('model', HistGradientBoostingClassifier(**params)),
    ])


# 构建逻辑回归流水线
def create_lr_pipeline(preprocessing, lr_params):
    return Pipeline([
        ('preprocessing', preprocessing),
        ('model', LogisticRegression(**lr_params)),
    ])


# 统一返回模型配置字典
def get_model_configs(include_cluster=False):
    logger.info(f"  Building model configs (include_cluster={include_cluster})...")

    # 为每类模型构建预处理管道
    pp_tree = build_preprocessing_pipeline('tree', include_cluster)
    pp_histgbt = build_preprocessing_pipeline('histgbt', include_cluster)
    pp_linear = build_preprocessing_pipeline('linear', include_cluster)

    # 按既定参数构建模型管道
    rf_pipeline = create_rf_pipeline(pp_tree, config.RF_PARAMS)
    histgbt_pipeline = create_histgbt_pipeline(pp_histgbt, config.HISTGBT_PARAMS, include_cluster)
    lr_pipeline = create_lr_pipeline(pp_linear, config.LR_PARAMS)

    configs = {
        'RandomForest': {
            'pipeline': rf_pipeline,
            'param_grid': config.RF_PARAM_GRID,
            'type': 'tree',
        },
        'HistGBT': {
            'pipeline': histgbt_pipeline,
            'param_grid': config.HISTGBT_PARAM_GRID,
            'type': 'histgbt',
        },
        'LogisticRegression': {
            'pipeline': lr_pipeline,
            'param_grid': config.LR_PARAM_GRID,
            'type': 'linear',
        },
    }

    for name, cfg in configs.items():
        logger.info(f"    {name}: pipeline steps = {[s[0] for s in cfg['pipeline'].steps]}")

    return configs
