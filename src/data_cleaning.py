"""
数据清洗模块 - 流水线第2和第6阶段。
职责:
1. 统计缺失模式（EDA）
2. 删除高缺失列（超过 30%）
3. 移除泄漏或结构性无信息列
4. 检测并裁剪数值异常
5. 在切分后对 Slight 类进行不对称下采样
6. 生成清洗前后对比报告
"""

# 数据清洗与采样相关函数

import os
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# 统计缺失率并给出机制推测
def identify_missing_patterns(df):
    logger.info("  Analyzing missing value patterns...")

    total_rows = len(df)
    records = []

    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct = n_missing / total_rows * 100

        # 缺失机制的启发式判断
        if pct > 30:
            mechanism = 'MNAR'
        elif pct > 1:
            mechanism = 'MAR'
        elif pct > 0:
            mechanism = 'MCAR'
        else:
            mechanism = 'Complete'

        records.append({
            'column': col,
            'missing_count': int(n_missing),
            'missing_pct': round(pct, 4),
            'suspected_mechanism': mechanism,
        })

    report_df = pd.DataFrame(records).sort_values('missing_pct', ascending=False)
    report_df = report_df.reset_index(drop=True)

    # 保存到输出目录供报告使用
    from config import RESULTS_DIR
    save_path = os.path.join(RESULTS_DIR, 'Table00_missing_patterns.csv')
    report_df.to_csv(save_path, index=False)
    logger.info(f"  Missing patterns saved to {save_path}")

    # 输出含缺失列的摘要
    has_missing = report_df[report_df['missing_count'] > 0]
    if len(has_missing) > 0:
        logger.info(f"  {len(has_missing)} columns with missing values:")
        for _, row in has_missing.iterrows():
            logger.info(f"    {row['column']}: {row['missing_count']:,} "
                        f"({row['missing_pct']:.2f}%) - {row['suspected_mechanism']}")

    return report_df


# 删除高缺失率字段
def drop_high_missing_columns(df, threshold=0.30):
    logger.info(f"  Dropping columns with >{threshold*100:.0f}% missing values...")

    total_rows = len(df)
    missing_ratios = df.isna().sum() / total_rows
    cols_to_drop = missing_ratios[missing_ratios > threshold].index.tolist()

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        for col in cols_to_drop:
            pct = missing_ratios[col] * 100
            logger.info(f"    Dropped '{col}': {pct:.2f}% missing")
    else:
        logger.info("    No columns exceeded missing threshold")

    return df, cols_to_drop


# 删除可能泄漏或冗余的列
def remove_leakage_columns(df):
    columns_to_remove = [
        # 后验结果变量（泄漏）
        ('Number_of_Casualties',
         'Post-hoc outcome: total casualty count is determined AFTER the accident, '
         'not available at the moment of first report to emergency services'),

        ('Did_Police_Officer_Attend_Scene_of_Accident',
         'Post-hoc and causally linked to severity: police are more likely to attend '
         'serious/fatal accidents, creating a reverse-causal leakage signal'),

        # 纯标识列（无预测价值）
        ('Accident_Index',
         'Primary key identifier: no generalizable predictive signal. '
         'Already used for vehicle merge; no longer needed as a feature'),

        ('1st_Road_Number',
         'Specific road number (e.g., A3218): too granular to generalize, '
         'and geographic info is captured by Local_Authority frequency encoding'),

        # 冗余或共线列
        ('Police_Force',
         'Near-collinear with Local_Authority_(District): both encode geographic '
         'jurisdiction. Retaining both adds no information but increases dimensionality'),

        ('Local_Authority_(Highway)',
         'String-typed highway authority code (e.g., "E09000020"), redundant with '
         'the numeric Local_Authority_(District). Would cause encoding complications'),

        # 过细的地理标识（与 LA 编码冗余）
        ('LSOA_of_Accident_Location',
         'Lower Super Output Area: extremely granular geographic ID (~35,000 unique). '
         'Geographic signal captured via Local_Authority frequency encoding instead'),

        ('Location_Easting_OSGR',
         'OS Grid reference (easting): raw coordinate, geographic info captured '
         'via Local_Authority. Lat/Lon retained for clustering spatial analysis'),

        ('Location_Northing_OSGR',
         'OS Grid reference (northing): same rationale as easting'),

        # 结构性 MNAR（0 表示不适用，比例较高）
        ('2nd_Road_Class',
         'Structural MNAR: 0 means "no second road" (not at junction) for ~40% of '
         'records. Junction information already captured by Junction_Detail and '
         'At_Junction binary feature. Low incremental predictive value'),

        ('2nd_Road_Number',
         'Same structural MNAR issue as 2nd_Road_Class: road number of a non-existent '
         'second road is semantically meaningless'),
    ]

    col_names = [name for name, _ in columns_to_remove]
    existing_cols = [c for c in col_names if c in df.columns]
    missing_cols = [c for c in col_names if c not in df.columns]

    logger.info(f"  Removing {len(existing_cols)} leakage/structural columns...")
    for name, reason in columns_to_remove:
        if name in df.columns:
            logger.info(f"    Removing '{name}': {reason[:80]}...")

    df = df.drop(columns=existing_cols, errors='ignore')

    if missing_cols:
        logger.info(f"    Already removed (by prior step): {missing_cols}")

    if 'Longitude' in df.columns and 'Latitude' in df.columns:
        logger.info("    Kept: Longitude, Latitude (needed for clustering Phase 8)")

    logger.info(f"  Shape after removal: {df.shape}")

    return df


# 按领域范围裁剪异常值
def clip_outliers(df):
    logger.info("  Clipping outliers using domain-knowledge bounds...")

    # 裁剪范围定义：{列: (最小, 最大)}
    clip_bounds = {
        'Speed_limit': (10, 70),
        'Mean_Driver_Age': (16, 90),
        'Mean_Engine_Capacity': (50, 8000),
        'Mean_Vehicle_Age': (0, 50),
        'Number_of_Vehicles': (1, 20),
    }

    for col, (lower, upper) in clip_bounds.items():
        if col not in df.columns:
            continue

        n_below = (df[col] < lower).sum()
        n_above = (df[col] > upper).sum()

        if n_below > 0 or n_above > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            logger.info(f"    {col}: clipped {n_below} below {lower}, "
                        f"{n_above} above {upper}")

    return df


# 对 Slight 类进行不对称下采样
def asymmetric_undersample(df, target_col='Accident_Severity', slight_ratio=0.4,
                           random_state=42):
    logger.info("  Applying asymmetric undersampling (training set only)...")

    # Accident_Severity 编码：1致死，2严重，3轻微
    fatal_serious = df[df[target_col] != 3]

    # 将 Slight(3) 按比例下采样
    slight = df[df[target_col] == 3]
    slight_sampled = slight.sample(
        frac=slight_ratio,
        random_state=random_state
    )
    result = pd.concat([fatal_serious, slight_sampled])
    result = result.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    logger.info(f"    Before: {len(df):,} rows — "
                f"Fatal={len(df[df[target_col]==1]):,}, "
                f"Serious={len(df[df[target_col]==2]):,}, "
                f"Slight={len(slight):,}")
    logger.info(f"    After:  {len(result):,} rows — "
                f"Fatal={len(result[result[target_col]==1]):,}, "
                f"Serious={len(result[result[target_col]==2]):,}, "
                f"Slight={len(slight_sampled):,}")

    return result


# 生成清洗前后对比报告
def generate_cleaning_report(df_before, df_after, target_col='Accident_Severity'):
    logger.info("  Generating cleaning report...")

    report = {}

    # 规模对比
    report['rows_before'] = len(df_before)
    report['rows_after'] = len(df_after)
    report['rows_removed'] = len(df_before) - len(df_after)
    report['rows_removed_pct'] = round(
        (len(df_before) - len(df_after)) / len(df_before) * 100, 4
    )
    report['cols_before'] = len(df_before.columns)
    report['cols_after'] = len(df_after.columns)
    report['cols_removed'] = len(df_before.columns) - len(df_after.columns)

    # 删除的列
    removed_cols = set(df_before.columns) - set(df_after.columns)
    report['columns_removed'] = sorted(removed_cols)

    # 目标分布对比（若两者均存在）
    if target_col in df_before.columns and target_col in df_after.columns:
        dist_before = df_before[target_col].value_counts(normalize=True).sort_index()
        dist_after = df_after[target_col].value_counts(normalize=True).sort_index()
        report['target_dist_before'] = dist_before.to_dict()
        report['target_dist_after'] = dist_after.to_dict()

    # 清洗后剩余 NaN 概况
    nan_after = df_after.isna().sum()
    nan_after = nan_after[nan_after > 0]
    report['remaining_nan_columns'] = {
        col: {'count': int(v), 'pct': round(v / len(df_after) * 100, 2)}
        for col, v in nan_after.items()
    }

    # 保存到输出目录
    from config import RESULTS_DIR
    save_path = os.path.join(RESULTS_DIR, 'Table00_cleaning_report.csv')

    # 转为适合 CSV 的平面结构
    summary_rows = [
        {'metric': 'rows_before', 'value': report['rows_before']},
        {'metric': 'rows_after', 'value': report['rows_after']},
        {'metric': 'rows_removed', 'value': report['rows_removed']},
        {'metric': 'rows_removed_pct', 'value': report['rows_removed_pct']},
        {'metric': 'cols_before', 'value': report['cols_before']},
        {'metric': 'cols_after', 'value': report['cols_after']},
        {'metric': 'cols_removed_count', 'value': report['cols_removed']},
        {'metric': 'columns_removed', 'value': ', '.join(report['columns_removed'])},
    ]
    pd.DataFrame(summary_rows).to_csv(save_path, index=False)
    logger.info(f"  Cleaning report saved to {save_path}")

    # 打印汇总
    logger.info(f"    Rows: {report['rows_before']:,} → {report['rows_after']:,} "
                f"(removed {report['rows_removed']:,}, {report['rows_removed_pct']}%)")
    logger.info(f"    Columns: {report['cols_before']} → {report['cols_after']} "
                f"(removed {report['cols_removed']})")
    if report.get('remaining_nan_columns'):
        logger.info(f"    Remaining NaN columns: {len(report['remaining_nan_columns'])}")

    return report
