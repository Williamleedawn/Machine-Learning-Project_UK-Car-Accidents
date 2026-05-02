"""
数据加载模块 - 流水线第1和第3阶段。
职责:
1. 加载 Accidents0515.csv 与 Vehicles0515.csv，并做健壮检查
2. 将 STATS19 的 -1 替换为 NaN（与标签无关）
3. 解析日期与时间字段
4. 按 Accident_Index 聚合车辆表并左连接到事故表
5. 加载 contextCSVs 对照表用于 EDA
"""

# 数据加载与聚合工具

import os
import glob
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
# 车辆类型编码集合
MOTORCYCLE_CODES = {2, 3, 4, 5, 23, 97}
HEAVY_VEHICLE_CODES = {10, 11, 19, 20, 21, 98}

# 加载事故主表
def load_accidents(filepath):
    # 按鲁棒性要求检查文件是否存在
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Accidents file not found at: {filepath}\n"
            f"Please download the STATS19 dataset and place it in the data/ directory.\n"
            f"See README.md for detailed setup instructions."
        )

    logger.info(f"  Loading Accidents from {filepath}...")

    df = pd.read_csv(filepath, encoding='utf-8-sig')

    logger.info(f"  Loaded shape: {df.shape}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    minus_one_counts = {}
    for col in numeric_cols:
        count = (df[col] == -1).sum()
        if count > 0:
            minus_one_counts[col] = int(count)

    if minus_one_counts:
        logger.info(f"  Columns with -1 values (pre-replacement): {minus_one_counts}")
    else:
        logger.info("  No -1 values found in numeric columns")

    df = df.replace(-1, np.nan)

    return df, minus_one_counts


# 加载车辆表
def load_vehicles(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Vehicles file not found at: {filepath}\n"
            f"Please download the STATS19 dataset and place it in the data/ directory.\n"
            f"See README.md for detailed setup instructions."
        )

    logger.info(f"  Loading Vehicles from {filepath}...")
    df = pd.read_csv(
        filepath,
        encoding='utf-8-sig',
        engine='python',
        on_bad_lines=lambda bad_line: bad_line[:22]
    )
    logger.info(f"  Loaded shape: {df.shape}")
    df = df.replace(-1, np.nan)

    return df


# 解析日期时间并抽取字段
def parse_datetime(df):
    logger.info("  Parsing Date and Time columns...")

    # 解析日期并提取年、月、日
    # 说明：指定格式后无需 dayfirst
    date_parsed = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Year'] = date_parsed.dt.year.astype(int)
    df['Month'] = date_parsed.dt.month.astype(int)
    df['Day'] = date_parsed.dt.day.astype(int)

    time_str = df['Time'].astype(str).str.strip()
    time_str = time_str.replace({'': np.nan, 'nan': np.nan})

    n_missing_time = time_str.isna().sum()
    if n_missing_time > 0:
        logger.info(f"  Dropping {n_missing_time} rows with missing Time "
                     f"({n_missing_time / len(df) * 100:.4f}%)")
        df = df[time_str.notna()].copy()
        time_str = time_str[time_str.notna()]

    df['Hour'] = time_str.str[:2].astype(int)

    df = df.drop(columns=['Date', 'Time'])

    logger.info(f"  Parsed: Year range [{df['Year'].min()}-{df['Year'].max()}], "
                f"Hour range [{df['Hour'].min()}-{df['Hour'].max()}]")

    return df


# 聚合车辆特征到事故粒度
def aggregate_vehicles(vehicles_df):
    logger.info("  Aggregating vehicle features by Accident_Index...")

    vehicles_df['_is_motorcycle'] = vehicles_df['Vehicle_Type'].isin(MOTORCYCLE_CODES).astype(int)
    vehicles_df['_is_heavy'] = vehicles_df['Vehicle_Type'].isin(HEAVY_VEHICLE_CODES).astype(int)
    vehicles_df['_is_young'] = (vehicles_df['Age_of_Driver'] <= 25).astype(int)

    grouped = vehicles_df.groupby('Accident_Index')

    agg_dict = {
        '_is_motorcycle': 'max',
        '_is_heavy': 'max',
        '_is_young': 'max',
        'Age_of_Driver': 'mean',     
        'Age_of_Vehicle': 'mean',
        'Engine_Capacity_(CC)': 'mean',
    }
    agg_df = grouped.agg(agg_dict)

    agg_df = agg_df.rename(columns={
        '_is_motorcycle': 'Has_Motorcycle',
        '_is_heavy': 'Has_Heavy_Vehicle',
        '_is_young': 'Has_Young_Driver',
        'Age_of_Driver': 'Mean_Driver_Age',
        'Age_of_Vehicle': 'Mean_Vehicle_Age',
        'Engine_Capacity_(CC)': 'Mean_Engine_Capacity',
    })

    logger.info(f"  Aggregated shape: {agg_df.shape}")
    logger.info(f"  Sample:\n{agg_df.head(3)}")

    return agg_df


# 合并事故表与车辆聚合表
def merge_accidents_vehicles(accidents_df, vehicles_agg_df):
    logger.info("  Merging accidents with aggregated vehicle features...")

    n_before = len(accidents_df)

    merged_df = pd.merge(
        accidents_df,
        vehicles_agg_df,
        left_on='Accident_Index',
        right_index=True,
        how='left'
    )

    n_matched = merged_df[vehicles_agg_df.columns[0]].notna().sum()
    n_unmatched = n_before - n_matched

    logger.info(f"  Merge result: {n_matched}/{n_before} matched "
                f"({n_unmatched} accidents without vehicle data)")
    logger.info(f"  Merged shape: {merged_df.shape}")

    return merged_df


# 加载上下文对照表
def load_context_tables(context_dir):
    if not os.path.exists(context_dir):
        logger.warning(f"Context directory not found: {context_dir}")
        return {}

    logger.info(f"  Loading context tables from {context_dir}...")

    tables = {}
    csv_files = glob.glob(os.path.join(context_dir, '*.csv'))

    for fpath in csv_files:
        stem = os.path.splitext(os.path.basename(fpath))[0]

        # 跳过命名混淆的指南文件（内容与 Vehicle_Type 重复）
        # 实际的 Vehicle_Type.csv 会单独加载
        if stem == 'Road-Accident-Safety-Data-Guide':
            continue

        try:
            tables[stem] = pd.read_csv(fpath, encoding='utf-8-sig')
        except Exception as e:
            logger.warning(f"  Failed to load {stem}: {e}")

    if 'Weather_Conditions' not in tables:
        tables['Weather_Conditions'] = pd.DataFrame({
            'code': [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'label': [
                'Data missing or out of range',
                'Fine no high winds',
                'Raining no high winds',
                'Raining + high winds',
                'Snowing no high winds',
                'Snowing + high winds',
                'Fog or mist',
                'Other',
                'Unknown',
                'Fine + high winds',
            ]
        })
        logger.info("  Added hardcoded Weather_Conditions lookup")

    if 'Road_Surface_Conditions' not in tables:
        tables['Road_Surface_Conditions'] = pd.DataFrame({
            'code': [-1, 1, 2, 3, 4, 5, 6, 7],
            'label': [
                'Data missing or out of range',
                'Dry',
                'Wet or damp',
                'Snow',
                'Frost or ice',
                'Flood over 3cm deep',
                'Oil or diesel',
                'Mud',
            ]
        })
        logger.info("  Added hardcoded Road_Surface_Conditions lookup")

    logger.info(f"  Loaded {len(tables)} context tables: {sorted(tables.keys())}")

    return tables
