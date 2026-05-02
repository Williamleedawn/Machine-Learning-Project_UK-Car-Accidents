"""
特征工程模块 - 流水线第4和第7阶段。
职责:
1. 构造时间特征（Time_Period、Is_Weekend、Season）
2. 构造二值风险特征（Is_Dark、Bad_Weather、Wet_Road 等）
3. 构造交互特征（Dark_And_Wet、Rural_High_Speed、Dark_Rural）
4. 对 Local_Authority_(District) 做频率编码
5. 提供特征分类清单用于管道构建
6. 按模型类型构建预处理管道（linear、tree、histgbt）
"""

# 特征工程与预处理管道

import logging

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

# 编码与分箱常量

DARK_CODES = {4, 5, 6, 7}
FINE_WEATHER_CODES = {1}
DRY_ROAD_CODES = {1}
WEEKEND_CODES = {1, 7}  
RURAL_CODE = 2
TIME_PERIOD_BINS = [-1, 5, 9, 15, 18, 23]  
TIME_PERIOD_LABELS = [0, 1, 2, 3, 4]
TIME_PERIOD_NAMES = {0: 'dawn', 1: 'morning_rush', 2: 'daytime',
                     3: 'evening_rush', 4: 'night'}

MONTH_TO_SEASON = {12: 0, 1: 0, 2: 0,   # 0 表示冬季
                   3: 1, 4: 1, 5: 1,     # 1 表示春季
                   6: 2, 7: 2, 8: 2,     # 2 表示夏季
                   9: 3, 10: 3, 11: 3}   # 3 表示秋季
SEASON_NAMES = {0: 'winter', 1: 'spring', 2: 'summer', 3: 'autumn'}


# 数值特征列表
NUMERIC_FEATURES = [
    'Speed_limit',          
    'Number_of_Vehicles',  
    'Hour',                
    'Year',                 
    'Month',             
    'Day',                 
    'Mean_Driver_Age',      
    'Mean_Vehicle_Age',     
    'Mean_Engine_Capacity', 
    'LA_Frequency',        
]

# 二值特征列表
BINARY_FEATURES = [
    'Is_Weekend',           
    'Is_Dark',              
    'Bad_Weather',        
    'Wet_Road',             
    'At_Junction',         
    'Is_Rural',        
    'Has_Motorcycle',       
    'Has_Heavy_Vehicle',    
    'Has_Young_Driver',    
    'Dark_And_Wet',        
    'Rural_High_Speed',     
    'Dark_Rural',          
]

# 类别特征列表
CATEGORICAL_FEATURES = [
    'Day_of_Week',          
    '1st_Road_Class',       
    'Road_Type',            
    'Junction_Detail',     
    'Light_Conditions',    
    'Weather_Conditions',   
    'Road_Surface_Conditions', 
    'Urban_or_Rural_Area', 
    'Pedestrian_Crossing-Human_Control',  
    'Pedestrian_Crossing-Physical_Facilities',  
    'Special_Conditions_at_Site', 
    'Carriageway_Hazards', 
    'Time_Period',          
    'Season',              
]

# 仅用于聚类的特征
CLUSTERING_ONLY_FEATURES = ['Latitude', 'Longitude']

TARGET = 'Accident_Severity'


# 构造时间相关特征
def create_temporal_features(df):
    logger.info("  Creating temporal features...")
    df = df.copy()
    df['Time_Period'] = pd.cut(
        df['Hour'],
        bins=TIME_PERIOD_BINS,
        labels=TIME_PERIOD_LABELS,
        right=True,
        include_lowest=True
    ).astype(float).astype('Int64')

    df['Is_Weekend'] = df['Day_of_Week'].isin(WEEKEND_CODES).astype(int)

    df['Season'] = df['Month'].map(MONTH_TO_SEASON)

    n_new = 3
    logger.info(f"    Time_Period: {df['Time_Period'].value_counts().sort_index().to_dict()}")
    logger.info(f"    Is_Weekend: {df['Is_Weekend'].value_counts().to_dict()}")
    logger.info(f"    Season: {df['Season'].value_counts().sort_index().to_dict()}")
    logger.info(f"    Added {n_new} temporal features")

    return df


# 构造风险二值特征
def create_binary_features(df, context_tables=None):
    logger.info("  Creating binary risk features...")
    df = df.copy()


    df['Is_Dark'] = df['Light_Conditions'].isin(DARK_CODES).astype(float)
    df.loc[df['Light_Conditions'].isna(), 'Is_Dark'] = np.nan


    df['Bad_Weather'] = (~df['Weather_Conditions'].isin(FINE_WEATHER_CODES)).astype(float)
    df.loc[df['Weather_Conditions'].isna(), 'Bad_Weather'] = np.nan

    df['Wet_Road'] = (~df['Road_Surface_Conditions'].isin(DRY_ROAD_CODES)).astype(float)
    df.loc[df['Road_Surface_Conditions'].isna(), 'Wet_Road'] = np.nan


    df['At_Junction'] = (df['Junction_Detail'] != 0).astype(float)
    df.loc[df['Junction_Detail'].isna(), 'At_Junction'] = np.nan

    df['Is_Rural'] = (df['Urban_or_Rural_Area'] == RURAL_CODE).astype(float)
    df.loc[df['Urban_or_Rural_Area'].isna(), 'Is_Rural'] = np.nan

    for feat in ['Is_Dark', 'Bad_Weather', 'Wet_Road', 'At_Junction', 'Is_Rural']:
        positive_rate = df[feat].mean()
        na_count = df[feat].isna().sum()
        logger.info(f"    {feat}: {positive_rate:.2%} positive"
                     f"{f', {na_count} NaN' if na_count > 0 else ''}")

    logger.info(f"    Added 5 binary features")
    return df


# 构造交互特征
def create_interaction_features(df):
    logger.info("  Creating interaction features...")
    df = df.copy()

    # Dark_And_Wet：能见度受限且路面湿滑
    # 说明：夜间降雨通常风险较高
    df['Dark_And_Wet'] = df['Is_Dark'] * df['Wet_Road']

    # Rural_High_Speed：乡村道路且限速较高
    # 说明：乡村高速路段更易发生严重后果
    df['Rural_High_Speed'] = df['Is_Rural'] * (df['Speed_limit'] >= 60).astype(float)

    # Dark_Rural：夜间且处于乡村区域
    # 说明：乡村夜间通常缺少照明
    df['Dark_Rural'] = df['Is_Dark'] * df['Is_Rural']

    for feat in ['Dark_And_Wet', 'Rural_High_Speed', 'Dark_Rural']:
        positive_rate = df[feat].mean()
        logger.info(f"    {feat}: {positive_rate:.2%} positive")

    logger.info(f"    Added 3 interaction features")
    return df


# 频率编码 Local_Authority_(District)
def frequency_encode_local_authority(train_df, test_df=None,
                                     col='Local_Authority_(District)'):
    logger.info("  Frequency-encoding Local Authority...")

    if col not in train_df.columns:
        logger.warning(f"    Column '{col}' not found — skipping frequency encoding")
        return train_df, test_df, {}

    train_df = train_df.copy()

    # 在训练集上计算频率映射
    freq_map = train_df[col].value_counts(normalize=True).to_dict()
    global_mean = 1.0 / train_df[col].nunique()

    logger.info(f"    {len(freq_map)} unique districts in training set")
    logger.info(f"    Global mean frequency: {global_mean:.6f}")

    # 应用到训练集
    train_df['LA_Frequency'] = train_df[col].map(freq_map)
    train_df = train_df.drop(columns=[col])

    # 应用到测试集（未见地区使用全局均值）
    if test_df is not None:
        test_df = test_df.copy()
        test_df['LA_Frequency'] = test_df[col].map(freq_map).fillna(global_mean)

        n_unseen = test_df['LA_Frequency'].eq(global_mean).sum()
        # 说明：计数包含未见地区及频率恰为全局均值的地区
        # 实际上 380 个地区下重合影响可忽略
        if n_unseen > 0:
            logger.info(f"    {n_unseen} test records with unseen districts → global mean")

        test_df = test_df.drop(columns=[col])

    logger.info(f"    Replaced '{col}' with 'LA_Frequency'")
    return train_df, test_df, freq_map


# 返回特征清单
def get_feature_lists():
    return {
        'numeric': list(NUMERIC_FEATURES),
        'binary': list(BINARY_FEATURES),
        'categorical': list(CATEGORICAL_FEATURES),
        'all_features': list(NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES),
        'excluded': list(CLUSTERING_ONLY_FEATURES),
        'target': TARGET,
    }


# 构建按模型类型区分的预处理管道
def build_preprocessing_pipeline(model_type, include_cluster=False):
    from src.clustering import ClusterFeatureTransformer
    import config

    features = get_feature_lists()
    num_features = features['numeric'] + features['binary']
    cat_features = features['categorical']

    if model_type == 'linear':
        # LR 需要独热与标准化以便解释系数
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(
                handle_unknown='ignore',  # 未见类别置为零向量
                sparse_output=False,      # 使用稠密数组便于下游处理
            )),
        ])

    elif model_type == 'tree':
        # RF：仅填补，不做编码（整数编码作为分裂近似）
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
        ])
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
        ])

    elif model_type == 'histgbt':
        # HistGBT：对类别做序数编码（通过索引指定类别列）
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
        ])
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1,  # 未见类别设为 -1（HistGBT 可处理）
            )),
        ])

    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. "
                         f"Must be one of 'linear', 'tree', 'histgbt'.")

    if include_cluster:
        cat_features = cat_features + ['Cluster_ID']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, num_features),
            ('cat', categorical_pipeline, cat_features),
        ],
        remainder='drop',  # 丢弃 Lat/Lon 与其他未列出的列
        verbose_feature_names_out=False,
    )

    if include_cluster:
        result = Pipeline([
            ('cluster', ClusterFeatureTransformer(
                n_clusters=4,  # K=4 来自第8阶段结论
                feature_cols=config.CLUSTER_FEATURES,
                random_state=config.RANDOM_SEED,
            )),
            ('column_transform', preprocessor),
        ])
    else:
        result = preprocessor

    return result
