import os

# 项目全局配置：路径、随机种子、模型参数等

# 随机种子与数据路径
RANDOM_SEED = 42  
DATA_DIR = "./data/UKCarAccidents/UKCarAccidents/"
ACCIDENTS_FILE = os.path.join(DATA_DIR, "Accidents0515.csv")
VEHICLES_FILE = os.path.join(DATA_DIR, "Vehicles0515.csv")

CONTEXT_DIR = os.path.join(DATA_DIR, "contextCSVs/")

# 输出路径
OUTPUT_DIR = "./outputs/"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures/")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results/")

for _dir in [FIGURES_DIR, RESULTS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# 训练/测试切分配置
TEST_SIZE = 0.2       
CV_FOLDS = 5         

# 不平衡采样与全量运行开关
SLIGHT_SAMPLE_RATIO = 0.4  
FULL_SCALE_RUN = False  

# 聚类参数
CLUSTER_K_RANGE = range(2, 11)   
CLUSTER_SAMPLE_SIZE = 80000      
CLUSTER_FEATURES = [             
    'Speed_limit',                
    'Hour',                       
    'Number_of_Vehicles',         
    'Is_Dark',                  
    'Bad_Weather',
    'Wet_Road',
    'Is_Rural',
    'At_Junction',
    'Road_Type',
]

# 时间切分配置（时序验证）
TEMPORAL_TRAIN_END_YEAR = 2012
TEMPORAL_TEST_START_YEAR = 2013

# 模型默认参数
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': None,         
    'min_samples_split': 5,    
    'class_weight': 'balanced', 
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
}

HISTGBT_PARAMS = {
    'max_iter': 200,
    'learning_rate': 0.1,
    'max_depth': 5,             
    'class_weight': 'balanced', 
    'random_state': RANDOM_SEED,
}

LR_PARAMS = {
    'C': 1.0,
    'class_weight': 'balanced',
    'max_iter': 1000,           
    'random_state': RANDOM_SEED,
    'solver': 'liblinear',     
}

# 调参网格
RF_PARAM_GRID = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [10, 20, 30, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
}

HISTGBT_PARAM_GRID = {
    'model__max_iter': [100, 200, 300],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__max_depth': [3, 5, 7],
}

LR_PARAM_GRID = {
    'model__C': [0.01, 0.1, 1, 10, 100],
    'model__penalty': ['l1', 'l2'],
    'model__solver': ['liblinear', 'saga'],
}

# 绘图参数
FIGURE_DPI = 150              
FIGURE_FORMAT = 'png'
FONT_SIZE = 12
