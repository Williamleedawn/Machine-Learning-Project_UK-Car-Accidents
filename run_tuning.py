import os
import sys
import json
import time
import logging
import pickle
import argparse

import numpy as np
import pandas as pd
from sklearn.base import clone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.models import get_model_configs
from src.tuning import run_gridsearch, summarize_best_params
from src.evaluation import compute_metrics

# 日志配置
log_path = os.path.join(config.RESULTS_DIR, 'tuning_log.txt')
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

CHECKPOINT_FILE = os.path.join(config.RESULTS_DIR, 'tuning_checkpoint.json')


# 读取调参检查点
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {'completed': {}, 'scheme': None}


# 保存调参检查点
def save_checkpoint(checkpoint):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


# 根据消融结果选择最佳方案
def determine_best_scheme(scheme_arg):
    """确定调参使用的方案。"""
    if scheme_arg != 'auto':
        return scheme_arg

    ablation_path = os.path.join(config.RESULTS_DIR, 'Table_ablation.csv')
    if not os.path.exists(ablation_path):
        logger.error("Ablation results not found. Run run_ablation.py first, "
                     "or specify --scheme explicitly.")
        sys.exit(1)

    df = pd.read_csv(ablation_path)
    scheme_means = df.groupby('scheme')['cv_macro_f1_mean'].mean()
    best = scheme_means.idxmax()
    logger.info(f"Auto-selected Scheme {best} (mean CV Macro F1: {scheme_means[best]:.4f})")
    logger.info(f"  All schemes: {scheme_means.to_dict()}")
    return best


# 主入口：按方案运行调参
def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning')
    parser.add_argument('--scheme', type=str, default='auto',
                        choices=['A', 'B', 'C', 'auto'],
                        help='Imbalance scheme to tune on (default: auto from ablation)')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("HYPERPARAMETER TUNING — Starting")
    logger.info("=" * 60)

    pkl_path = os.path.join(config.RESULTS_DIR, 'phase7_data.pkl')
    if not os.path.exists(pkl_path):
        logger.error(f"Phase 7 data not found at {pkl_path}.")
        sys.exit(1)

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    best_scheme = determine_best_scheme(args.scheme)
    logger.info(f"Tuning on Scheme {best_scheme}")

    if best_scheme == 'A':
        X_tune = data['X_train_full_enc']
        y_tune = data['y_train_full']
        tune_cw = 'balanced'
    elif best_scheme == 'B':
        X_tune = data['X_train_sampled']
        y_tune = data['y_train_sampled']
        tune_cw = None
    else:  # C 方案
        X_tune = data['X_train_sampled']
        y_tune = data['y_train_sampled']
        tune_cw = 'balanced'

    logger.info(f"  Training data: {X_tune.shape}, class_weight={tune_cw}")

    model_configs = get_model_configs(include_cluster=False)

    param_grids = {
        'RandomForest': config.RF_PARAM_GRID,
        'HistGBT': config.HISTGBT_PARAM_GRID,
        'LogisticRegression': config.LR_PARAM_GRID,
    }

    checkpoint = load_checkpoint()
    checkpoint['scheme'] = best_scheme
    grid_results = {}

    execution_order = ['LogisticRegression', 'HistGBT', 'RandomForest']

    for name in execution_order:
        if name in checkpoint['completed']:
            logger.info(f"\n  [{name}] Skipping (already completed, "
                        f"best score: {checkpoint['completed'][name]['best_score']:.4f})")
            continue

        cfg = model_configs[name]
        grid = param_grids[name]

        logger.info(f"\n{'='*60}")
        logger.info(f"TUNING: {name}")
        logger.info(f"{'='*60}")

        pipeline = clone(cfg['pipeline'])
        pipeline.set_params(model__class_weight=tune_cw)

        if name == 'RandomForest':
            pipeline.set_params(model__n_jobs=1)
            n_jobs = -1
            use_randomized = len(X_tune) > 1_000_000
        elif name == 'HistGBT':
            n_jobs = 2
            use_randomized = False
        else:  # LR
            n_jobs = -1
            use_randomized = False

        if use_randomized:
            logger.info(f"  Using RandomizedSearchCV (n_iter=30) for {name} "
                        f"on {len(X_tune):,} rows (>1M)")
            from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
            gs = RandomizedSearchCV(
                pipeline, grid, n_iter=30, cv=cv, scoring='f1_macro',
                n_jobs=n_jobs, refit=True, random_state=config.RANDOM_SEED, verbose=1
            )
            t_start = time.time()
            gs.fit(X_tune, y_tune)
            elapsed = time.time() - t_start
            best_params = gs.best_params_
            best_score = gs.best_score_
        else:
            t_start = time.time()
            gs, best_params, best_score = run_gridsearch(
                pipeline, grid, X_tune, y_tune,
                cv_folds=config.CV_FOLDS,
                scoring='f1_macro',
                random_state=config.RANDOM_SEED,
                n_jobs=n_jobs
            )
            elapsed = time.time() - t_start

        logger.info(f"  [{name}] DONE: best CV Macro F1 = {best_score:.4f} ({elapsed/60:.1f}min)")
        logger.info(f"  Best params: {best_params}")

        # 保存该模型的详细 CV 结果
        cv_results_path = os.path.join(config.RESULTS_DIR, f'tuning_{name}_cv_results.csv')
        pd.DataFrame(gs.cv_results_).to_csv(cv_results_path, index=False)
        logger.info(f"  CV results saved to {cv_results_path}")

        # 保存最佳流水线
        pipeline_path = os.path.join(config.RESULTS_DIR, f'tuning_{name}_best_pipeline.pkl')
        with open(pipeline_path, 'wb') as f:
            pickle.dump(gs.best_estimator_, f)

        # 更新检查点
        checkpoint['completed'][name] = {
            'best_score': float(best_score),
            'best_params': {k: str(v) for k, v in best_params.items()},
            'elapsed_min': round(elapsed / 60, 1),
        }
        save_checkpoint(checkpoint)
        grid_results[name] = gs

    # 汇总
    logger.info(f"\n{'='*60}")
    logger.info("TUNING COMPLETE")
    logger.info(f"{'='*60}")

    # 保存最佳参数表
    if grid_results:
        params_df = summarize_best_params(grid_results)
        params_path = os.path.join(config.RESULTS_DIR, 'Table_tuning_best_params.csv')
        params_df.to_csv(params_path, index=False)
        logger.info(f"Best params saved to {params_path}")
        logger.info(f"\n{params_df.to_string(index=False)}")


if __name__ == '__main__':
    main()
