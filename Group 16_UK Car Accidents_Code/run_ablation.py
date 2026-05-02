import os
import sys
import json
import time
import logging
import pickle
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import f1_score

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.models import get_model_configs
from src.evaluation import compute_metrics

# 日志配置（控制台 + 文件）
log_path = os.path.join(config.RESULTS_DIR, 'ablation_log.txt')
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

# 检查点与结果文件管理
CHECKPOINT_FILE = os.path.join(config.RESULTS_DIR, 'ablation_checkpoint.json')
RESULTS_FILE = os.path.join(config.RESULTS_DIR, 'Table_ablation.csv')


# 读取检查点
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {'completed': []}


# 保存检查点
def save_checkpoint(checkpoint):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


# 主入口：按方案运行 5 折 CV
def main():
    parser = argparse.ArgumentParser(description='Run imbalance ablation')
    parser.add_argument('--scheme', type=str, choices=['A', 'B', 'C'],
                        help='Run only this scheme (default: all)')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("IMBALANCE ABLATION — Starting")
    logger.info("=" * 60)

    # 读取已序列化的第7阶段数据
    pkl_path = os.path.join(config.RESULTS_DIR, 'phase7_data.pkl')
    if not os.path.exists(pkl_path):
        logger.error(f"Phase 7 data not found at {pkl_path}. "
                     f"Run main.py through Phase 7 first.")
        sys.exit(1)

    logger.info(f"Loading Phase 7 data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    X_train_sampled = data['X_train_sampled']
    y_train_sampled = data['y_train_sampled']
    X_train_full_enc = data['X_train_full_enc']
    y_train_full = data['y_train_full']

    logger.info(f"  Sampled train: {X_train_sampled.shape}")
    logger.info(f"  Full train: {X_train_full_enc.shape}")

    # 构建模型配置（include_cluster=False，第10阶段已确认无提升）
    model_configs = get_model_configs(include_cluster=False)

    # 定义方案
    schemes_cfg = {
        'A': {'data': (X_train_full_enc, y_train_full), 'class_weight': 'balanced',
               'desc': 'class_weight only (no sampling, ~1.42M rows)'},
        'B': {'data': (X_train_sampled, y_train_sampled), 'class_weight': None,
               'desc': 'sampling only (no class_weight, ~697K rows)'},
        'C': {'data': (X_train_sampled, y_train_sampled), 'class_weight': 'balanced',
               'desc': 'both sampling + class_weight (~697K rows)'},
    }

    # 如指定则只运行该方案
    if args.scheme:
        schemes_cfg = {args.scheme: schemes_cfg[args.scheme]}
        logger.info(f"Running only Scheme {args.scheme}")

    # 读取检查点和已有结果
    checkpoint = load_checkpoint()
    results = []
    if os.path.exists(RESULTS_FILE):
        existing_df = pd.read_csv(RESULTS_FILE)
        results = existing_df.to_dict('records')
        logger.info(f"Loaded {len(results)} existing results from {RESULTS_FILE}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

    t_total_start = time.time()

    for scheme_id, scheme_cfg in schemes_cfg.items():
        X_s, y_s = scheme_cfg['data']
        cw = scheme_cfg['class_weight']

        logger.info(f"\n{'='*60}")
        logger.info(f"SCHEME {scheme_id}: {scheme_cfg['desc']}")
        logger.info(f"{'='*60}")

        for name in ['LogisticRegression', 'HistGBT', 'RandomForest']:
            cfg = model_configs[name]
            unit_id = f"{scheme_id}_{name}"

            if unit_id in checkpoint['completed']:
                logger.info(f"  [{unit_id}] Skipping (already completed)")
                continue

            t_start = time.time()
            logger.info(f"\n  [{unit_id}] Starting 5-fold CV...")

            # 克隆流水线并设置当前方案的 class_weight
            pipeline = clone(cfg['pipeline'])
            pipeline.set_params(model__class_weight=cw)

            if name == 'RandomForest':
                try:
                    pipeline.set_params(model__n_jobs=1)
                except Exception:
                    pass

            fold_scores = []
            all_y_true, all_y_pred, all_y_proba = [], [], []

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_s, y_s)):
                t_fold = time.time()
                X_fold_train = X_s.iloc[train_idx]
                X_fold_val = X_s.iloc[val_idx]
                y_fold_train = y_s.iloc[train_idx]
                y_fold_val = y_s.iloc[val_idx]

                fold_pipeline = clone(pipeline)
                fold_pipeline.fit(X_fold_train, y_fold_train)

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

                elapsed_fold = time.time() - t_fold
                logger.info(f"    Fold {fold_idx+1}/5: Macro F1={fold_f1:.4f} ({elapsed_fold:.1f}s)")

            # 汇总折外预测指标
            y_proba_concat = np.vstack(all_y_proba) if all_y_proba else None
            metrics = compute_metrics(
                np.array(all_y_true), np.array(all_y_pred), y_proba_concat
            )
            metrics['scheme'] = scheme_id
            metrics['model'] = name
            metrics['cv_macro_f1_mean'] = round(np.mean(fold_scores), 6)
            metrics['cv_macro_f1_std'] = round(np.std(fold_scores), 6)
            metrics['fold_scores'] = ','.join(f'{s:.4f}' for s in fold_scores)

            elapsed = time.time() - t_start
            logger.info(f"  [{unit_id}] DONE: CV Macro F1 = {metrics['cv_macro_f1_mean']:.4f} "
                         f"± {metrics['cv_macro_f1_std']:.4f}, "
                         f"Fatal Recall = {metrics.get('Fatal_recall', 0):.4f}, "
                         f"Fatal Precision = {metrics.get('Fatal_precision', 0):.4f} "
                         f"({elapsed:.1f}s = {elapsed/60:.1f}min)")

            # 立即保存结果
            results.append(metrics)
            pd.DataFrame(results).to_csv(RESULTS_FILE, index=False, float_format='%.6f')

            # 更新检查点
            checkpoint['completed'].append(unit_id)
            save_checkpoint(checkpoint)

    total_elapsed = time.time() - t_total_start
    logger.info(f"\n{'='*60}")
    logger.info(f"ABLATION COMPLETE — Total time: {total_elapsed:.1f}s ({total_elapsed/3600:.1f}h)")
    logger.info(f"Results: {RESULTS_FILE}")
    logger.info(f"{'='*60}")

    # 输出汇总表
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        summary_cols = ['scheme', 'model', 'cv_macro_f1_mean', 'cv_macro_f1_std',
                        'Fatal_recall', 'Fatal_precision', 'balanced_accuracy']
        existing_cols = [c for c in summary_cols if c in df.columns]
        logger.info(f"\nAblation Summary:\n{df[existing_cols].to_string(index=False)}")


if __name__ == '__main__':
    main()
