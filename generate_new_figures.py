# 生成补充图表的脚本
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

import config

FIGURES_DIR = config.FIGURES_DIR
RESULTS_DIR = config.RESULTS_DIR
os.makedirs(FIGURES_DIR, exist_ok=True)


# 图23：三层评估对比（宏 F1 与致死召回）
def fig23_three_layer_comparison():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'Table_cv_vs_test_vs_temporal.csv'))

    models = df['model'].tolist()
    model_labels = ['RF', 'HistGBT', 'LR']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 图A：三层宏 F1 对比
    x = np.arange(len(models))
    width = 0.25
    colors = ['#4472C4', '#ED7D31', '#70AD47']

    ax = axes[0]
    ax.bar(x - width, df['CV_macro_f1'], width, label='CV (L1)', color=colors[0])
    ax.bar(x, df['test_macro_f1'], width, label='Test (L2)', color=colors[1])
    ax.bar(x + width, df['temporal_macro_f1'], width, label='Temporal (L3)', color=colors[2])
    ax.set_ylabel('Macro F1', fontsize=12)
    ax.set_title('(a) Macro F1: Stable Across Layers', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=11)
    ax.set_ylim(0.30, 0.48)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    # 添加数值标注
    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)

    # 图B：测试集与时序集的致死召回对比
    ax = axes[1]
    ax.bar(x - width/2, df['test_fatal_recall'], width, label='Test (L2)', color=colors[1])
    ax.bar(x + width/2, df['temporal_fatal_recall'], width, label='Temporal (L3)', color=colors[2])
    ax.set_ylabel('Fatal Recall', fontsize=12)
    ax.set_title('(b) Fatal Recall: Significant Temporal Drop', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=11)
    ax.set_ylim(0, 0.75)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    for bars in ax.containers:
        ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)

    # 标注差值
    for i, row in df.iterrows():
        drop = row['temporal_fatal_recall'] - row['test_fatal_recall']
        mid_x = x[i]
        mid_y = max(row['test_fatal_recall'], row['temporal_fatal_recall']) + 0.03
        ax.annotate(f'{drop:+.3f}', xy=(mid_x, mid_y), ha='center', fontsize=9,
                    color='red', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'Fig23_three_layer_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# 图24：消融实验的致死召回热图
def fig24_ablation_fatal_recall_heatmap():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'Table_ablation.csv'))

    # 透视表：行=方案，列=模型，值=致死召回
    pivot = df.pivot_table(index='scheme', columns='model', values='Fatal_recall')
    # 调整顺序
    scheme_order = ['A', 'B', 'C']
    model_order = ['LogisticRegression', 'HistGBT', 'RandomForest']
    pivot = pivot.reindex(index=scheme_order, columns=model_order)

    # 用于展示的命名
    pivot.columns = ['LR', 'HistGBT', 'RF']
    pivot.index = ['A: class_weight only', 'B: sampling only', 'C: both']

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', linewidths=1,
                linecolor='white', ax=ax, vmin=0, vmax=0.7,
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    ax.set_title('Fatal Recall by Imbalance Scheme × Model\n'
                 '(Scheme B ≈ 0 across all models; HistGBT Scheme A/C > 0.6)',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('Imbalance Strategy', fontsize=11)
    ax.set_xlabel('Model', fontsize=11)
    plt.yticks(rotation=0)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'Fig24_ablation_fatal_recall_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# 图25：跨模型特征重要性对比
def fig25_cross_model_feature_importance():
    # 读取 Phase7 数据并训练简化模型提取重要性
    pkl_path = os.path.join(RESULTS_DIR, 'phase7_data.pkl')
    if not os.path.exists(pkl_path):
        print("  ERROR: phase7_data.pkl not found, skipping Fig25")
        return

    print("  Loading phase7_data.pkl...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    X_train = data['X_train_sampled']
    y_train = data['y_train_sampled']

    feature_names = list(X_train.columns)

    # 训练简化的 RF 和 HistGBT 提取重要性
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    import warnings
    warnings.filterwarnings('ignore')

    # 简单填补 NaN
    imputer = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_names)

    print("  Training RF for feature importance...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=4,
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_imp, y_train)
    rf_imp = pd.Series(rf.feature_importances_, index=feature_names)

    print("  Training HistGBT for feature importance...")
    hgbt = HistGradientBoostingClassifier(max_iter=200, max_depth=5, learning_rate=0.1,
                                           class_weight='balanced', random_state=42)
    hgbt.fit(X_imp, y_train)
    # HistGBT 无直接重要性时使用替代方案
    # 如果有 feature_importances_ 则优先使用
    # 否则检查属性存在性
    if hasattr(hgbt, 'feature_importances_'):
        hgbt_imp = pd.Series(hgbt.feature_importances_, index=feature_names)
    else:
        # 兜底：使用置换重要性
        from sklearn.inspection import permutation_importance
        print("  Computing permutation importance for HistGBT (this may take a moment)...")
        perm = permutation_importance(hgbt, X_imp.iloc[:50000], y_train.iloc[:50000],
                                       n_repeats=5, random_state=42, n_jobs=-1,
                                       scoring='f1_macro')
        hgbt_imp = pd.Series(perm.importances_mean, index=feature_names)

    # 取前10特征的并集
    rf_top10 = set(rf_imp.nlargest(10).index)
    hgbt_top10 = set(hgbt_imp.nlargest(10).index)
    top_features = sorted(rf_top10 | hgbt_top10,
                          key=lambda f: rf_imp[f] + hgbt_imp[f], reverse=True)[:12]

    # 归一化为百分比便于对比
    rf_pct = (rf_imp[top_features] / rf_imp.sum() * 100)
    hgbt_pct = (hgbt_imp[top_features] / hgbt_imp.sum() * 100)

    # 绘制并排横向柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top_features))
    bar_height = 0.35

    bars1 = ax.barh(y_pos - bar_height/2, rf_pct.values, bar_height,
                     label='Random Forest', color='#4472C4', alpha=0.85)
    bars2 = ax.barh(y_pos + bar_height/2, hgbt_pct.values, bar_height,
                     label='HistGBT', color='#ED7D31', alpha=0.85)

    ax.set_yticks(y_pos)
    # 简化特征名用于显示
    short_names = [f.replace('_', ' ').replace('Mean ', '').replace('Has ', '')
                   for f in top_features]
    ax.set_yticklabels(short_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Relative Importance (%)', fontsize=11)
    ax.set_title('Cross-Model Feature Importance: RF vs HistGBT (Top Features)\n'
                 'Shared top features suggest main-effect dominance over interactions',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    # 添加数值标注
    for bar in bars1:
        w = bar.get_width()
        if w > 0.5:
            ax.text(w + 0.2, bar.get_y() + bar.get_height()/2, f'{w:.1f}%',
                    va='center', fontsize=8)
    for bar in bars2:
        w = bar.get_width()
        if w > 0.5:
            ax.text(w + 0.2, bar.get_y() + bar.get_height()/2, f'{w:.1f}%',
                    va='center', fontsize=8)

    # 标出同时进入前10的特征
    shared = rf_top10 & hgbt_top10
    for i, f in enumerate(top_features):
        if f in shared:
            ax.get_yticklabels()[i].set_fontweight('bold')
            ax.get_yticklabels()[i].set_color('#006400')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'Fig25_cross_model_feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    print(f"  Shared top-10 features (bold green): {shared}")


# 脚本入口：按顺序生成新图
if __name__ == '__main__':
    print("Generating new figures for Session 9...")
    print()
    print("Fig23: Three-layer evaluation comparison")
    fig23_three_layer_comparison()
    print()
    print("Fig24: Fatal Recall ablation heatmap")
    fig24_ablation_fatal_recall_heatmap()
    print()
    print("Fig25: Cross-model feature importance")
    fig25_cross_model_feature_importance()
    print()
    print("Done! All new figures saved to outputs/figures/")
