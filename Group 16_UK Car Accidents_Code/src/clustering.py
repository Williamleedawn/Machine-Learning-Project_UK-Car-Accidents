"""
聚类模块 - 流水线第8阶段。
职责:
1. 通过肘部法和轮廓系数寻找最佳 K 的 KMeans
2. 层次聚类用于方法对比
3. 聚类画像与可视化（PCA 散点、特征热图、树状图）
4. ClusterFeatureTransformer 用于在监督管道中安全生成 Cluster_ID
5. 聚类与严重度交叉分析
"""

# 聚类相关功能：K 搜索、聚类拟合、可视化与管道集成

import os
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互后端用于保存图像
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

import config

logger = logging.getLogger(__name__)

# 限制线程数，避免聚类过度占用资源
os.environ['OMP_NUM_THREADS'] = '4'


# 将聚类封装为 Pipeline 可用的特征生成器
class ClusterFeatureTransformer(BaseEstimator, TransformerMixin):
    """将 KMeans 封装为可用于 Pipeline 的特征生成器，用于 CV 安全的 Cluster_ID。
    CFT 接收原始 DataFrame，选取 9 个聚类特征（config.CLUSTER_FEATURES），
    做标准化并拟合 KMeans，然后把 Cluster_ID 追加为新的类别列。
    之后 ColumnTransformer 再按模型类型处理该列。
    """

    def __init__(self, n_clusters=4, feature_cols=None, random_state=42):
        self.n_clusters = n_clusters
        self.feature_cols = feature_cols  
        self.random_state = random_state

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame) and self.feature_cols is not None:
            subset = X[self.feature_cols]
        else:
            subset = X
        if isinstance(subset, pd.DataFrame):
            valid_mask = subset.notna().all(axis=1)
            subset = subset[valid_mask]

        # 内部标准化
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(subset)

        # KMeans 聚类
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.kmeans_.fit(scaled)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame) and self.feature_cols is not None:
            subset = X[self.feature_cols].copy()
            subset = subset.fillna(0)
            scaled = self.scaler_.transform(subset)
            cluster_ids = self.kmeans_.predict(scaled)

            # 在原始 DataFrame 追加 Cluster_ID 列
            result = X.copy()
            result['Cluster_ID'] = cluster_ids
            return result
        else:
            cluster_ids = self.kmeans_.predict(X).reshape(-1, 1)
            return np.hstack([X if isinstance(X, np.ndarray) else X.values, cluster_ids])


# =============================================
# 核心聚类函数
# =============================================

# 准备聚类输入数据并标准化
def prepare_clustering_data(df, feature_cols, sample_size=None, random_state=42):
    logger.info(f"  Preparing clustering data ({len(feature_cols)} features)...")
    subset = df[feature_cols].copy()
    n_before = len(subset)
    valid_mask = subset.notna().all(axis=1)
    subset = subset[valid_mask]
    n_dropped = n_before - len(subset)
    if n_dropped > 0:
        logger.info(f"    Dropped {n_dropped} rows with NaN in clustering features "
                     f"({n_dropped / n_before * 100:.2f}%)")

    valid_iloc_positions = np.where(valid_mask.values)[0]
    sample_indices = None
    if sample_size is not None and len(subset) > sample_size:
        rng = np.random.RandomState(random_state)
        sample_positions = rng.choice(len(subset), size=sample_size, replace=False)
        sample_positions.sort()
        subset = subset.iloc[sample_positions]
        # 映射回原始 DataFrame 的 iloc 位置
        sample_indices = valid_iloc_positions[sample_positions]
        logger.info(f"    Subsampled to {sample_size:,} rows for clustering speed")
    else:
        sample_indices = valid_iloc_positions

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(subset)

    logger.info(f"    Scaled data shape: {scaled_data.shape}")
    return scaled_data, scaler, sample_indices


# 计算不同 K 的 SSE 与轮廓系数
def find_optimal_k(scaled_data, k_range, random_state=42):
    logger.info(f"  Finding optimal K in range {list(k_range)}...")

    k_values = list(k_range)
    sse_scores = []
    silhouette_scores = []

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(scaled_data)

        sse = km.inertia_
        sil = silhouette_score(scaled_data, labels)

        sse_scores.append(sse)
        silhouette_scores.append(sil)

        logger.info(f"    K={k}: SSE={sse:,.0f}, Silhouette={sil:.4f}")

    return {
        'k_values': k_values,
        'sse_scores': sse_scores,
        'silhouette_scores': silhouette_scores,
    }


# 拟合 K-Means 并返回标签
def fit_kmeans(scaled_data, n_clusters, random_state=42):
    logger.info(f"  Fitting K-Means with K={n_clusters}...")

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(scaled_data)

    # 记录各簇样本数
    unique, counts = np.unique(labels, return_counts=True)
    for c, n in zip(unique, counts):
        logger.info(f"    Cluster {c}: {n:,} samples ({n / len(labels) * 100:.1f}%)")

    return km, labels


# 拟合层次聚类用于对比
def fit_agglomerative(scaled_data, n_clusters):
    n_samples = scaled_data.shape[0]
    logger.info(f"  Fitting Agglomerative (Ward) with K={n_clusters} on {n_samples:,} samples...")

    if n_samples > 30000:
        logger.warning(f"    ⚠️ {n_samples} samples may cause memory issues for Agglomerative. "
                        f"Consider subsampling to ≤20K.")

    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = agg.fit_predict(scaled_data)

    unique, counts = np.unique(labels, return_counts=True)
    for c, n in zip(unique, counts):
        logger.info(f"    Cluster {c}: {n:,} samples ({n / len(labels) * 100:.1f}%)")

    return agg, labels


# 计算每个簇的特征均值画像
def compute_cluster_profiles(df, cluster_labels, feature_cols):
    profile_df = df[feature_cols].copy()
    profile_df['Cluster'] = cluster_labels

    profiles = profile_df.groupby('Cluster')[feature_cols].mean()

    logger.info(f"  Cluster profiles ({len(profiles)} clusters × {len(feature_cols)} features):")
    logger.info(f"\n{profiles.round(3).to_string()}")

    return profiles


# 统计聚类与事故严重程度的交叉分布
def cluster_severity_crosstab(df_or_series, cluster_labels, target_col='Accident_Severity'):
    severity_labels = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}

    if isinstance(df_or_series, pd.Series):
        severity = df_or_series
    else:
        severity = df_or_series[target_col]

    # 计数交叉表
    ct_counts = pd.crosstab(
        pd.Series(cluster_labels, name='Cluster'),
        severity.values,
    )
    ct_counts.columns = [severity_labels.get(c, c) for c in ct_counts.columns]

    # 每个簇内的比例（行归一化）
    ct_props = ct_counts.div(ct_counts.sum(axis=1), axis=0)

    logger.info(f"  Cluster × Severity proportions:")
    logger.info(f"\n{ct_props.round(4).to_string()}")

    return ct_counts, ct_props


# 计算 PCA 二维投影用于可视化
def compute_pca_2d(scaled_data, random_state=42):
    logger.info("  Computing PCA 2D projection for visualization...")

    pca = PCA(n_components=2, random_state=random_state)
    pca_2d = pca.fit_transform(scaled_data)

    total_var = sum(pca.explained_variance_ratio_) * 100
    logger.info(f"    PC1: {pca.explained_variance_ratio_[0]*100:.1f}%, "
                f"PC2: {pca.explained_variance_ratio_[1]*100:.1f}%, "
                f"Total: {total_var:.1f}%")

    return pca_2d, pca.explained_variance_ratio_, pca


# =============================================
# 可视化函数（生成报告用图）
# =============================================

# 绘制肘部法曲线
def plot_elbow(k_results, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_results['k_values'], k_results['sse_scores'], 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('SSE (Within-Cluster Sum of Squares)', fontsize=12)
    ax.set_title('Elbow Method for Optimal K', fontsize=14)
    ax.set_xticks(k_results['k_values'])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"    Saved: {save_path}")
    plt.close(fig)
    return fig


# 绘制轮廓系数曲线
def plot_silhouette(k_results, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_results['k_values'], k_results['silhouette_scores'],
            'rs-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Mean Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Score vs K', fontsize=14)
    ax.set_xticks(k_results['k_values'])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"    Saved: {save_path}")
    plt.close(fig)
    return fig


# 绘制 PCA 聚类散点图
def plot_pca_clusters(pca_2d, labels, var_ratio, save_path=None):
    """绘制按簇着色的 PCA 二维散点图（Fig06）。"""
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(pca_2d[:, 0], pca_2d[:, 1],
                         c=labels, cmap='Set2', alpha=0.3, s=5, edgecolors='none')
    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title(f'PCA 2D Projection — K-Means Clusters '
                 f'(total {sum(var_ratio)*100:.1f}% variance)', fontsize=14)
    plt.colorbar(scatter, ax=ax, label='Cluster ID')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"    Saved: {save_path}")
    plt.close(fig)
    return fig


# 绘制簇画像热图
def plot_cluster_profiles(profiles, feature_names_map=None, save_path=None):
    fig, ax = plt.subplots(figsize=(12, max(4, len(profiles) * 0.8 + 2)))

    profiles_norm = profiles.copy()
    for col in profiles.columns:
        col_range = profiles[col].max() - profiles[col].min()
        if col_range > 0:
            profiles_norm[col] = (profiles[col] - profiles[col].min()) / col_range
        else:
            profiles_norm[col] = 0.5

    sns.heatmap(profiles_norm, annot=profiles.round(2).values,
                fmt='', cmap='YlOrRd', ax=ax, linewidths=0.5,
                cbar_kws={'label': 'Normalized Value'})
    ax.set_title('Cluster Feature Profiles (annotated with actual values)', fontsize=14)
    ax.set_ylabel('Cluster', fontsize=12)
    ax.set_xlabel('Feature', fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"    Saved: {save_path}")
    plt.close(fig)
    return fig


# 绘制簇与严重程度分布图
def plot_cluster_severity(ct_counts, ct_props, save_path=None):
    """绘制聚类与严重度的堆叠柱状图（Fig08）。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 计数
    ct_counts.plot(kind='bar', stacked=True, ax=axes[0],
                   color=['#d62728', '#ff7f0e', '#2ca02c'])
    axes[0].set_title('Cluster × Severity (Counts)', fontsize=13)
    axes[0].set_xlabel('Cluster', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].legend(title='Severity')
    axes[0].tick_params(axis='x', rotation=0)

    # 比例
    ct_props.plot(kind='bar', stacked=True, ax=axes[1],
                  color=['#d62728', '#ff7f0e', '#2ca02c'])
    axes[1].set_title('Cluster × Severity (Proportions)', fontsize=13)
    axes[1].set_xlabel('Cluster', fontsize=11)
    axes[1].set_ylabel('Proportion', fontsize=11)
    axes[1].legend(title='Severity')
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"    Saved: {save_path}")
    plt.close(fig)
    return fig


# 绘制层次聚类树状图
def plot_dendrogram(scaled_data, save_path=None, max_samples=5000, random_state=42):
    logger.info(f"  Computing dendrogram (max {max_samples} samples)...")

    # 如有需要则子采样
    if len(scaled_data) > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(scaled_data), size=max_samples, replace=False)
        data_sample = scaled_data[idx]
    else:
        data_sample = scaled_data

    # 计算链接矩阵（Ward 方法）
    Z = linkage(data_sample, method='ward')

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=30,
               leaf_rotation=90, leaf_font_size=8,
               show_contracted=True)
    ax.set_title('Agglomerative Clustering Dendrogram (Ward Linkage)', fontsize=14)
    ax.set_xlabel('Cluster Size', fontsize=12)
    ax.set_ylabel('Distance (Ward)', fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        logger.info(f"    Saved: {save_path}")
    plt.close(fig)
    return fig
