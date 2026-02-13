#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户画像分析可视化脚本
生成基于analysis_summary.md和dimension_insights.json的可视化图表
并进行PCA+K-means聚类分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PersonaVisualizer:
    def __init__(self):
        self.dimension_stats = {}
        self.user_profiles = {}
        self.user_vectors = None
        self.pca_data = None
        self.cluster_labels = None
        
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        
        # 加载dimension_insights.json
        with open('outputs/dimension_insights.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.dimension_stats = data['dimension_stats']
            self.user_profiles = data['user_profiles']
        
        # 加载用户向量矩阵
        self.user_vectors = pd.read_csv('outputs/user_vector_matrix.csv')
        print(f"加载了 {len(self.user_vectors)} 条用户向量数据")
        
    def create_dimension_analysis_charts(self):
        """创建维度分析图表"""
        print("创建维度分析图表...")
        
        # 提取维度数据
        dimensions = list(self.dimension_stats.keys())
        avg_intensity = [self.dimension_stats[dim]['平均强度'] for dim in dimensions]
        coverage = [self.dimension_stats[dim]['覆盖率'] for dim in dimensions]
        user_counts = [self.dimension_stats[dim]['有效用户数'] for dim in dimensions]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('用户画像维度分析', fontsize=16, fontweight='bold')
        
        # 1. 平均关注强度排名
        ax1 = axes[0, 0]
        bars1 = ax1.barh(dimensions, avg_intensity, color='skyblue', alpha=0.8)
        ax1.set_xlabel('平均关注强度')
        ax1.set_title('各维度平均关注强度')
        ax1.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        # 2. 用户覆盖率
        ax2 = axes[0, 1]
        bars2 = ax2.barh(dimensions, [c*100 for c in coverage], color='lightcoral', alpha=0.8)
        ax2.set_xlabel('用户覆盖率 (%)')
        ax2.set_title('各维度用户覆盖率')
        ax2.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}%', ha='left', va='center', fontsize=9)
        
        # 3. 有效用户数量
        ax3 = axes[1, 0]
        bars3 = ax3.bar(range(len(dimensions)), user_counts, color='lightgreen', alpha=0.8)
        ax3.set_xlabel('维度')
        ax3.set_ylabel('有效用户数')
        ax3.set_title('各维度有效用户数量')
        ax3.set_xticks(range(len(dimensions)))
        ax3.set_xticklabels(dimensions, rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 50, 
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        # 4. 强度vs覆盖率散点图
        ax4 = axes[1, 1]
        scatter = ax4.scatter([c*100 for c in coverage], avg_intensity, 
                            s=200, alpha=0.7, c='purple')
        ax4.set_xlabel('用户覆盖率 (%)')
        ax4.set_ylabel('平均关注强度')
        ax4.set_title('关注强度 vs 覆盖率')
        ax4.grid(True, alpha=0.3)
        
        # 添加维度标签
        for i, dim in enumerate(dimensions):
            ax4.annotate(dim, (coverage[i]*100, avg_intensity[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('outputs/dimension_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_user_profile_charts(self):
        """创建用户画像类型图表"""
        print("创建用户画像类型图表...")
        
        # 提取Top 10用户画像类型
        profile_names = list(self.user_profiles.keys())[:10]
        user_counts = [self.user_profiles[profile]['user_count'] for profile in profile_names]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('主要用户画像类型分析', fontsize=16, fontweight='bold')
        
        # 1. 用户画像类型柱状图
        bars = ax1.barh(range(len(profile_names)), user_counts, color='steelblue', alpha=0.8)
        ax1.set_xlabel('用户数量')
        ax1.set_ylabel('用户画像类型')
        ax1.set_title('Top 10 用户画像类型')
        ax1.set_yticks(range(len(profile_names)))
        ax1.set_yticklabels([name.replace(' + ', '\n') for name in profile_names], fontsize=10)
        ax1.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 5, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center', fontsize=10)
        
        # 2. 用户画像类型饼图
        colors = plt.cm.Set3(np.linspace(0, 1, len(profile_names)))
        wedges, texts, autotexts = ax2.pie(user_counts, labels=None, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax2.set_title('用户画像类型分布')
        
        # 添加图例
        ax2.legend(wedges, [name.replace(' + ', '+') for name in profile_names], 
                  title="用户画像类型", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
        
        plt.tight_layout()
        plt.savefig('outputs/user_profile_types.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def perform_pca_analysis(self):
        """进行PCA主成分分析"""
        print("进行PCA主成分分析...")
        
        # 准备数据 - 提取8个维度的向量
        dimension_cols = ['外观设计', '内饰质感', '智能配置', '空间实用', '舒适体验', '操控性能', '续航能耗', '价值认知']
        X = self.user_vectors[dimension_cols].values
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA分析
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # 保存PCA结果
        self.pca_data = X_pca
        
        # 创建PCA分析图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('PCA主成分分析结果', fontsize=16, fontweight='bold')
        
        # 1. 解释方差比例
        ax1 = axes[0, 0]
        ax1.bar(range(1, 9), pca.explained_variance_ratio_, alpha=0.8, color='lightblue')
        ax1.set_xlabel('主成分')
        ax1.set_ylabel('解释方差比例')
        ax1.set_title('各主成分解释方差比例')
        ax1.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, ratio in enumerate(pca.explained_variance_ratio_):
            ax1.text(i+1, ratio + 0.005, f'{ratio:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. 累积解释方差
        ax2 = axes[0, 1]
        cumsum_ratio = np.cumsum(pca.explained_variance_ratio_)
        ax2.plot(range(1, 9), cumsum_ratio, 'o-', color='red', linewidth=2, markersize=8)
        ax2.axhline(y=0.8, color='gray', linestyle='--', alpha=0.7, label='80%阈值')
        ax2.axhline(y=0.9, color='gray', linestyle='--', alpha=0.7, label='90%阈值')
        ax2.set_xlabel('主成分数量')
        ax2.set_ylabel('累积解释方差比例')
        ax2.set_title('累积解释方差比例')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 添加数值标签
        for i, ratio in enumerate(cumsum_ratio):
            ax2.text(i+1, ratio + 0.02, f'{ratio:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. 前两个主成分的用户分布
        ax3 = axes[1, 0]
        scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=20)
        ax3.set_xlabel(f'PC1 (解释方差: {pca.explained_variance_ratio_[0]:.3f})')
        ax3.set_ylabel(f'PC2 (解释方差: {pca.explained_variance_ratio_[1]:.3f})')
        ax3.set_title('前两个主成分的用户分布')
        ax3.grid(True, alpha=0.3)
        
        # 4. 主成分载荷图
        ax4 = axes[1, 1]
        loadings = pca.components_[:2, :].T
        for i, (x, y) in enumerate(loadings):
            ax4.arrow(0, 0, x, y, head_width=0.03, head_length=0.03, fc='red', ec='red')
            ax4.text(x*1.1, y*1.1, dimension_cols[i], fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        ax4.set_xlim(-1, 1)
        ax4.set_ylim(-1, 1)
        ax4.set_xlabel(f'PC1 (解释方差: {pca.explained_variance_ratio_[0]:.3f})')
        ax4.set_ylabel(f'PC2 (解释方差: {pca.explained_variance_ratio_[1]:.3f})')
        ax4.set_title('主成分载荷图')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='k', linewidth=0.5)
        ax4.axvline(x=0, color='k', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig('outputs/pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return pca, scaler
        
    def perform_kmeans_clustering(self, n_clusters=20):
        """进行K-means++聚类分析"""
        print(f"进行K-means++聚类分析 (k={n_clusters})...")
        
        if self.pca_data is None:
            raise ValueError("请先进行PCA分析")
        
        # 使用前几个主成分进行聚类（保留90%方差）
        pca = PCA()
        dimension_cols = ['外观设计', '内饰质感', '智能配置', '空间实用', '舒适体验', '操控性能', '续航能耗', '价值认知']
        X = self.user_vectors[dimension_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_pca = pca.fit_transform(X_scaled)
        
        # 选择保留90%方差的主成分数量
        cumsum_ratio = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum_ratio >= 0.9) + 1
        X_pca_reduced = X_pca[:, :n_components]
        
        print(f"使用前 {n_components} 个主成分进行聚类 (保留方差: {cumsum_ratio[n_components-1]:.3f})")
        
        # 确定最优聚类数量
        silhouette_scores = []
        K_range = range(10, 31)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_pca_reduced)
            score = silhouette_score(X_pca_reduced, labels)
            silhouette_scores.append(score)
        
        # 找到最优k值
        optimal_k_idx = np.argmax(silhouette_scores)
        optimal_k = K_range[optimal_k_idx]
        
        print(f"最优聚类数量: {optimal_k} (轮廓系数: {silhouette_scores[optimal_k_idx]:.3f})")
        
        # 使用最优k值进行聚类
        kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
        self.cluster_labels = kmeans_final.fit_predict(X_pca_reduced)
        
        # 创建聚类分析图表
        self.create_clustering_charts(K_range, silhouette_scores, optimal_k, X_pca, kmeans_final)
        
        return kmeans_final, optimal_k
        
    def create_clustering_charts(self, K_range, silhouette_scores, optimal_k, X_pca, kmeans_model):
        """创建聚类分析图表"""
        print("创建聚类分析图表...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('K-means++ 聚类分析结果', fontsize=16, fontweight='bold')
        
        # 1. 轮廓系数图
        ax1 = axes[0, 0]
        ax1.plot(K_range, silhouette_scores, 'o-', linewidth=2, markersize=8)
        ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'最优k={optimal_k}')
        ax1.set_xlabel('聚类数量 (k)')
        ax1.set_ylabel('轮廓系数')
        ax1.set_title('不同聚类数量的轮廓系数')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 聚类结果可视化 (PC1 vs PC2)
        ax2 = axes[0, 1]
        scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=self.cluster_labels, 
                             cmap='tab20', alpha=0.6, s=20)
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title('聚类结果 (前两个主成分)')
        ax2.grid(True, alpha=0.3)
        
        # 添加聚类中心
        centers_2d = kmeans_model.cluster_centers_[:, :2]  # 只取前两个主成分
        ax2.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='x', s=200, linewidths=3)
        
        # 3. 聚类大小分布
        ax3 = axes[1, 0]
        cluster_sizes = pd.Series(self.cluster_labels).value_counts().sort_index()
        bars = ax3.bar(cluster_sizes.index, cluster_sizes.values, alpha=0.8, color='lightgreen')
        ax3.set_xlabel('聚类编号')
        ax3.set_ylabel('用户数量')
        ax3.set_title('各聚类用户数量分布')
        ax3.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 5, 
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # 4. 聚类质量指标
        ax4 = axes[1, 1]
        inertia = kmeans_model.inertia_
        silhouette_avg = silhouette_score(X_pca[:, :len(kmeans_model.cluster_centers_[0])], 
                                         self.cluster_labels)
        
        metrics = ['簇内平方和', '平均轮廓系数']
        values = [inertia/1000, silhouette_avg]  # 归一化显示
        colors = ['skyblue', 'lightcoral']
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.8)
        ax4.set_ylabel('数值')
        ax4.set_title('聚类质量指标')
        ax4.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        ax4.text(0, values[0] + values[0]*0.05, f'{inertia:.0f}', ha='center', va='bottom', fontsize=10)
        ax4.text(1, values[1] + values[1]*0.05, f'{silhouette_avg:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('outputs/clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_cluster_characteristics(self, kmeans_model, optimal_k):
        """分析各聚类的特征"""
        print("分析各聚类特征...")
        
        dimension_cols = ['外观设计', '内饰质感', '智能配置', '空间实用', '舒适体验', '操控性能', '续航能耗', '价值认知']
        
        # 计算每个聚类的均值特征
        cluster_profiles = []
        for i in range(optimal_k):
            cluster_mask = self.cluster_labels == i
            cluster_data = self.user_vectors[cluster_mask][dimension_cols]
            
            profile = {
                '聚类编号': i,
                '用户数量': sum(cluster_mask),
                '占比': f"{sum(cluster_mask)/len(self.user_vectors)*100:.1f}%"
            }
            
            # 计算各维度均值
            for dim in dimension_cols:
                profile[f'{dim}_均值'] = cluster_data[dim].mean()
            
            # 找出最突出的维度（Top 3）
            dim_means = [cluster_data[dim].mean() for dim in dimension_cols]
            top_dims = sorted(zip(dimension_cols, dim_means), key=lambda x: x[1], reverse=True)[:3]
            profile['主要特征'] = ' + '.join([dim for dim, _ in top_dims if _ > 0.3])
            
            cluster_profiles.append(profile)
        
        # 转换为DataFrame
        cluster_df = pd.DataFrame(cluster_profiles)
        
        # 保存聚类特征分析
        cluster_df.to_csv('outputs/cluster_characteristics.csv', index=False, encoding='utf-8-sig')
        
        # 创建聚类特征热力图
        self.create_cluster_heatmap(cluster_df, dimension_cols)
        
        return cluster_df
        
    def create_cluster_heatmap(self, cluster_df, dimension_cols):
        """创建聚类特征热力图"""
        print("创建聚类特征热力图...")
        
        # 准备热力图数据
        heatmap_data = []
        for _, row in cluster_df.iterrows():
            dim_values = [row[f'{dim}_均值'] for dim in dimension_cols]
            heatmap_data.append(dim_values)
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                 columns=dimension_cols,
                                 index=[f"聚类{i}\n({row['用户数量']}人)" 
                                       for i, row in cluster_df.iterrows()])
        
        # 创建热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': '维度强度'})
        plt.title('各聚类用户画像特征热力图', fontsize=16, fontweight='bold')
        plt.xlabel('用户画像维度')
        plt.ylabel('聚类群体')
        plt.tight_layout()
        plt.savefig('outputs/cluster_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_comprehensive_report(self, cluster_df):
        """生成综合分析报告"""
        print("生成综合分析报告...")
        
        report = f"""# 用户画像聚类分析报告

## 数据概览
- 总用户数: {len(self.user_vectors):,}
- 识别聚类数: {len(cluster_df)}
- 数据维度: 8个核心用户画像维度

## 聚类结果概览

"""
        
        for _, row in cluster_df.iterrows():
            if row['主要特征']:
                report += f"""### 聚类 {row['聚类编号']} - {row['主要特征']}型用户
- **用户数量**: {row['用户数量']} 人 ({row['占比']})
- **主要特征**: {row['主要特征']}
- **关键维度强度**:
"""
                dimension_cols = ['外观设计', '内饰质感', '智能配置', '空间实用', '舒适体验', '操控性能', '续航能耗', '价值认知']
                for dim in dimension_cols:
                    strength = row[f'{dim}_均值']
                    if strength > 0.3:
                        report += f"  - {dim}: {strength:.3f}\n"
                report += "\n"
        
        report += """
## 主要发现

1. **最大聚类群体**: 聚类规模最大的用户群体及其特征
2. **高价值用户群**: 在多个维度都表现活跃的用户群体  
3. **专业化用户**: 在特定维度有突出表现的专业用户群体
4. **潜力用户群**: 用户数量多但参与度有提升空间的群体

## 建议

基于聚类分析结果，建议：
- 针对不同聚类制定差异化的产品策略
- 关注头部聚类的需求满足
- 挖掘小众聚类的特殊需求
- 制定个性化的营销和推荐策略

---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 保存报告
        with open('outputs/clustering_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("综合分析报告已保存到 outputs/clustering_report.md")
        
    def run_full_analysis(self):
        """运行完整分析流程"""
        print("=" * 60)
        print("开始用户画像可视化分析")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 创建维度分析图表
        self.create_dimension_analysis_charts()
        
        # 3. 创建用户画像类型图表
        self.create_user_profile_charts()
        
        # 4. PCA分析
        pca, scaler = self.perform_pca_analysis()
        
        # 5. K-means聚类
        kmeans_model, optimal_k = self.perform_kmeans_clustering()
        
        # 6. 分析聚类特征
        cluster_df = self.analyze_cluster_characteristics(kmeans_model, optimal_k)
        
        # 7. 生成综合报告
        self.create_comprehensive_report(cluster_df)
        
        print("=" * 60)
        print("分析完成！所有结果已保存到 outputs/ 目录")
        print("=" * 60)

def main():
    visualizer = PersonaVisualizer()
    visualizer.run_full_analysis()

if __name__ == "__main__":
    main()