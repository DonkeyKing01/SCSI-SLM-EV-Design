"""
用户聚类画像管理模块
基于User_Preference_Clustering/outputs中的聚类结果
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import os


@dataclass
class UserProfile:
    """用户画像数据结构"""
    profile_id: int
    name: str
    description: str
    user_count: int
    percentage: float
    main_features: List[str]
    dimension_strengths: Dict[str, float]
    cluster_characteristics: Dict[str, float]


class UserClusteringManager:
    """用户聚类管理器"""
    
    def __init__(self, clustering_dir: str = "../02_User_Modeling/User_Preference_Clustering/outputs"):
        self.clustering_dir = clustering_dir
        self.user_profiles: Dict[int, UserProfile] = {}
        self.feature_dimensions = [
            "外观设计", "内饰质感", "智能配置", "空间实用", 
            "舒适体验", "操控性能", "续航能耗", "价值认知"
        ]
        self.load_clustering_results()
    
    def load_clustering_results(self):
        """加载聚类结果"""
        # 加载聚类特征CSV
        cluster_csv_path = os.path.join(self.clustering_dir, "cluster_characteristics.csv")
        if os.path.exists(cluster_csv_path):
            df = pd.read_csv(cluster_csv_path)
            self._parse_cluster_characteristics(df)
        else:
            # 如果CSV不存在，从markdown解析
            self._parse_clustering_report()
    
    def _parse_cluster_characteristics(self, df: pd.DataFrame):
        """解析聚类特征CSV"""
        for _, row in df.iterrows():
            profile_id = int(row['聚类编号'])
            
            # 解析主要特征
            main_features = row['主要特征'].split(' + ')
            
            # 构建维度强度字典
            dimension_strengths = {}
            for dim in self.feature_dimensions:
                col_name = f"{dim}_均值"
                if col_name in row:
                    dimension_strengths[dim] = float(row[col_name])
            
            # 创建用户画像
            profile = UserProfile(
                profile_id=profile_id,
                name=row['主要特征'],
                description=f"聚类{profile_id}: {row['主要特征']}型用户",
                user_count=int(row['用户数量']),
                percentage=float(row['占比'].rstrip('%')),
                main_features=main_features,
                dimension_strengths=dimension_strengths,
                cluster_characteristics=dimension_strengths.copy()
            )
            
            self.user_profiles[profile_id] = profile
    
    def _parse_clustering_report(self):
        """从markdown报告解析聚类信息"""
        report_path = os.path.join(self.clustering_dir, "clustering_report.md")
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"聚类报告文件不存在: {report_path}")
        
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析markdown内容
        sections = content.split('### 聚类')
        for section in sections[1:]:  # 跳过第一个空部分
            lines = section.strip().split('\n')
            if not lines:
                continue
            
            # 提取聚类编号
            first_line = lines[0].strip()
            if not first_line.startswith(' '):
                continue
            
            try:
                profile_id = int(first_line.split(' - ')[0])
            except (ValueError, IndexError):
                continue
            
            # 解析其他信息
            name = ""
            user_count = 0
            percentage = 0.0
            main_features = []
            dimension_strengths = {}
            
            for line in lines:
                line = line.strip()
                if '**名称概括**:' in line:
                    name = line.split('**名称概括**:')[1].strip()
                elif '**用户数量**:' in line:
                    count_text = line.split('**用户数量**:')[1].strip()
                    user_count = int(count_text.split(' ')[0])
                    percentage = float(count_text.split('(')[1].split('%')[0])
                elif '**主要特征**:' in line:
                    features_text = line.split('**主要特征**:')[1].strip()
                    main_features = [f.strip() for f in features_text.split('+')]
                elif '**关键维度强度**:' in line:
                    # 解析维度强度
                    continue
                elif ':' in line and any(dim in line for dim in self.feature_dimensions):
                    for dim in self.feature_dimensions:
                        if dim in line:
                            try:
                                value = float(line.split(':')[1].strip())
                                dimension_strengths[dim] = value
                            except ValueError:
                                pass
            
            # 创建用户画像
            profile = UserProfile(
                profile_id=profile_id,
                name=name or f"聚类{profile_id}",
                description=f"聚类{profile_id}: {' + '.join(main_features)}型用户",
                user_count=user_count,
                percentage=percentage,
                main_features=main_features,
                dimension_strengths=dimension_strengths,
                cluster_characteristics=dimension_strengths.copy()
            )
            
            self.user_profiles[profile_id] = profile
    
    def get_user_profile(self, profile_id: int) -> Optional[UserProfile]:
        """获取指定ID的用户画像"""
        return self.user_profiles.get(profile_id)
    
    def get_all_profiles(self) -> List[UserProfile]:
        """获取所有用户画像"""
        return list(self.user_profiles.values())
    
    def calculate_similarity(self, user_vector: Dict[str, float], profile_id: int) -> float:
        """计算用户向量与画像的相似度"""
        profile = self.user_profiles.get(profile_id)
        if not profile:
            return 0.0
        
        # 计算余弦相似度
        profile_vector = np.array([profile.dimension_strengths.get(dim, 0.0) for dim in self.feature_dimensions])
        user_vector_array = np.array([user_vector.get(dim, 0.0) for dim in self.feature_dimensions])
        
        # 避免除零
        if np.linalg.norm(profile_vector) == 0 or np.linalg.norm(user_vector_array) == 0:
            return 0.0
        
        similarity = np.dot(profile_vector, user_vector_array) / (np.linalg.norm(profile_vector) * np.linalg.norm(user_vector_array))
        return float(similarity)
    
    def find_best_matching_profile(self, user_vector: Dict[str, float]) -> Tuple[int, float]:
        """找到最佳匹配的用户画像"""
        best_profile_id = -1
        best_similarity = -1.0
        
        for profile_id in self.user_profiles:
            similarity = self.calculate_similarity(user_vector, profile_id)
            if similarity > best_similarity:
                best_similarity = similarity
                best_profile_id = profile_id
        
        return best_profile_id, best_similarity
    
    def get_profile_statistics(self) -> Dict:
        """获取画像统计信息"""
        total_users = sum(profile.user_count for profile in self.user_profiles.values())
        
        stats = {
            "total_profiles": len(self.user_profiles),
            "total_users": total_users,
            "profile_distribution": {},
            "feature_coverage": {dim: 0 for dim in self.feature_dimensions}
        }
        
        for profile in self.user_profiles.values():
            stats["profile_distribution"][profile.profile_id] = {
                "name": profile.name,
                "user_count": profile.user_count,
                "percentage": profile.percentage
            }
            
            # 统计特征覆盖
            for dim in self.feature_dimensions:
                if profile.dimension_strengths.get(dim, 0.0) > 0.5:  # 强度大于0.5的特征
                    stats["feature_coverage"][dim] += 1
        
        return stats
    
    def export_to_neo4j_format(self) -> List[Dict]:
        """导出为Neo4j格式的数据"""
        neo4j_data = []
        
        for profile in self.user_profiles.values():
            # 创建UserProfile节点数据
            node_data = {
                "profileId": profile.profile_id,
                "name": profile.name,
                "description": profile.description,
                "userCount": profile.user_count,
                "percentage": profile.percentage,
                "mainFeatures": profile.main_features,
                "dimensionStrengths": profile.dimension_strengths
            }
            
            neo4j_data.append({
                "type": "UserProfile",
                "data": node_data
            })
        
        return neo4j_data


def create_feature_nodes() -> List[Dict]:
    """创建8个核心特征节点"""
    features = [
        {"name": "外观设计", "category": "视觉", "description": "车辆外观设计的美观性和时尚感"},
        {"name": "内饰质感", "category": "触觉", "description": "车内装饰材料的质感和做工"},
        {"name": "智能配置", "category": "科技", "description": "智能驾驶辅助和车载科技功能"},
        {"name": "空间实用", "category": "实用", "description": "车内空间大小和实用性"},
        {"name": "舒适体验", "category": "体验", "description": "乘坐舒适性和驾驶体验"},
        {"name": "操控性能", "category": "性能", "description": "车辆操控性和驾驶乐趣"},
        {"name": "续航能耗", "category": "经济", "description": "电池续航里程和能耗效率"},
        {"name": "价值认知", "category": "价值", "description": "性价比和品牌价值认知"}
    ]
    
    return [{"type": "Feature", "data": feature} for feature in features]


if __name__ == "__main__":
    # 测试聚类管理器
    clustering_manager = UserClusteringManager()
    
    print(f"加载了 {len(clustering_manager.user_profiles)} 个用户画像")
    
    # 显示统计信息
    stats = clustering_manager.get_profile_statistics()
    print(f"总用户数: {stats['total_users']}")
    print(f"特征覆盖: {stats['feature_coverage']}")
    
    # 测试相似度计算
    test_vector = {
        "外观设计": 0.8,
        "内饰质感": 0.6,
        "智能配置": 0.9,
        "空间实用": 0.4,
        "舒适体验": 0.7,
        "操控性能": 0.5,
        "续航能耗": 0.8,
        "价值认知": 0.3
    }
    
    best_profile_id, similarity = clustering_manager.find_best_matching_profile(test_vector)
    best_profile = clustering_manager.get_user_profile(best_profile_id)
    
    print(f"\n测试向量最佳匹配:")
    print(f"画像ID: {best_profile_id}")
    print(f"画像名称: {best_profile.name}")
    print(f"相似度: {similarity:.4f}") 