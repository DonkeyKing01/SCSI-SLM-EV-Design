"""
新能源汽车知识图谱构建器
整合用户聚类、数据管理和Neo4j操作
"""
import os
import sys
import logging
import time
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from user_clustering import UserClusteringManager, create_feature_nodes
from data_manager import DataManager
from neo4j_manager import Neo4jManager


class KnowledgeGraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self, 
                 csv_file_path: str,
                 neo4j_uri: str = "bolt://localhost:7688",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "neo4j123",
                 neo4j_database: str = "neo4j",
                 clustering_dir: str = "../02_User_Modeling/User_Preference_Clustering/outputs"):
        
        self.csv_file_path = csv_file_path
        self.clustering_dir = clustering_dir
        
        # 初始化各个管理器
        self.user_clustering = UserClusteringManager(clustering_dir)
        self.data_manager = DataManager(csv_file_path)
        self.neo4j_manager = Neo4jManager(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            database=neo4j_database
        )
        
        # 统计信息
        self.build_stats = {
            "start_time": None,
            "end_time": None,
            "nodes_created": 0,
            "relationships_created": 0,
            "errors": []
        }
    
    def build_knowledge_graph(self, clear_existing: bool = False) -> Dict:
        """构建完整的知识图谱"""
        self.build_stats["start_time"] = time.time()
        
        try:
            logging.info("开始构建知识图谱...")
            
            # 1. 清空现有数据（如果需要）
            if clear_existing:
                logging.info("清空现有数据库...")
                self.neo4j_manager.clear_database()
            
            # 2. 创建约束和索引
            logging.info("创建数据库约束和索引...")
            self.neo4j_manager.create_constraints_and_indexes()
            
            # 3. 创建基础节点
            logging.info("创建基础节点...")
            self._create_base_nodes()
            
            # 4. 创建评论节点
            logging.info("创建评论节点...")
            self._create_review_nodes()
            
            # 5. 创建关系
            logging.info("创建关系...")
            self._create_relationships()
            
            # 6. 生成推断关系
            logging.info("生成推断关系...")
            self._generate_inferred_relationships()
            
            # 7. 验证构建结果
            logging.info("验证构建结果...")
            self._validate_build()
            
            self.build_stats["end_time"] = time.time()
            build_duration = self.build_stats["end_time"] - self.build_stats["start_time"]
            
            logging.info(f"知识图谱构建完成！耗时: {build_duration:.2f}秒")
            
            return {
                "status": "success",
                "duration": build_duration,
                "stats": self.build_stats,
                "database_stats": self.neo4j_manager.get_database_stats()
            }
            
        except Exception as e:
            error_msg = f"构建知识图谱失败: {e}"
            logging.error(error_msg)
            self.build_stats["errors"].append(error_msg)
            
            return {
                "status": "error",
                "error": error_msg,
                "stats": self.build_stats
            }
    
    def _create_base_nodes(self):
        """创建基础节点（车型、用户画像、特征）"""
        # 1. 创建车型节点
        car_models_data = []
        for car_model in self.data_manager.get_car_models():
            car_models_data.append({
                "type": "CarModel",
                "data": car_model
            })
        
        if car_models_data:
            result = self.neo4j_manager.batch_create_nodes(car_models_data)
            self.build_stats["nodes_created"] += result["total_created"]
            logging.info(f"创建了 {result['total_created']} 个车型节点")
        
        # 2. 创建用户画像节点
        user_profiles_data = self.user_clustering.export_to_neo4j_format()
        if user_profiles_data:
            result = self.neo4j_manager.batch_create_nodes(user_profiles_data)
            self.build_stats["nodes_created"] += result["total_created"]
            logging.info(f"创建了 {result['total_created']} 个用户画像节点")
        
        # 3. 创建特征节点
        features_data = create_feature_nodes()
        if features_data:
            result = self.neo4j_manager.batch_create_nodes(features_data)
            self.build_stats["nodes_created"] += result["total_created"]
            logging.info(f"创建了 {result['total_created']} 个特征节点")
    
    def _create_review_nodes(self):
        """创建评论节点"""
        reviews_data = []
        review_count = 0
        
        # 分批处理评论
        for batch in self.data_manager.get_reviews_batch(batch_size=1000):
            batch_reviews = []
            
            for _, review_row in batch.iterrows():
                try:
                    # 计算评论重要性
                    importance = self.data_manager.calculate_review_importance(review_row)
                    
                    # 计算整体情感分数
                    overall_sentiment = self._calculate_overall_sentiment(review_row)
                    
                    review_data = {
                        "reviewId": str(review_row.get('comment_id', review_count)),
                        "content": str(review_row.get('original_comment', '')),
                        "userId": str(review_row.get('user_name', f'user_{review_count}')),
                        "overallSentiment": overall_sentiment,
                        "importance": importance
                    }
                    
                    batch_reviews.append(review_data)
                    review_count += 1
                    
                except Exception as e:
                    error_msg = f"处理评论失败 (ID: {review_row.get('comment_id', 'unknown')}): {e}"
                    logging.warning(error_msg)
                    self.build_stats["errors"].append(error_msg)
                    continue
            
            if batch_reviews:
                created = self.neo4j_manager.create_review_nodes(batch_reviews)
                self.build_stats["nodes_created"] += created
                
                if review_count % 5000 == 0:
                    logging.info(f"已处理 {review_count} 条评论，创建了 {created} 个评论节点")
        
        logging.info(f"总共创建了 {self.build_stats['nodes_created']} 个评论节点")
    
    def _calculate_overall_sentiment(self, review_row) -> str:
        """计算评论的整体情感"""
        sentiment_scores = self.data_manager.get_sentiment_scores_from_review(review_row)
        
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for dim, scores in sentiment_scores.items():
            sentiment = scores.get("sentiment", "neutral")
            intensity = scores.get("intensity", 0.0)
            
            if intensity > 0.3:  # 只考虑强度大于0.3的特征
                if sentiment == "正面":
                    positive_count += 1
                elif sentiment == "负面":
                    negative_count += 1
                else:
                    neutral_count += 1
        
        # 根据情感分布判断整体情感
        if positive_count > negative_count and positive_count > neutral_count:
            return "positive"
        elif negative_count > positive_count and negative_count > neutral_count:
            return "negative"
        else:
            return "neutral"
    
    def _create_relationships(self):
        """创建关系"""
        relationships_data = []
        relationship_count = 0
        
        # 分批处理评论
        for batch in self.data_manager.get_reviews_batch(batch_size=1000):
            batch_relationships = []
            
            for _, review_row in batch.iterrows():
                try:
                    review_id = str(review_row.get('comment_id', relationship_count))
                    car_model_id = str(review_row.get('car_model', ''))
                    
                    if not car_model_id or (hasattr(pd, 'isna') and pd.isna(car_model_id)) or car_model_id == '' or str(car_model_id) == 'nan':
                        continue
                    
                    # 1. 创建MENTIONS关系
                    sentiment_scores = self.data_manager.get_sentiment_scores_from_review(review_row)
                    overall_sentiment_score = self._calculate_sentiment_score(sentiment_scores)
                    importance = self.data_manager.calculate_review_importance(review_row)
                    
                    mentions_rel = {
                        "type": "MENTIONS",
                        "data": {
                            "reviewId": review_id,
                            "carModelId": car_model_id,
                            "sentimentScore": overall_sentiment_score,
                            "importance": importance
                        }
                    }
                    batch_relationships.append(mentions_rel)
                    
                    # 2. 创建PUBLISHED关系
                    user_vector = self.data_manager.get_user_vector_from_review(review_row)
                    best_profile_id, user_match_score = self.user_clustering.find_best_matching_profile(user_vector)
                    
                    if best_profile_id >= 0:
                        published_rel = {
                            "type": "PUBLISHED",
                            "data": {
                                "profileId": best_profile_id,
                                "reviewId": review_id,
                                "userMatchScore": round(user_match_score, 4)
                            }
                        }
                        batch_relationships.append(published_rel)
                    
                    # 3. 创建CONTAINS_ASPECT关系
                    for dim in self.user_clustering.feature_dimensions:
                        intensity_col = f"{dim}_强度"
                        sentiment_col = f"{dim}_情感"
                        
                        if intensity_col in review_row and sentiment_col in review_row:
                            intensity = review_row[intensity_col]
                            sentiment = review_row[sentiment_col]
                            
                            # 检查pandas是否可用，否则使用基本检查
                            try:
                                intensity_valid = pd.notna(intensity) if hasattr(pd, 'notna') else (intensity is not None and str(intensity) != 'nan')
                                sentiment_valid = pd.notna(sentiment) if hasattr(pd, 'notna') else (sentiment is not None and str(sentiment) != 'nan')
                            except:
                                intensity_valid = intensity is not None and str(intensity) != 'nan'
                                sentiment_valid = sentiment is not None and str(sentiment) != 'nan'
                            
                            if intensity_valid and sentiment_valid and intensity > 0:
                                aspect_rel = {
                                    "type": "CONTAINS_ASPECT",
                                    "data": {
                                        "reviewId": review_id,
                                        "featureName": dim,
                                        "aspectSentiment": str(sentiment),
                                        "intensity": float(intensity)
                                    }
                                }
                                batch_relationships.append(aspect_rel)
                    
                    relationship_count += 1
                    
                except Exception as e:
                    error_msg = f"创建关系失败 (评论ID: {review_row.get('comment_id', 'unknown')}): {e}"
                    logging.warning(error_msg)
                    self.build_stats["errors"].append(error_msg)
                    continue
            
            if batch_relationships:
                result = self.neo4j_manager.create_relationships(batch_relationships)
                self.build_stats["relationships_created"] += result["total_created"]
                
                if relationship_count % 5000 == 0:
                    logging.info(f"已处理 {relationship_count} 条评论，创建了 {result['total_created']} 个关系")
        
        logging.info(f"总共创建了 {self.build_stats['relationships_created']} 个关系")
    
    def _calculate_sentiment_score(self, sentiment_scores: Dict) -> float:
        """计算情感分数 (-1到1之间)"""
        total_score = 0.0
        total_weight = 0.0
        
        for dim, scores in sentiment_scores.items():
            sentiment = scores.get("sentiment", "neutral")
            intensity = scores.get("intensity", 0.0)
            
            if intensity > 0:
                # 情感映射
                if sentiment == "正面":
                    score = 1.0
                elif sentiment == "负面":
                    score = -1.0
                else:
                    score = 0.0
                
                total_score += score * intensity
                total_weight += intensity
        
        if total_weight > 0:
            return round(total_score / total_weight, 4)
        else:
            return 0.0
    
    def _generate_inferred_relationships(self):
        """生成推断关系（用户画像与车型的兴趣关系）"""
        logging.info("生成用户画像与车型的兴趣关系...")
        
        # 获取所有用户画像
        user_profiles = self.user_clustering.get_all_profiles()
        
        for profile in user_profiles:
            try:
                # 查询该画像下所有评论对车型的评分
                query = """
                MATCH (p:UserProfile {profileId: $profileId})-[pub:PUBLISHED]->(r:Review)-[m:MENTIONS]->(c:CarModel)
                RETURN c.modelId as carModelId,
                       avg(pub.userMatchScore * m.sentimentScore * m.importance) as correlationScore,
                       count(r) as totalReviews,
                       sum(CASE WHEN m.sentimentScore > 0 THEN 1 ELSE 0 END) as positiveMentions,
                       sum(CASE WHEN m.sentimentScore < 0 THEN 1 ELSE 0 END) as negativeMentions
                """
                
                results = self.neo4j_manager.execute_query(query, {"profileId": profile.profile_id})
                
                for result in results:
                    if result["totalReviews"] > 0:
                        # 获取该画像最关注的特征
                        top_aspects = self._get_top_aspects_for_profile(profile.profile_id)
                        
                        interested_in_rel = {
                            "type": "INTERESTED_IN",
                            "data": {
                                "profileId": profile.profile_id,
                                "carModelId": result["carModelId"],
                                "correlationScore": round(result["correlationScore"], 4),
                                "positiveMentions": int(result["positiveMentions"]),
                                "negativeMentions": int(result["negativeMentions"]),
                                "topAspects": top_aspects
                            }
                        }
                        
                        # 创建关系
                        result_count = self.neo4j_manager.create_relationships([interested_in_rel])
                        self.build_stats["relationships_created"] += result_count["total_created"]
                
                logging.info(f"为用户画像 {profile.profile_id} 生成了兴趣关系")
                
            except Exception as e:
                error_msg = f"为用户画像 {profile.profile_id} 生成兴趣关系失败: {e}"
                logging.warning(error_msg)
                self.build_stats["errors"].append(error_msg)
                continue
    
    def _get_top_aspects_for_profile(self, profile_id: int) -> List[str]:
        """获取用户画像最关注的特征"""
        query = """
        MATCH (p:UserProfile {profileId: $profileId})-[pub:PUBLISHED]->(r:Review)-[ca:CONTAINS_ASPECT]->(f:Feature)
        RETURN f.name as featureName,
               avg(ca.intensity) as avgIntensity,
               count(r) as mentionCount
        ORDER BY avgIntensity DESC, mentionCount DESC
        LIMIT 3
        """
        
        try:
            results = self.neo4j_manager.execute_query(query, {"profileId": profile_id})
            return [result["featureName"] for result in results]
        except Exception as e:
            logging.warning(f"获取用户画像 {profile_id} 的特征失败: {e}")
            return []
    
    def _validate_build(self):
        """验证构建结果"""
        logging.info("验证知识图谱构建结果...")
        
        # 获取数据库统计
        db_stats = self.neo4j_manager.get_database_stats()
        
        # 验证节点数量
        expected_nodes = {
            "car_models": len(self.data_manager.get_car_models()),
            "user_profiles": len(self.user_clustering.get_all_profiles()),
            "features": 8,  # 固定的8个特征
            "reviews": len(self.data_manager.data) if self.data_manager.data is not None else 0
        }
        
        validation_results = {}
        for node_type, expected in expected_nodes.items():
            actual = db_stats.get(node_type, 0)
            validation_results[node_type] = {
                "expected": expected,
                "actual": actual,
                "status": "✓" if actual == expected else "✗"
            }
        
        # 输出验证结果
        logging.info("构建验证结果:")
        for node_type, result in validation_results.items():
            status = result["status"]
            expected = result["expected"]
            actual = result["actual"]
            logging.info(f"  {node_type}: {status} 期望: {expected}, 实际: {actual}")
        
        return validation_results
    
    def get_build_summary(self) -> Dict:
        """获取构建摘要"""
        return {
            "build_stats": self.build_stats,
            "data_summary": self.data_manager.get_data_summary(),
            "clustering_summary": self.user_clustering.get_profile_statistics(),
            "database_stats": self.neo4j_manager.get_database_stats()
        }
    
    def close(self):
        """关闭所有连接"""
        self.neo4j_manager.close()


def build_knowledge_graph_from_csv(csv_file_path: str,
                                 neo4j_uri: str = "bolt://localhost:7688",
                                 neo4j_user: str = "neo4j",
                                 neo4j_password: str = "neo4j123",
                                 neo4j_database: str = "neo4j",
                                 clustering_dir: str = "../02_User_Modeling/User_Preference_Clustering/outputs",
                                 clear_existing: bool = False) -> Dict:
    """从CSV文件构建知识图谱的便捷函数"""
    
    builder = KnowledgeGraphBuilder(
        csv_file_path=csv_file_path,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
        clustering_dir=clustering_dir
    )
    
    try:
        result = builder.build_knowledge_graph(clear_existing=clear_existing)
        return result
    finally:
        builder.close()


if __name__ == "__main__":
    # 测试知识图谱构建器
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 使用相对路径
    csv_path = "../02_User_Modeling/User_Preference_Clustering/outputs/user_dimension_vectors.csv"
    
    try:
        result = build_knowledge_graph_from_csv(
            csv_file_path=csv_path,
            neo4j_uri="bolt://localhost:7688",
            neo4j_user="neo4j",
            neo4j_password="neo4j123",
            clear_existing=False
        )
        
        print(f"构建结果: {result}")
        
    except Exception as e:
        print(f"构建失败: {e}")