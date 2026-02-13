"""
相似度计算和评分算法模块
用于新能源汽车知识图谱的关系权重计算
"""
import numpy as np
import math
from typing import Dict, List, Tuple, Any
from loguru import logger


class SimilarityCalculator:
    """相似度计算器"""
    
    def __init__(self):
        self.dimension_names = [
            "外观设计", "内饰质感", "智能配置", "空间实用",
            "舒适体验", "操控性能", "续航能耗", "价值认知"
        ]
    
    def calculate_user_match_score(self, user_profile_vector: List[float], 
                                 review_vector: List[float]) -> float:
        """
        计算用户画像与评论的匹配度 (余弦相似度)
        
        参数:
        - user_profile_vector: 用户画像的8维强度向量
        - review_vector: 评论的8维强度向量
        
        返回:
        - 余弦相似度分数 (0-1)
        """
        try:
            # 转换为numpy数组
            user_vec = np.array(user_profile_vector, dtype=float)
            review_vec = np.array(review_vector, dtype=float)
            
            # 检查向量长度
            if len(user_vec) != 8 or len(review_vec) != 8:
                logger.warning(f"向量长度不正确: user={len(user_vec)}, review={len(review_vec)}")
                return 0.0
            
            # 计算向量的模长
            user_norm = np.linalg.norm(user_vec)
            review_norm = np.linalg.norm(review_vec)
            
            # 避免除零错误
            if user_norm == 0 or review_norm == 0:
                return 0.0
            
            # 计算余弦相似度
            dot_product = np.dot(user_vec, review_vec)
            cosine_similarity = dot_product / (user_norm * review_norm)
            
            # 将结果映射到0-1范围 (余弦相似度范围是-1到1)
            normalized_score = (cosine_similarity + 1) / 2
            
            # 确保结果在0-1范围内
            return max(0.0, min(1.0, normalized_score))
            
        except Exception as e:
            logger.error(f"计算用户匹配分数失败: {e}")
            return 0.0
    
    def calculate_overall_sentiment(self, dimension_data: Dict[str, Dict[str, float]]) -> float:
        """
        计算评论的整体情感倾向 (加权平均)
        
        参数:
        - dimension_data: 包含8维强度和情感的字典
          格式: {"外观设计": {"intensity": 0.8, "sentiment": 0.5}, ...}
        
        返回:
        - 加权平均情感分数 (-1到1)
        """
        try:
            total_weight = 0.0
            weighted_sum = 0.0
            
            for dimension in self.dimension_names:
                if dimension in dimension_data:
                    data = dimension_data[dimension]
                    intensity = data.get('intensity', 0.0)
                    sentiment = data.get('sentiment', 0.0)
                    
                    # 验证数据范围
                    if not (0.0 <= intensity <= 1.0):
                        logger.warning(f"强度值超出范围: {dimension} = {intensity}")
                        continue
                    
                    if not (-1.0 <= sentiment <= 1.0):
                        logger.warning(f"情感值超出范围: {dimension} = {sentiment}")
                        continue
                    
                    total_weight += intensity
                    weighted_sum += intensity * sentiment
            
            # 避免除零错误
            if total_weight == 0:
                return 0.0
            
            overall_sentiment = weighted_sum / total_weight
            
            # 确保结果在-1到1范围内
            return max(-1.0, min(1.0, overall_sentiment))
            
        except Exception as e:
            logger.error(f"计算整体情感倾向失败: {e}")
            return 0.0
    
    def calculate_importance_score(self, review_content: str, 
                                 dimension_data: Dict[str, Dict[str, float]]) -> float:
        """
        计算评论的重要性评分
        
        参数:
        - review_content: 评论内容
        - dimension_data: 8维数据，用于计算关键词密度
        
        返回:
        - 重要性分数 (0-1)
        """
        try:
            # 1. 评论长度评分 (0-1)
            # 假设500字符为满分长度
            length_score = min(len(review_content) / 500.0, 1.0)
            
            # 2. 维度覆盖度评分 (0-1)
            # 计算有效维度的数量 (强度>0.1的维度)
            valid_dimensions = 0
            total_intensity = 0.0
            
            for dimension in self.dimension_names:
                if dimension in dimension_data:
                    intensity = dimension_data[dimension].get('intensity', 0.0)
                    if intensity > 0.1:  # 阈值过滤
                        valid_dimensions += 1
                        total_intensity += intensity
            
            coverage_score = valid_dimensions / len(self.dimension_names)
            
            # 3. 平均强度评分 (0-1)
            avg_intensity = total_intensity / max(valid_dimensions, 1)
            
            # 4. 综合重要性评分
            # 长度权重40%，覆盖度权重35%，强度权重25%
            importance = (length_score * 0.4 + 
                         coverage_score * 0.35 + 
                         avg_intensity * 0.25)
            
            return max(0.0, min(1.0, importance))
            
        except Exception as e:
            logger.error(f"计算重要性评分失败: {e}")
            return 0.0
    
    def calculate_correlation_score(self, user_match_scores: List[float],
                                  sentiment_scores: List[float],
                                  importance_scores: List[float]) -> float:
        """
        计算用户画像对车型的综合相关性评分
        
        参数:
        - user_match_scores: 用户匹配分数列表
        - sentiment_scores: 情感分数列表  
        - importance_scores: 重要性分数列表
        
        返回:
        - 综合相关性评分 (0-1)
        """
        try:
            if not user_match_scores or len(user_match_scores) == 0:
                return 0.0
            
            # 确保所有列表长度一致
            min_length = min(len(user_match_scores), len(sentiment_scores), len(importance_scores))
            
            if min_length == 0:
                return 0.0
            
            # 截取到相同长度
            user_scores = user_match_scores[:min_length]
            sentiment_scores = sentiment_scores[:min_length]
            importance_scores = importance_scores[:min_length]
            
            # 计算加权评分
            total_score = 0.0
            total_weight = 0.0
            
            for i in range(min_length):
                user_match = user_scores[i]
                sentiment = sentiment_scores[i]
                importance = importance_scores[i]
                
                # 将情感分数从-1~1映射到0~1
                normalized_sentiment = (sentiment + 1) / 2
                
                # 计算单个评论的综合分数
                review_score = user_match * normalized_sentiment * importance
                
                total_score += review_score
                total_weight += importance  # 使用重要性作为权重
            
            # 计算加权平均
            if total_weight == 0:
                return 0.0
            
            correlation_score = total_score / total_weight
            
            return max(0.0, min(1.0, correlation_score))
            
        except Exception as e:
            logger.error(f"计算相关性评分失败: {e}")
            return 0.0
    
    def calculate_aspect_statistics(self, reviews_data: List[Dict[str, Any]], 
                                  feature_name: str) -> Dict[str, Any]:
        """
        计算特定特征维度的统计信息
        
        参数:
        - reviews_data: 评论数据列表
        - feature_name: 特征名称
        
        返回:
        - 统计信息字典
        """
        try:
            sentiments = []
            intensities = []
            valid_reviews = 0
            
            for review in reviews_data:
                dimension_data = review.get('dimension_data', {})
                if feature_name in dimension_data:
                    data = dimension_data[feature_name]
                    sentiment = data.get('sentiment', 0.0)
                    intensity = data.get('intensity', 0.0)
                    
                    # 只统计有效的数据 (强度>0.1)
                    if intensity > 0.1:
                        sentiments.append(sentiment)
                        intensities.append(intensity)
                        valid_reviews += 1
            
            if valid_reviews == 0:
                return {
                    'avg_sentiment': 0.0,
                    'avg_intensity': 0.0,
                    'sentiment_std': 0.0,
                    'intensity_std': 0.0,
                    'mention_count': 0,
                    'positive_ratio': 0.0
                }
            
            # 计算统计指标
            avg_sentiment = np.mean(sentiments)
            avg_intensity = np.mean(intensities)
            sentiment_std = np.std(sentiments)
            intensity_std = np.std(intensities)
            
            # 计算正面评价比例
            positive_count = sum(1 for s in sentiments if s > 0.1)
            positive_ratio = positive_count / valid_reviews
            
            return {
                'avg_sentiment': float(avg_sentiment),
                'avg_intensity': float(avg_intensity),
                'sentiment_std': float(sentiment_std),
                'intensity_std': float(intensity_std),
                'mention_count': valid_reviews,
                'positive_ratio': float(positive_ratio)
            }
            
        except Exception as e:
            logger.error(f"计算特征统计信息失败: {e}")
            return {
                'avg_sentiment': 0.0,
                'avg_intensity': 0.0,
                'sentiment_std': 0.0,
                'intensity_std': 0.0,
                'mention_count': 0,
                'positive_ratio': 0.0
            }
    
    def find_similar_profiles(self, target_profile: Dict[str, float],
                            all_profiles: List[Dict[str, Any]],
                            threshold: float = 0.8) -> List[Tuple[int, float]]:
        """
        找到相似的用户画像
        
        参数:
        - target_profile: 目标用户画像的8维向量
        - all_profiles: 所有用户画像列表
        - threshold: 相似度阈值
        
        返回:
        - 相似画像列表 [(profile_id, similarity_score), ...]
        """
        try:
            target_vector = [target_profile.get(dim, 0.0) for dim in self.dimension_names]
            similar_profiles = []
            
            for profile in all_profiles:
                profile_id = profile.get('profile_id')
                dimension_strengths = profile.get('dimension_strengths', {})
                
                profile_vector = [dimension_strengths.get(dim, 0.0) for dim in self.dimension_names]
                
                # 计算相似度
                similarity = self.calculate_user_match_score(target_vector, profile_vector)
                
                if similarity >= threshold:
                    similar_profiles.append((profile_id, similarity))
            
            # 按相似度降序排序
            similar_profiles.sort(key=lambda x: x[1], reverse=True)
            
            return similar_profiles
            
        except Exception as e:
            logger.error(f"查找相似画像失败: {e}")
            return []


# 全局相似度计算器实例
similarity_calculator = SimilarityCalculator()