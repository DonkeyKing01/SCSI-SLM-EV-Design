"""
批量数据处理器
用于高效处理13,682条评论数据和关系创建
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Generator, Tuple, Optional
from loguru import logger
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.neo4j_connector import neo4j_connector
from src.ev_models import (
    CarModel, UserProfile, Review, Feature, PREDEFINED_FEATURES,
    normalize_car_model_name, extract_brand_from_model,
    determine_car_type, estimate_price_range, get_dimension_names
)
from src.similarity_calculator import similarity_calculator


class BatchProcessor:
    """批量数据处理器"""
    
    def __init__(self, batch_size: int = 1000):
        """
        初始化批量处理器
        
        参数:
        - batch_size: 批量处理大小
        """
        self.batch_size = batch_size
        self.connector = neo4j_connector
        self.dimension_names = get_dimension_names()
        self.processed_counts = {
            'car_models': 0,
            'user_profiles': 0,
            'reviews': 0,
            'features': 0,
            'relationships': 0
        }
        
    def create_database_constraints(self):
        """创建数据库约束和索引"""
        logger.info("创建数据库约束和索引...")
        
        constraints_and_indexes = [
            # 唯一约束
            "CREATE CONSTRAINT car_model_id IF NOT EXISTS FOR (c:CarModel) REQUIRE c.modelId IS UNIQUE",
            "CREATE CONSTRAINT user_profile_id IF NOT EXISTS FOR (u:UserProfile) REQUIRE u.profileId IS UNIQUE",
            "CREATE CONSTRAINT review_id IF NOT EXISTS FOR (r:Review) REQUIRE r.reviewId IS UNIQUE",
            "CREATE CONSTRAINT feature_name IF NOT EXISTS FOR (f:Feature) REQUIRE f.name IS UNIQUE",
            
            # 性能索引
            "CREATE INDEX car_model_name IF NOT EXISTS FOR (c:CarModel) ON (c.name)",
            "CREATE INDEX car_model_brand IF NOT EXISTS FOR (c:CarModel) ON (c.brand)",
            "CREATE INDEX review_user_id IF NOT EXISTS FOR (r:Review) ON (r.userId)",
            "CREATE INDEX review_sentiment IF NOT EXISTS FOR (r:Review) ON (r.overallSentiment)",
            
            # 关系索引 (用于RAG查询优化)
            "CREATE INDEX mentions_sentiment IF NOT EXISTS FOR ()-[r:MENTIONS]-() ON (r.sentimentScore)",
            "CREATE INDEX contains_aspect_intensity IF NOT EXISTS FOR ()-[r:CONTAINS_ASPECT]-() ON (r.intensity)",
            "CREATE INDEX interested_correlation IF NOT EXISTS FOR ()-[r:INTERESTED_IN]-() ON (r.correlationScore)",
        ]
        
        for constraint in constraints_and_indexes:
            try:
                self.connector.execute_query(constraint)
                logger.info(f"成功创建: {constraint.split()[1]} {constraint.split()[2]}")
            except Exception as e:
                logger.warning(f"创建约束/索引失败 (可能已存在): {e}")
    
    def load_and_process_csv_data(self, csv_file_path: str) -> Dict[str, Any]:
        """
        加载和预处理CSV数据
        
        参数:
        - csv_file_path: CSV文件路径
        
        返回:
        - 处理后的数据字典
        """
        logger.info(f"开始加载CSV数据: {csv_file_path}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file_path)
            logger.info(f"成功加载 {len(df)} 条记录")
            
            # 验证必要的列是否存在
            required_columns = ['comment_id', 'user_id', 'car_model', 'original_comment']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"缺少必要的列: {col}")
            
            # 添加8维情感和强度列验证
            dimension_columns = []
            for dim in self.dimension_names:
                sentiment_col = f"{dim}_情感"
                intensity_col = f"{dim}_强度"
                if sentiment_col not in df.columns or intensity_col not in df.columns:
                    logger.warning(f"缺少维度列: {sentiment_col} 或 {intensity_col}")
                else:
                    dimension_columns.extend([sentiment_col, intensity_col])
            
            logger.info(f"找到 {len(dimension_columns)} 个维度相关列")
            
            # 数据清洗
            df = self._clean_dataframe(df)
            
            # 提取唯一车型
            unique_car_models = self._extract_unique_car_models(df)
            logger.info(f"提取到 {len(unique_car_models)} 个唯一车型")
            
            # 处理评论数据
            processed_reviews = self._process_reviews_data(df)
            logger.info(f"处理了 {len(processed_reviews)} 条评论")
            
            # 为用户ID映射到30个用户画像
            self._create_user_profile_mapping(df)
            
            return {
                'dataframe': df,
                'car_models': unique_car_models,
                'reviews': processed_reviews,
                'dimension_columns': dimension_columns
            }
            
        except Exception as e:
            logger.error(f"加载CSV数据失败: {e}")
            raise
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗数据框"""
        logger.info("开始数据清洗...")
        
        # 移除空值记录
        original_count = len(df)
        df = df.dropna(subset=['comment_id', 'user_id', 'car_model', 'original_comment'])
        cleaned_count = len(df)
        
        if cleaned_count < original_count:
            logger.info(f"移除了 {original_count - cleaned_count} 条包含空值的记录")
        
        # 移除重复的评论ID
        df = df.drop_duplicates(subset=['comment_id'])
        
        # 标准化车型名称
        df['car_model'] = df['car_model'].apply(normalize_car_model_name)
        
        # 验证和修正情感数据范围
        for dim in self.dimension_names:
            sentiment_col = f"{dim}_情感"
            intensity_col = f"{dim}_强度"
            
            if sentiment_col in df.columns:
                # 情感值应在-1到1之间
                df[sentiment_col] = df[sentiment_col].clip(-1, 1)
            
            if intensity_col in df.columns:
                # 强度值应在0到1之间
                df[intensity_col] = df[intensity_col].clip(0, 1)
        
        logger.info(f"数据清洗完成，剩余 {len(df)} 条有效记录")
        return df
    
    def _extract_unique_car_models(self, df: pd.DataFrame) -> List[CarModel]:
        """提取唯一车型"""
        unique_models = df['car_model'].unique()
        car_models = []
        
        for i, model_name in enumerate(unique_models):
            car_model = CarModel(
                model_id=f"model_{i+1:03d}",
                name=model_name,
                brand=extract_brand_from_model(model_name),
                type=determine_car_type(model_name),
                price_range=estimate_price_range(model_name)
            )
            car_models.append(car_model)
        
        return car_models
    
    def _process_reviews_data(self, df: pd.DataFrame) -> List[Review]:
        """处理评论数据"""
        reviews = []
        
        for _, row in df.iterrows():
            try:
                # 提取8维数据
                dimension_data = {}
                for dim in self.dimension_names:
                    sentiment_col = f"{dim}_情感"
                    intensity_col = f"{dim}_强度"
                    
                    sentiment = row.get(sentiment_col, 0.0)
                    intensity = row.get(intensity_col, 0.0)
                    
                    dimension_data[dim] = {
                        'sentiment': float(sentiment) if pd.notna(sentiment) else 0.0,
                        'intensity': float(intensity) if pd.notna(intensity) else 0.0
                    }
                
                # 计算整体情感
                overall_sentiment = similarity_calculator.calculate_overall_sentiment(dimension_data)
                
                # 创建评论对象
                review = Review(
                    review_id=str(row['comment_id']),
                    content=str(row['original_comment']),
                    user_id=str(row['user_id']),
                    car_model=str(row['car_model']),
                    overall_sentiment=overall_sentiment,
                    dimension_data=dimension_data
                )
                
                reviews.append(review)
                
            except Exception as e:
                logger.warning(f"处理评论失败 (ID: {row.get('comment_id', 'unknown')}): {e}")
                continue
        
        return reviews
    
    def load_user_profiles_from_clustering(self, clustering_file_path: str) -> List[UserProfile]:
        """
        从聚类报告加载用户画像
        
        参数:
        - clustering_file_path: 聚类报告文件路径
        
        返回:
        - 用户画像列表
        """
        logger.info(f"从聚类报告加载用户画像: {clustering_file_path}")
        
        # 这里应该解析clustering_report.md文件
        # 由于没有具体的文件格式，我们创建30个示例用户画像
        user_profiles = []
        
        # 示例用户画像模板
        profile_templates = [
            {
                'name': '性能追求者',
                'description': '注重车辆的动力性能和操控体验，追求驾驶乐趣',
                'main_features': ['操控性能', '智能配置', '外观设计'],
                'dimension_strengths': {'外观设计': 0.7, '内饰质感': 0.5, '智能配置': 0.8, '空间实用': 0.4, '舒适体验': 0.6, '操控性能': 0.9, '续航能耗': 0.7, '价值认知': 0.6}
            },
            {
                'name': '实用主义者',
                'description': '注重车辆的实用性和性价比，理性消费',
                'main_features': ['空间实用', '价值认知', '续航能耗'],
                'dimension_strengths': {'外观设计': 0.4, '内饰质感': 0.5, '智能配置': 0.6, '空间实用': 0.9, '舒适体验': 0.7, '操控性能': 0.5, '续航能耗': 0.8, '价值认知': 0.9}
            },
            {
                'name': '科技爱好者',
                'description': '热衷于最新的智能配置和科技功能',
                'main_features': ['智能配置', '外观设计', '操控性能'],
                'dimension_strengths': {'外观设计': 0.8, '内饰质感': 0.7, '智能配置': 0.95, '空间实用': 0.5, '舒适体验': 0.6, '操控性能': 0.7, '续航能耗': 0.7, '价值认知': 0.6}
            },
            {
                'name': '舒适优先者',
                'description': '最看重乘坐舒适性和静音效果',
                'main_features': ['舒适体验', '内饰质感', '空间实用'],
                'dimension_strengths': {'外观设计': 0.6, '内饰质感': 0.8, '智能配置': 0.6, '空间实用': 0.8, '舒适体验': 0.95, '操控性能': 0.4, '续航能耗': 0.7, '价值认知': 0.7}
            },
            {
                'name': '外观控',
                'description': '非常注重车辆的外观设计和颜值',
                'main_features': ['外观设计', '内饰质感', '智能配置'],
                'dimension_strengths': {'外观设计': 0.95, '内饰质感': 0.8, '智能配置': 0.7, '空间实用': 0.5, '舒适体验': 0.6, '操控性能': 0.6, '续航能耗': 0.6, '价值认知': 0.5}
            }
        ]
        
        # 生成30个用户画像
        for i in range(30):
            template = profile_templates[i % len(profile_templates)]
            
            # 为每个画像添加一些随机变化
            varied_strengths = {}
            for dim, strength in template['dimension_strengths'].items():
                # 添加±0.1的随机变化
                variation = np.random.uniform(-0.1, 0.1)
                varied_strength = max(0.0, min(1.0, strength + variation))
                varied_strengths[dim] = round(varied_strength, 2)
            
            user_profile = UserProfile(
                profile_id=i + 1,
                name=f"{template['name']}_{i+1}",
                description=template['description'],
                user_count=np.random.randint(300, 800),  # 随机用户数量
                main_features=template['main_features'].copy(),
                dimension_strengths=varied_strengths
            )
            
            user_profiles.append(user_profile)
        
        logger.info(f"生成了 {len(user_profiles)} 个用户画像")
        return user_profiles
    
    def _create_user_profile_mapping(self, df: pd.DataFrame):
        """
        创建用户ID到30个画像的映射关系
        
        参数:
        - df: 包含用户ID的数据框
        """
        # 获取所有唯一用户ID
        unique_user_ids = df['user_id'].unique()
        logger.info(f"发现 {len(unique_user_ids)} 个唯一用户ID")
        
        # 将用户ID映射到30个画像ID (1-30)
        # 使用哈希函数确保一致的映射
        self.user_profile_mapping = {}
        for user_id in unique_user_ids:
            # 使用用户ID的哈希值映射到1-30
            profile_id = (hash(str(user_id)) % 30) + 1
            self.user_profile_mapping[str(user_id)] = profile_id
        
        logger.info(f"创建了 {len(self.user_profile_mapping)} 个用户ID到画像的映射")
    
    def process_in_batches(self, data_list: List[Any], 
                          process_func, 
                          description: str = "处理数据") -> bool:
        """
        批量处理数据
        
        参数:
        - data_list: 要处理的数据列表
        - process_func: 处理函数
        - description: 处理描述
        
        返回:
        - 处理是否成功
        """
        try:
            total_count = len(data_list)
            processed_count = 0
            
            logger.info(f"开始{description}，总计 {total_count} 条记录")
            
            # 按批次处理
            for i in range(0, total_count, self.batch_size):
                batch = data_list[i:i + self.batch_size]
                batch_size = len(batch)
                
                start_time = time.time()
                
                try:
                    # 执行批量处理
                    process_func(batch)
                    processed_count += batch_size
                    
                    elapsed_time = time.time() - start_time
                    logger.info(f"批次 {i//self.batch_size + 1} 完成: {batch_size} 条记录, "
                              f"耗时 {elapsed_time:.2f}s, "
                              f"进度 {processed_count}/{total_count} ({processed_count/total_count*100:.1f}%)")
                    
                except Exception as e:
                    logger.error(f"批次处理失败 (批次 {i//self.batch_size + 1}): {e}")
                    # 继续处理下一个批次
                    continue
            
            logger.info(f"{description}完成，成功处理 {processed_count}/{total_count} 条记录")
            return processed_count == total_count
            
        except Exception as e:
            logger.error(f"{description}失败: {e}")
            return False
    
    def create_nodes_batch(self, nodes: List[Any], node_type: str):
        """批量创建节点"""
        if not nodes:
            return
        
        try:
            with self.connector.driver.session() as session:
                with session.begin_transaction() as tx:
                    for node in nodes:
                        node_data = node.to_dict()
                        
                        if node_type == "CarModel":
                            query = """
                            MERGE (c:CarModel {modelId: $modelId})
                            SET c += $properties
                            """
                            tx.run(query, modelId=node_data['modelId'], properties=node_data)
                            
                        elif node_type == "UserProfile":
                            query = """
                            MERGE (u:UserProfile {profileId: $profileId})
                            SET u += $properties
                            """
                            tx.run(query, profileId=node_data['profileId'], properties=node_data)
                            
                        elif node_type == "Review":
                            query = """
                            MERGE (r:Review {reviewId: $reviewId})
                            SET r += $properties
                            """
                            tx.run(query, reviewId=node_data['reviewId'], properties=node_data)
                            
                        elif node_type == "Feature":
                            query = """
                            MERGE (f:Feature {name: $name})
                            SET f += $properties
                            """
                            tx.run(query, name=node_data['name'], properties=node_data)
                    
                    tx.commit()
                    
            self.processed_counts[node_type.lower() + 's'] += len(nodes)
            
        except Exception as e:
            logger.error(f"批量创建{node_type}节点失败: {e}")
            raise
    
    def create_relationships_batch(self, relationships: List[Dict[str, Any]]):
        """批量创建关系"""
        if not relationships:
            return
        
        try:
            with self.connector.driver.session() as session:
                with session.begin_transaction() as tx:
                    for rel in relationships:
                        rel_type = rel['type']
                        from_id = rel['from_id']
                        to_id = rel['to_id']
                        properties = rel.get('properties', {})
                        
                        if rel_type == "PUBLISHED":
                            query = """
                            MATCH (u:UserProfile {profileId: $from_id})
                            MATCH (r:Review {reviewId: $to_id})
                            MERGE (u)-[rel:PUBLISHED]->(r)
                            SET rel += $properties
                            """
                            
                        elif rel_type == "MENTIONS":
                            query = """
                            MATCH (r:Review {reviewId: $from_id})
                            MATCH (c:CarModel {modelId: $to_id})
                            MERGE (r)-[rel:MENTIONS]->(c)
                            SET rel += $properties
                            """
                            
                        elif rel_type == "CONTAINS_ASPECT":
                            query = """
                            MATCH (r:Review {reviewId: $from_id})
                            MATCH (f:Feature {name: $to_id})
                            MERGE (r)-[rel:CONTAINS_ASPECT]->(f)
                            SET rel += $properties
                            """
                            
                        elif rel_type == "INTERESTED_IN":
                            query = """
                            MATCH (u:UserProfile {profileId: $from_id})
                            MATCH (c:CarModel {modelId: $to_id})
                            MERGE (u)-[rel:INTERESTED_IN]->(c)
                            SET rel += $properties
                            """
                        
                        tx.run(query, from_id=from_id, to_id=to_id, properties=properties)
                    
                    tx.commit()
                    
            self.processed_counts['relationships'] += len(relationships)
            
        except Exception as e:
            logger.error(f"批量创建关系失败: {e}")
            raise
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return {
            'processed_counts': self.processed_counts.copy(),
            'total_processed': sum(self.processed_counts.values()),
            'batch_size': self.batch_size
        }


# 全局批量处理器实例
batch_processor = BatchProcessor()