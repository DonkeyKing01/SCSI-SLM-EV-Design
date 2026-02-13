#!/usr/bin/env python3
"""
向量数据加载脚本
加载车型数据和用户画像数据到Chroma向量存储
同时为Neo4j中的评论节点添加嵌入向量
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from tools.vector_tool import VectorTool
from database.neo4j_connection import neo4j_conn
from config.settings import Settings
from langchain_openai import OpenAIEmbeddings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorDataLoader:
    """向量数据加载器"""
    
    def __init__(self):
        """初始化加载器"""
        self.vector_tool = VectorTool()
        self.neo4j_graph = neo4j_conn.get_graph()
        
        # 初始化OpenAI嵌入模型
        openai_config = Settings.get_openai_config()
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_config['api_key'],
            openai_api_base=openai_config['base_url'],
            model=Settings.EMBEDDING_MODEL
        )
        
        logger.info("VectorDataLoader初始化完成")
    
    def load_all_data(self):
        """加载所有向量数据"""
        try:
            logger.info("🚀 开始加载向量数据")
            
            # 1. 加载车型特征数据到Chroma
            car_model_path = "../02_User_Modeling/Product_IPA_Analysis/outputs/car_model_scores.csv"
            if os.path.exists(car_model_path):
                logger.info("正在加载车型特征数据...")
                car_count = self.vector_tool.load_car_model_data(car_model_path)
                logger.info(f"✅ 成功加载 {car_count} 个车型数据")
            else:
                logger.warning(f"车型数据文件不存在: {car_model_path}")
            
            # 2. 加载用户画像数据到Chroma
            persona_path = "../02_User_Modeling/User_Preference_Clustering/outputs/user_vector_matrix.csv"
            if os.path.exists(persona_path):
                logger.info("正在加载用户画像数据...")
                persona_count = self.vector_tool.load_user_persona_data(persona_path)
                logger.info(f"✅ 成功加载 {persona_count} 个用户画像数据")
            else:
                logger.warning(f"用户画像数据文件不存在: {persona_path}")
            
            # 3. 为Neo4j评论节点添加嵌入向量
            logger.info("正在为Neo4j评论节点添加嵌入向量...")
            embedding_count = self.add_embeddings_to_reviews()
            logger.info(f"✅ 成功为 {embedding_count} 个评论添加嵌入向量")
            
            logger.info("🎉 所有向量数据加载完成！")
            
        except Exception as e:
            logger.error(f"❌ 向量数据加载失败: {e}")
            raise
    
    def add_embeddings_to_reviews(self) -> int:
        """为Neo4j中的评论节点添加嵌入向量"""
        try:
            # 批量处理评论，避免内存问题，注意API批次限制
            batch_size = 25
            total_count = 0
            
            # 获取所有没有嵌入向量的评论
            query = """
            MATCH (r:Review)
            WHERE r.embedding IS NULL
            RETURN ID(r) as review_id, r.content as content
            ORDER BY ID(r)
            """
            
            logger.info("获取需要处理的评论...")
            result = self.neo4j_graph.query(query)
            reviews = [{"review_id": record["review_id"], "content": record["content"]} 
                      for record in result]
            
            logger.info(f"找到 {len(reviews)} 个需要添加嵌入向量的评论")
            
            # 分批处理
            for i in range(0, len(reviews), batch_size):
                batch = reviews[i:i + batch_size]
                batch_embeddings = self._generate_embeddings_batch(batch)
                
                # 更新Neo4j中的评论节点
                for review, embedding in zip(batch, batch_embeddings):
                    update_query = """
                    MATCH (r:Review) WHERE ID(r) = $review_id
                    SET r.embedding = $embedding
                    """
                    self.neo4j_graph.query(
                        update_query, 
                        {
                            "review_id": review["review_id"],
                            "embedding": embedding
                        }
                    )
                
                total_count += len(batch)
                logger.info(f"已处理 {total_count}/{len(reviews)} 个评论")
            
            return total_count
            
        except Exception as e:
            logger.error(f"为评论添加嵌入向量失败: {e}")
            raise
    
    def _generate_embeddings_batch(self, reviews: List[Dict]) -> List[List[float]]:
        """批量生成嵌入向量"""
        try:
            # 截断评论内容，确保不超过token限制
            contents = []
            for review in reviews:
                content = review["content"]
                # 更保守的截断：1个中文字符约等于1.5个token，截断到200字符以确保安全
                # 512 tokens / 1.5 ≈ 340 characters，但要留出一些安全空间
                if len(content) > 200:
                    content = content[:200] + "..."
                    logger.debug(f"截断长评论，原长度: {len(review['content'])}, 截断后: {len(content)}")
                contents.append(content)
            
            # 记录最长内容的长度用于调试
            max_len = max(len(c) for c in contents) if contents else 0
            logger.debug(f"批次中最长内容长度: {max_len} 字符")
            
            embeddings = self.embeddings.embed_documents(contents)
            return embeddings
            
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
            # 如果还是失败，尝试更激进的截断
            if "512 tokens" in str(e):
                logger.warning("检测到token限制错误，尝试更激进的截断...")
                contents = []
                for review in reviews:
                    content = review["content"]
                    # 更激进的截断到100字符
                    if len(content) > 100:
                        content = content[:100] + "..."
                    contents.append(content)
                try:
                    embeddings = self.embeddings.embed_documents(contents)
                    logger.warning("使用激进截断成功生成嵌入向量")
                    return embeddings
                except Exception as e2:
                    logger.error(f"即使使用激进截断也失败: {e2}")
                    raise
            raise
    
    def check_vector_store_status(self):
        """检查向量存储状态"""
        try:
            logger.info("📊 检查向量存储状态...")
            
            # 检查Chroma向量存储
            if self.vector_tool.vector_store is not None:
                # 尝试获取集合统计信息
                try:
                    collection = self.vector_tool.vector_store._collection
                    count = collection.count()
                    logger.info(f"Chroma向量存储: {count} 个文档")
                except:
                    logger.info("Chroma向量存储: 已初始化，但无法获取文档数量")
            else:
                logger.warning("Chroma向量存储: 未初始化")
            
            # 检查Neo4j评论嵌入向量
            query = """
            MATCH (r:Review)
            WHERE r.embedding IS NOT NULL
            RETURN count(r) as count
            """
            result = self.neo4j_graph.query(query)
            neo4j_embedding_count = result[0]["count"] if result else 0
            logger.info(f"Neo4j评论嵌入向量: {neo4j_embedding_count} 个")
            
            # 检查总评论数
            query = "MATCH (r:Review) RETURN count(r) as total"
            result = self.neo4j_graph.query(query)
            total_reviews = result[0]["total"] if result else 0
            logger.info(f"Neo4j总评论数: {total_reviews} 个")
            
        except Exception as e:
            logger.error(f"检查向量存储状态失败: {e}")

def main():
    """主函数"""
    try:
        loader = VectorDataLoader()
        
        # 检查当前状态
        loader.check_vector_store_status()
        
        # 加载数据
        loader.load_all_data()
        
        # 再次检查状态
        logger.info("\n" + "="*50)
        logger.info("最终状态检查:")
        loader.check_vector_store_status()
        
    except Exception as e:
        logger.error(f"数据加载过程失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()