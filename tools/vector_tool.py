"""
向量检索工具模块
用于基于用户查询进行向量相似度搜索
"""
import os
from typing import List, Dict, Any
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd
import numpy as np
from config.settings import Settings
import logging

logger = logging.getLogger(__name__)

class VectorTool:
    """向量检索工具类"""
    
    def __init__(self):
        """初始化向量工具"""
        self.embeddings = None
        self.vector_store = None
        self.text_splitter = None
        self._initialize()
    
    def _initialize(self):
        """初始化组件"""
        try:
            # 初始化嵌入模型
            openai_config = Settings.get_openai_config()
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=openai_config['api_key'],
                openai_api_base=openai_config['base_url'],
                model=Settings.EMBEDDING_MODEL
            )
            
            # 初始化文本分割器 - 使用更小的chunk size以适应嵌入模型限制
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200,  # 进一步减小chunk size，确保不超过512 tokens
                chunk_overlap=20,
                length_function=len,
            )
            
            # 初始化或加载向量存储
            self._setup_vector_store()
            
            logger.info("VectorTool初始化成功")
            
        except Exception as e:
            logger.error(f"VectorTool初始化失败: {e}")
            raise
    
    def _setup_vector_store(self):
        """设置向量存储"""
        persist_directory = Settings.VECTOR_STORE_PATH
        
        if os.path.exists(persist_directory):
            # 加载现有的向量存储
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            logger.info("已加载现有向量存储")
        else:
            # 创建新的向量存储
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            logger.info("创建新的向量存储")
    
    def load_car_model_data(self, car_model_path: str):
        """加载车型特征数据"""
        try:
            df = pd.read_csv(car_model_path)
            documents = []
            
            for _, row in df.iterrows():
                # 构建车型特征文档
                content = self._build_car_model_content(row)
                doc = Document(
                    page_content=content,
                    metadata={
                        'type': 'car_model',
                        'car_model': row['car_model'],
                        'review_count': row['review_count'],
                        'source': 'Model_New'
                    }
                )
                documents.append(doc)
            
            # 分割文档并分批添加到向量存储
            texts = self.text_splitter.split_documents(documents)
            self._add_documents_in_batches(texts)
            self.vector_store.persist()
            
            logger.info(f"成功加载 {len(documents)} 个车型数据")
            return len(documents)
            
        except Exception as e:
            logger.error(f"加载车型数据失败: {e}")
            raise
    
    def load_user_persona_data(self, persona_path: str):
        """加载用户画像数据"""
        try:
            df = pd.read_csv(persona_path)
            documents = []
            
            # 按车型分组统计用户画像
            for car_model, group in df.groupby('car_model'):
                content = self._build_user_persona_content(car_model, group)
                doc = Document(
                    page_content=content,
                    metadata={
                        'type': 'user_persona',
                        'car_model': car_model,
                        'user_count': len(group),
                        'source': 'Persona_New'
                    }
                )
                documents.append(doc)
            
            # 分割文档并分批添加到向量存储
            texts = self.text_splitter.split_documents(documents)
            self._add_documents_in_batches(texts)
            self.vector_store.persist()
            
            logger.info(f"成功加载 {len(documents)} 个用户画像数据")
            return len(documents)
            
        except Exception as e:
            logger.error(f"加载用户画像数据失败: {e}")
            raise
    
    def _build_car_model_content(self, row: pd.Series) -> str:
        """构建车型特征内容"""
        dimensions = ['外观设计', '内饰质感', '智能配置', '空间实用', '舒适体验', '操控性能', '续航能耗', '价值认知']
        
        content = f"车型: {row['car_model']}\n"
        content += f"评论数量: {row['review_count']}\n\n"
        content += "车型特征评分分析:\n"
        
        for dim in dimensions:
            performance = row.get(f'{dim}_performance', 0)
            importance = row.get(f'{dim}_importance', 0)
            mention_rate = row.get(f'{dim}_mention_rate', 0)
            
            content += f"- {dim}: 绩效评分 {performance:.3f}, 重要度 {importance:.3f}, 提及率 {mention_rate:.1%}\n"
        
        # 添加IPA分析建议
        content += "\n基于IPA分析的改进建议:\n"
        for dim in dimensions:
            performance = row.get(f'{dim}_performance', 0)
            importance = row.get(f'{dim}_importance', 0)
            
            if importance > 0.12 and performance > 0.11:
                content += f"- {dim}: 优势保持区，继续发挥优势\n"
            elif importance > 0.12 and performance <= 0.11:
                content += f"- {dim}: 集中改进区，需要重点提升\n"
            elif importance <= 0.12 and performance <= 0.11:
                content += f"- {dim}: 低优先级区，维持现状\n"
            else:
                content += f"- {dim}: 过度投入区，可适度调整资源分配\n"
        
        return content
    
    def _build_user_persona_content(self, car_model: str, group: pd.DataFrame) -> str:
        """构建用户画像内容"""
        dimensions = ['外观设计', '内饰质感', '智能配置', '空间实用', '舒适体验', '操控性能', '续航能耗', '价值认知']
        
        content = f"车型: {car_model}\n"
        content += f"用户样本数量: {len(group)}\n\n"
        content += "用户关注度统计:\n"
        
        # 计算各维度的平均关注度
        for dim in dimensions:
            avg_strength = group[dim].mean()
            active_users = (group[dim] > 0).sum()
            content += f"- {dim}: 平均关注强度 {avg_strength:.3f}, 活跃用户数 {active_users} ({active_users/len(group):.1%})\n"
        
        # 识别主要用户类型
        content += "\n主要用户画像类型:\n"
        for _, row in group.head(5).iterrows():
            top_dims = []
            for dim in dimensions:
                if row[dim] > 0.5:
                    top_dims.append(f"{dim}({row[dim]:.2f})")
            
            if top_dims:
                content += f"- 关注: {', '.join(top_dims)}\n"
        
        return content
    
    def search(self, query: str, k: int = 5, filter_dict: Dict = None) -> List[Document]:
        """执行向量搜索"""
        try:
            if not self.vector_store:
                logger.warning("向量存储未初始化")
                return []
            
            # 执行相似度搜索
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            logger.info(f"向量搜索返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    def search_with_scores(self, query: str, k: int = 5, filter_dict: Dict = None) -> List[tuple]:
        """执行带评分的向量搜索"""
        try:
            if not self.vector_store:
                logger.warning("向量存储未初始化")
                return []
            
            # 执行相似度搜索并返回评分
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            logger.info(f"向量搜索返回 {len(results)} 个带评分结果")
            return results
            
        except Exception as e:
            logger.error(f"带评分向量搜索失败: {e}")
            return []
    
    def get_relevant_context(self, query: str, max_context_length: int = 3000) -> str:
        """获取与查询相关的上下文"""
        try:
            # 搜索相关文档
            results = self.search(query, k=10)
            
            if not results:
                return "暂无相关数据。"
            
            # 构建上下文
            context_parts = []
            current_length = 0
            
            for doc in results:
                content = doc.page_content
                if current_length + len(content) <= max_context_length:
                    context_parts.append(content)
                    current_length += len(content)
                else:
                    # 截断最后一个文档以不超过长度限制
                    remaining_length = max_context_length - current_length
                    if remaining_length > 100:  # 至少保留100字符
                        context_parts.append(content[:remaining_length] + "...")
                    break
            
            context = "\n\n".join(context_parts)
            logger.info(f"构建了长度为 {len(context)} 的上下文")
            
            return context
            
        except Exception as e:
            logger.error(f"获取相关上下文失败: {e}")
            return "获取上下文时发生错误。"
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取向量存储统计信息"""
        try:
            if not self.vector_store:
                return {'status': 'not_initialized'}
            
            # 这里可以添加更多统计信息
            return {
                'status': 'initialized',
                'store_path': Settings.VECTOR_STORE_PATH
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _add_documents_in_batches(self, documents: List[Document], batch_size: int = 10):
        """分批添加文档到向量存储"""
        try:
            total_docs = len(documents)
            added_count = 0
            
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                self.vector_store.add_documents(batch)
                added_count += len(batch)
                logger.info(f"已添加 {added_count}/{total_docs} 个文档到向量存储")
            
            logger.info(f"分批添加完成，总计 {total_docs} 个文档")
            
        except Exception as e:
            logger.error(f"分批添加文档失败: {e}")
            raise

# 全局实例
vector_tool = VectorTool()