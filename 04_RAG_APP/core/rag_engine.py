"""
RAG系统核心引擎
整合向量检索、图谱查询和LLM生成的完整RAG流程
"""
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from tools.vector_tool import vector_tool
from tools.vector_graph_tool import vector_graph_tool
from tools.graph_cypher_tool import graph_cypher_tool
from config.settings import Settings
from config.search_config import SearchConfig, SearchMode, QuestionType
from core.question_analyzer import question_analyzer
import logging
import json
import uuid
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class RAGEngine:
    """RAG系统核心引擎"""
    
    def __init__(self):
        """初始化RAG引擎"""
        self.llm = None
        self.prompt_template = None
        self.config = SearchConfig()
        self.analyzer = question_analyzer
        self._initialize()
    
    def _initialize(self):
        """初始化组件"""
        try:
            # 初始化LLM
            openai_config = Settings.get_openai_config()
            self.llm = ChatOpenAI(
                openai_api_key=openai_config['api_key'],
                openai_api_base=openai_config['base_url'],
                model_name=openai_config['model'],
                temperature=0.1
            )
            
            # 初始化提示词模板
            self.prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template=Settings.SYSTEM_PROMPT_TEMPLATE
            )
            
            logger.info("RAG引擎初始化成功")
            
        except Exception as e:
            logger.error(f"RAG引擎初始化失败: {e}")
            raise
    
    def _log_evaluation_data(self, record: Dict[str, Any]):
        """
        [新增] 将 RAG Triad 评估所需的完整数据写入 JSONL 日志文件
        
        Args:
            record: 包含查询、答案、检索上下文等完整信息的字典
        """
        try:
            log_dir = os.path.dirname(Settings.EVAL_LOG_PATH)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            with open(Settings.EVAL_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"写入评估日志失败: {e}")
    
    def query(self, question: str, search_mode: str = "auto") -> Dict[str, Any]:
        """
        处理用户查询 (包含 Triad 评估日志记录)
        
        Args:
            question: 用户问题
            search_mode: 搜索模式 ("vector", "graph", "cypher", "auto")
        
        Returns:
            包含答案和相关信息的字典
        """
        # 生成唯一的 query_id，便于追踪
        query_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            logger.info(f"[{query_id}] 处理查询: {question}, 模式: {search_mode}")
            
            # 1. 分析问题
            analysis = self.analyzer.analyze_question(question)
            question_type = analysis['question_type']
            
            # 2. 确定搜索模式
            if search_mode == "auto":
                actual_mode = analysis['recommended_mode']
                logger.info(f"自动选择搜索模式: {actual_mode.value}")
            else:
                actual_mode = self.analyzer.validate_search_mode(search_mode)
            
            # 3. 获取搜索参数
            search_params = analysis['search_params']
            
            # 4. 检索上下文 (这是原始的 Z 集合)
            context_docs = self._retrieve_context(question, actual_mode.value, search_params)
            
            # 处理无结果情况
            if not context_docs:
                return {
                    "answer": "抱歉，我没有找到相关信息来回答您的问题。",
                    "sources": [],
                    "search_mode": actual_mode.value,
                    "question_type": question_type.value,
                    "analysis": analysis,
                    "context_count": 0
                }
            
            # 5. 构建上下文 (这是喂给 LLM 的最终 Prompt 上下文部分)
            context_str = self._build_context(context_docs)
            
            # 6. 生成回答 (这是 a)
            answer = self._generate_answer(question, context_str)
            
            # 7. 提取来源 (用于前端显示)
            sources = self._extract_sources(context_docs)
            
            # =======================================================
            # [新增] 构建 RAG Triad 评估数据集 (Z 集合详情)
            # =======================================================
            structured_z = []
            for rank, doc in enumerate(context_docs):
                # 区分 Vector 还是 Graph 来源
                # vector_graph_tool 中会在 metadata['type'] 中标记 'ai_graph_search', 'vector_search' 等
                source_type = doc.metadata.get('type', 'unknown')
                
                # 如果是 hybrid 搜索，我们可以根据 type 区分是 vector 还是 graph 贡献的
                retrieval_method = "graph" if "graph" in source_type else "vector"
                if "cypher" in source_type: 
                    retrieval_method = "cypher"
                
                structured_z.append({
                    "rank": rank + 1,                 # 排序位置
                    "content": doc.page_content,      # 文本内容
                    "metadata": doc.metadata,         # 完整元数据 (包含 score, id 等)
                    "source_type": source_type,       # 具体的搜索类型标记
                    "retrieval_category": retrieval_method # 宽泛分类 (vector/graph/cypher) 用于消融实验
                })

            eval_record = {
                "query_id": query_id,
                "timestamp": start_time.isoformat(),
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
                "inputs": {
                    "question": question,             # q
                    "search_mode_used": actual_mode.value
                },
                "outputs": {
                    "answer": answer,                 # a
                    "final_context_str": context_str  # 实际喂给 LLM 的拼接文本
                },
                "retrieval_context_Z": structured_z,  # 结构化的 Top-K 集合
                "analysis_details": {
                    "intent": analysis.get('question_type').value if hasattr(analysis.get('question_type'), 'value') else str(analysis.get('question_type')),
                    "keywords": analysis.get('keywords', [])
                }
            }
            
            # 写入日志
            self._log_evaluation_data(eval_record)
            # =======================================================
            
            return {
                "answer": answer,
                "sources": sources,
                "search_mode": actual_mode.value,
                "question_type": question_type.value,
                "analysis": analysis,
                "context_count": len(context_docs),
                "context": context_str[:500] + "..." if len(context_str) > 500 else context_str
            }
            
        except Exception as e:
            logger.error(f"查询处理失败: {e}")
            return {
                "answer": f"处理查询时发生错误: {str(e)}",
                "sources": [],
                "search_mode": search_mode,
                "error": str(e)
            }
    
    def _retrieve_context(self, question: str, search_mode: str, search_params: Dict[str, any]) -> List[Document]:
        """根据模式检索上下文"""
        try:
            k = search_params.get('default_k', 5)
            
            if search_mode == "vector":
                return self._vector_search(question, k)
            elif search_mode == "graph":
                return self._graph_search(question, k)
            elif search_mode == "cypher":
                return self._cypher_search(question, k)
            else:
                logger.warning(f"未知搜索模式: {search_mode}, 使用默认graph模式")
                return self._graph_search(question, k)
                
        except Exception as e:
            logger.error(f"检索上下文失败: {e}")
            return []
    
    def _vector_search(self, question: str, k: int = 5) -> List[Document]:
        """向量搜索"""
        try:
            # 使用vector_tool进行搜索
            results = vector_tool.search(question, k=k)
            logger.info(f"向量搜索返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    def _graph_search(self, question: str, k: int = 5) -> List[Document]:
        """图谱搜索"""
        try:
            # 使用vector_graph_tool进行智能混合搜索
            results = vector_graph_tool.hybrid_search(question, k=k, search_type="auto")
            logger.info(f"图谱搜索返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"图谱搜索失败: {e}")
            return []
    
    def _cypher_search(self, question: str, k: int = 5) -> List[Document]:
        """Cypher查询搜索"""
        try:
            # 分析问题，判断是否为统计查询
            analysis = self.analyzer.analyze_question(question)
            question_type = analysis['question_type']
            keywords = analysis['keywords']
            
            if question_type == QuestionType.STATISTICS:
                # 使用模板处理统计查询
                results = self._handle_statistics_query(question, keywords, k)
            else:
                # 其他类型使用自然语言转Cypher（有风险，可能失败）
                logger.info("非统计查询，尝试自然语言转Cypher")
                try:
                    answer = graph_cypher_tool.natural_language_query(question)
                    results = [Document(
                        page_content=answer,
                        metadata={'type': 'cypher_result', 'source': 'natural_language_cypher'}
                    )]
                except Exception as cypher_error:
                    logger.warning(f"自然语言转Cypher失败: {cypher_error}，降级到图谱搜索")
                    # 降级到图谱搜索
                    results = self._graph_search(question, k)
            
            logger.info(f"Cypher搜索返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"Cypher搜索失败: {e}")
            return []
    
    def _handle_statistics_query(self, question: str, keywords: List[str], k: int) -> List[Document]:
        """处理统计查询"""
        try:
            # 检测是否为车型统计查询
            if any(keyword in question for keyword in ['电车', '电动车', '新能源车', '车型', '车']):
                # 使用模板查询车型数量
                template = self.config.get_cypher_template(QuestionType.STATISTICS, 'car_count')
                if template:
                    keyword = keywords[0] if keywords else '电'
                    result = graph_cypher_tool.execute_cypher_query(
                        template, 
                        {"keyword": keyword}
                    )
                else:
                    # 降级查询
                    template = self.config.get_cypher_template(QuestionType.STATISTICS, 'total_cars')
                    result = graph_cypher_tool.execute_cypher_query(template)
                
                if result:
                    record = result[0]
                    count = record.get('total_count', 0)
                    models = record.get('sample_models', [])
                    
                    content = f"""数据库统计结果：
总计车型数量：{count} 款

部分车型列表：
{chr(10).join([f"{i+1}. {model}" for i, model in enumerate(models[:10])])}

这些数据来自知识图谱的精确统计查询。"""
                    
                    return [Document(
                        page_content=content,
                        metadata={
                            'type': 'statistics_result',
                            'count': count,
                            'source': 'template_cypher'
                        }
                    )]
            
            # 其他统计查询的处理...
            return []
            
        except Exception as e:
            logger.error(f"统计查询处理失败: {e}")
            return []
    
    
    
    def _get_car_recommendations(self, question: str) -> List[Document]:
        """获取车型推荐"""
        try:
            # 从问题中提取预算和偏好信息（简化实现）
            criteria = self._extract_criteria_from_question(question)
            results = graph_cypher_tool.get_car_models_by_criteria(criteria)
            return results
            
        except Exception as e:
            logger.error(f"获取车型推荐失败: {e}")
            return []
    
    def _get_feature_rankings(self, question: str) -> List[Document]:
        """获取特征排名"""
        try:
            # 从问题中提取特征名称
            features = ['外观设计', '内饰质感', '智能配置', '空间实用', '舒适体验', '操控性能', '续航能耗', '价值认知']
            
            for feature in features:
                if feature in question:
                    return graph_cypher_tool.get_feature_rankings(feature)
            
            # 如果没有明确特征，返回综合排名
            return graph_cypher_tool.get_car_models_by_criteria({'min_reviews': 10})
            
        except Exception as e:
            logger.error(f"获取特征排名失败: {e}")
            return []
    
    def _get_user_analysis(self, question: str) -> List[Document]:
        """获取用户分析"""
        try:
            # 如果问题中包含具体车型，分析该车型的用户
            car_models = self._extract_car_models_from_question(question)
            
            results = []
            for car_model in car_models[:3]:  # 最多分析3个车型
                car_results = graph_cypher_tool.get_user_preferences_by_car(car_model)
                results.extend(car_results)
            
            if not results:
                # 通用用户分析
                answer = graph_cypher_tool.natural_language_query(question)
                results = [Document(
                    page_content=answer,
                    metadata={'type': 'user_analysis', 'source': 'general_analysis'}
                )]
            
            return results
            
        except Exception as e:
            logger.error(f"获取用户分析失败: {e}")
            return []
    
    def _get_comparison_analysis(self, question: str) -> List[Document]:
        """获取对比分析"""
        try:
            # 提取要对比的车型
            car_models = self._extract_car_models_from_question(question)
            
            if len(car_models) >= 2:
                results = []
                for car_model in car_models[:3]:  # 最多对比3个车型
                    car_info = vector_graph_tool.get_car_comprehensive_info(car_model)
                    if car_info:
                        results.append(car_info)
                return results
            else:
                # 如果没有明确车型，使用自然语言查询
                answer = graph_cypher_tool.natural_language_query(question)
                return [Document(
                    page_content=answer,
                    metadata={'type': 'comparison', 'source': 'natural_language'}
                )]
                
        except Exception as e:
            logger.error(f"获取对比分析失败: {e}")
            return []
    
    def _extract_criteria_from_question(self, question: str) -> Dict[str, Any]:
        """从问题中提取筛选条件"""
        criteria = {}
        
        # 提取品牌信息
        brands = ['小米', '智界', '享界', '极氪', '宝马', '奔驰', '特斯拉', '比亚迪', '蔚来', '理想']
        for brand in brands:
            if brand in question:
                criteria['brand'] = brand
                break
        
        # 提取价格区间
        if '50万以上' in question or '高端' in question:
            criteria['price_range'] = '50万以上'
        elif '30-50万' in question or '中高端' in question:
            criteria['price_range'] = '30-50万'
        elif '20-30万' in question or '中端' in question:
            criteria['price_range'] = '20-30万'
        elif '20万以下' in question or '经济' in question:
            criteria['price_range'] = '20万以下'
        
        # 设置最小评论数要求
        criteria['min_reviews'] = 5
        
        return criteria
    
    def _extract_car_models_from_question(self, question: str) -> List[str]:
        """从问题中提取车型名称"""
        # 简化的车型提取（实际应用中可以使用更复杂的NER）
        car_models = [
            '小米SU7 Ultra', '智界S7', '享界S9', '极氪001', '极氪007',
            '宝马i5 M60', '宝马i7 M70', '奔驰EQE53 AMG', '奔驰EQS580',
            '特斯拉Model 3', '特斯拉Model Y', '比亚迪汉EV', '蔚来ES6', '理想L9'
        ]
        
        found_models = []
        for model in car_models:
            if model in question:
                found_models.append(model)
        
        return found_models
    
    def _deduplicate_results(self, results: List[Document]) -> List[Document]:
        """去重结果"""
        seen_content = set()
        unique_results = []
        
        for doc in results:
            # 使用内容的前100字符作为去重标识
            content_key = doc.page_content[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(doc)
        
        return unique_results
    
    def _build_context(self, docs: List[Document]) -> str:
        """构建上下文字符串"""
        context_parts = []
        
        for i, doc in enumerate(docs[:8]):  # 最多使用8个文档
            source_info = ""
            if doc.metadata.get('type'):
                source_info = f"[{doc.metadata['type']}] "
            if doc.metadata.get('car_model'):
                source_info += f"({doc.metadata['car_model']}) "
                
            context_parts.append(f"{source_info}{doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """生成回答"""
        try:
            prompt = self.prompt_template.format(context=context, question=question)
            response = self.llm.invoke(prompt)
            
            # 提取回答内容
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            logger.info("成功生成回答")
            return answer
            
        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return f"生成回答时发生错误: {str(e)}"
    
    def _extract_sources(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """提取来源信息"""
        sources = []
        
        for doc in docs:
            source = {
                'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                'metadata': doc.metadata
            }
            sources.append(source)
        
        return sources
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            status = {
                'rag_engine': 'healthy',
                'vector_tool': 'healthy' if vector_tool else 'error',
                'vector_graph_tool': 'healthy' if vector_graph_tool else 'error',
                'graph_cypher_tool': 'healthy' if graph_cypher_tool else 'error',
                'llm': 'healthy' if self.llm else 'error'
            }
            
            # 获取向量存储统计
            vector_stats = vector_tool.get_collection_stats()
            status['vector_store'] = vector_stats
            
            # 获取数据库统计
            db_stats = graph_cypher_tool.get_database_statistics()
            status['database'] = db_stats.metadata if hasattr(db_stats, 'metadata') else {}
            
            return status
            
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {'status': 'error', 'error': str(e)}

# 全局实例
rag_engine = RAGEngine()