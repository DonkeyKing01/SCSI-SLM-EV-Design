"""
向量图谱工具模块
结合Neo4j图谱结构和向量检索的混合搜索工具
使用大模型和提示词进行智能查询生成
"""
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from database.neo4j_connection import neo4j_conn
from config.settings import Settings
import logging
import json

logger = logging.getLogger(__name__)

class VectorGraphTool:
    """向量图谱工具类"""
    
    def __init__(self):
        """初始化向量图谱工具"""
        self.neo4j_vector = None
        self.embeddings = None
        self.llm = None
        self.prompts = {}
        self._initialize()
    
    def _initialize(self):
        """初始化组件"""
        try:
            # 初始化嵌入模型和大语言模型
            openai_config = Settings.get_openai_config()
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=openai_config['api_key'],
                openai_api_base=openai_config['base_url'],
                model=Settings.EMBEDDING_MODEL
            )
            
            self.llm = ChatOpenAI(
                openai_api_key=openai_config['api_key'],
                openai_api_base=openai_config['base_url'],
                model=Settings.OPENAI_MODEL,
                temperature=0.1
            )
            
            # 初始化提示词模板
            self._setup_prompts()
            
            # 初始化Neo4j向量索引
            self._setup_neo4j_vector()
            
            logger.info("VectorGraphTool初始化成功")
            
        except Exception as e:
            logger.error(f"VectorGraphTool初始化失败: {e}")
            raise
    
    def _setup_prompts(self):
        """设置提示词模板"""
        # 查询意图分析提示词
        self.prompts['query_analysis'] = ChatPromptTemplate.from_messages([
            ("system", """你是一个汽车评论数据分析专家。请分析用户查询的意图，只返回JSON，不要其他文字。

分析查询意图，识别关键词和实体，返回标准JSON格式。
意图类型：car_recommendation, user_analysis, feature_comparison, car_info, general_search

只返回以下JSON格式，不要其他内容：
{"intent": "意图类型", "keywords": ["关键词"], "entities": {"brands": [], "car_models": [], "features": []}, "search_focus": "focus类型"}"""),
            ("user", "查询: {query}")
        ])
        
        # Cypher查询生成提示词
        self.prompts['cypher_generation'] = ChatPromptTemplate.from_messages([
            ("system", """你是Neo4j Cypher专家。生成简单可靠的查询。

基本模式：MATCH (r:Review)-[m:MENTIONS]->(c:CarModel)

常用参数：$keyword_0, $limit
过滤条件：size(r.content) > 30
排序：r.importance DESC, m.sentimentScore DESC

只返回一个完整的Cypher查询语句，确保语法正确。不要解释。"""),
            ("user", "关键词: {keywords}, 限制: {limit}条记录。生成查询：")
        ])
    
    def _setup_neo4j_vector(self):
        """设置Neo4j向量索引（社区版兼容）"""
        try:
            # Neo4j社区版不支持向量索引，我们直接设置为None
            # 使用基于内容的相似度搜索替代
            self.neo4j_vector = None
            logger.info("Neo4j社区版模式：使用基于内容的相似度搜索")
            
        except Exception as e:
            logger.warning(f"Neo4j向量索引设置失败: {e}")
            self.neo4j_vector = None
    
    def create_vector_index(self):
        """创建向量索引（如果不存在）"""
        try:
            # Neo4j社区版不支持向量索引，我们只存储嵌入向量作为属性
            # 在企业版中可以创建向量索引以提高性能
            logger.info("向量索引初始化完成（社区版仅存储向量属性）")
            
        except Exception as e:
            logger.error(f"创建向量索引失败: {e}")
            raise
    
    def hybrid_search(self, query: str, k: int = 5, 
                     search_type: str = "auto") -> List[Document]:
        """智能混合搜索：根据查询类型自动选择最佳搜索策略"""
        try:
            # 如果指定了搜索类型，直接使用
            if search_type == "vector_only":
                return self._vector_search(query, k)
            elif search_type == "graph_only":
                return self._graph_search(query, k)
            elif search_type == "hybrid":
                return self._hybrid_search(query, k)
            elif search_type != "auto":
                logger.warning(f"未知搜索类型: {search_type}，使用自动选择")
            
            # 自动选择搜索策略
            return self._intelligent_search(query, k)
                
        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            return []
    
    def _intelligent_search(self, query: str, k: int) -> List[Document]:
        """智能搜索：基于查询分析自动选择最佳策略"""
        try:
            # 分析查询意图
            analysis = self._analyze_query_with_llm(query)
            intent = analysis.get('intent', 'general_search')
            entities = analysis.get('entities', {})
            
            logger.info(f"智能搜索策略选择 - 意图: {intent}, 实体: {entities}")
            
            # 根据意图选择搜索策略
            if intent == "car_recommendation":
                # 推荐查询：结合图谱关系和情感分析
                return self._recommendation_search(query, analysis, k)
            
            elif intent == "feature_comparison":
                # 对比查询：重点关注具体特性和对比信息
                return self._comparison_search(query, analysis, k)
            
            elif intent == "car_info":
                # 信息查询：优先图谱结构化信息，辅以详细评论
                return self._info_search(query, analysis, k)
            
            elif intent == "brand_analysis":
                # 品牌分析：聚合品牌相关信息
                return self._brand_search(query, analysis, k)
            
            else:
                # 一般搜索：使用混合策略
                return self._general_search(query, analysis, k)
                
        except Exception as e:
            logger.error(f"智能搜索失败: {e}")
            # 降级到基础混合搜索
            return self._hybrid_search(query, k)
    
    def _recommendation_search(self, query: str, analysis: Dict, k: int) -> List[Document]:
        """推荐导向的搜索"""
        try:
            # 优先使用AI驱动的内容搜索，因为它能考虑情感分析
            ai_results = self._content_similarity_search(query, k)
            
            # 如果AI搜索结果不足，补充基础搜索
            if len(ai_results) < k:
                remaining = k - len(ai_results)
                fallback_results = self._fallback_simple_search(query, remaining)
                ai_results.extend(fallback_results)
            
            return ai_results[:k]
            
        except Exception as e:
            logger.error(f"推荐搜索失败: {e}")
            return self._hybrid_search(query, k)
    
    def _comparison_search(self, query: str, analysis: Dict, k: int) -> List[Document]:
        """对比导向的搜索"""
        try:
            # 对比查询优先使用结构化信息
            graph_results = self._graph_search(query, k//2)
            vector_results = self._vector_search(query, k//2)
            
            # 合并结果
            combined = graph_results + vector_results
            return self._deduplicate_results(combined)[:k]
            
        except Exception as e:
            logger.error(f"对比搜索失败: {e}")
            return self._hybrid_search(query, k)
    
    def _info_search(self, query: str, analysis: Dict, k: int) -> List[Document]:
        """信息导向的搜索"""
        try:
            # 信息查询优先使用图谱搜索
            graph_results = self._graph_search(query, int(k * 0.7))
            vector_results = self._vector_search(query, int(k * 0.3))
            
            combined = graph_results + vector_results
            return self._deduplicate_results(combined)[:k]
            
        except Exception as e:
            logger.error(f"信息搜索失败: {e}")
            return self._hybrid_search(query, k)
    
    def _brand_search(self, query: str, analysis: Dict, k: int) -> List[Document]:
        """品牌导向的搜索"""
        try:
            # 品牌查询使用AI驱动搜索，能更好理解品牌相关信息
            return self._content_similarity_search(query, k)
            
        except Exception as e:
            logger.error(f"品牌搜索失败: {e}")
            return self._hybrid_search(query, k)
    
    def _general_search(self, query: str, analysis: Dict, k: int) -> List[Document]:
        """一般搜索策略"""
        try:
            # 一般查询使用平衡的混合搜索
            return self._hybrid_search(query, k)
            
        except Exception as e:
            logger.error(f"一般搜索失败: {e}")
            return self._vector_search(query, k)
    
    def _deduplicate_results(self, results: List[Document]) -> List[Document]:
        """去重搜索结果"""
        seen_content = set()
        unique_results = []
        
        for doc in results:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(doc)
        
        return unique_results
    
    def _vector_search(self, query: str, k: int) -> List[Document]:
        """纯向量搜索（社区版兼容）"""
        try:
            if not self.neo4j_vector:
                # 社区版使用基于内容的相似度搜索
                return self._content_similarity_search(query, k)
            
            results = self.neo4j_vector.similarity_search(query, k=k)
            logger.info(f"向量搜索返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    def _content_similarity_search(self, query: str, k: int) -> List[Document]:
        """基于内容相似度的搜索（使用大模型生成查询）"""
        try:
            # 分析查询意图
            query_analysis = self._analyze_query_with_llm(query)
            
            # 生成Cypher查询
            cypher_query, params = self._generate_cypher_query(query_analysis, k)
            
            # 执行查询
            graph = neo4j_conn.get_graph()
            result = graph.query(cypher_query, params=params)
            
            # 构建文档
            documents = []
            for record in result:
                content = self._format_search_result(record, query_analysis['intent'])
                
                doc = Document(
                    page_content=content,
                    metadata={
                        'type': 'llm_generated_search',
                        'intent': query_analysis['intent'],
                        'car_model': record.get('carModel', ''),
                        'sentiment': record.get('sentiment', ''),
                        'source': 'ai_powered_search'
                    }
                )
                documents.append(doc)
            
            logger.info(f"AI驱动的内容搜索返回 {len(documents)} 个结果")
            return documents[:k]
            
        except Exception as e:
            logger.error(f"AI驱动的内容搜索失败: {e}")
            # 降级到简单搜索
            return self._fallback_simple_search(query, k)
    
    def _analyze_query_with_llm(self, query: str) -> Dict[str, Any]:
        """使用大模型分析查询意图"""
        try:
            # 构建分析提示词
            analysis_prompt = f"""
你是一个汽车领域的智能分析助手。请分析以下用户查询，提取关键信息：

用户查询："{query}"

请按以下格式返回JSON分析结果：
{{
    "intent": "查询意图（car_recommendation/feature_comparison/car_info/general_search/brand_analysis）",
    "core_keywords": ["核心关键词1", "核心关键词2"],
    "search_terms": ["用于搜索的词汇1", "用于搜索的词汇2"],
    "entities": {{
        "brands": ["提及的品牌"],
        "car_models": ["提及的车型"],
        "features": ["提及的功能特性"]
    }},
    "search_focus": "content"
}}

分析说明：
1. intent分类：
   - car_recommendation: 推荐、选择、买车、哪个好等
   - feature_comparison: 对比、比较、区别等
   - car_info: 信息、详情、介绍、参数等
   - brand_analysis: 品牌分析、市场情况等
   - general_search: 一般性搜索

2. core_keywords: 提取最核心的1-3个关键词，去除助词、量词等
   例如："有几种电车" -> ["电车"]
   例如："特斯拉Model Y怎么样" -> ["特斯拉", "Model Y"]

3. search_terms: 用于数据库搜索的词汇，可以包含同义词
   例如："电车" -> ["电车", "电动车", "新能源车"]

4. 品牌识别：小米、智界、享界、极氪、宝马、奔驰、特斯拉、比亚迪、蔚来、理想、问界等

只返回JSON，不要其他文字。
"""

            # 调用LLM
            response = self.llm.invoke(analysis_prompt)
            analysis_text = response.content.strip()
            
            # 清理响应文本
            if analysis_text.startswith('```json'):
                analysis_text = analysis_text[7:]
            if analysis_text.endswith('```'):
                analysis_text = analysis_text[:-3]
            analysis_text = analysis_text.strip()
            
            # 解析JSON
            import json
            try:
                analysis = json.loads(analysis_text)
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {e}, 响应内容: {analysis_text[:200]}...")
                # 降级到简单分析
                return self._simple_query_analysis(query)
            
            # 验证和补充必要字段
            if "intent" not in analysis:
                analysis["intent"] = "general_search"
            if "core_keywords" not in analysis:
                analysis["core_keywords"] = [query]
            if "search_terms" not in analysis:
                analysis["search_terms"] = analysis.get("core_keywords", [query])
            if "entities" not in analysis:
                analysis["entities"] = {"brands": [], "car_models": [], "features": []}
            if "search_focus" not in analysis:
                analysis["search_focus"] = "content"
            
            # 为了兼容现有代码，保留keywords字段
            analysis["keywords"] = analysis["search_terms"]
            
            logger.info(f"LLM查询意图分析: {analysis}")
            return analysis
            
        except Exception as e:
            logger.error(f"LLM查询意图分析失败: {e}")
            # 降级到简单分析
            return self._simple_query_analysis(query)
    
    def _simple_query_analysis(self, query: str) -> Dict[str, Any]:
        """简单的查询分析（降级方案）"""
        try:
            import jieba
            
            # 使用jieba分词提取关键词
            words = list(jieba.cut(query))
            
            # 过滤停用词和无意义词
            stop_words = {'有', '几', '种', '个', '什么', '怎么', '样', '的', '了', '吗', '呢', '啊', '吧'}
            keywords = [word for word in words if len(word) > 1 and word not in stop_words]
            
            # 如果没有关键词，使用原始查询
            if not keywords:
                keywords = [query]
            
            # 意图分析
            query_lower = query.lower()
            intent = "general_search"
            if any(word in query_lower for word in ['推荐', '选择', '买', '哪个']):
                intent = "car_recommendation"
            elif any(word in query_lower for word in ['对比', '比较', '区别']):
                intent = "feature_comparison"
            elif any(word in query_lower for word in ['信息', '详情', '介绍']):
                intent = "car_info"
            
            # 品牌识别
            brands = []
            brand_list = ['小米', '智界', '享界', '极氪', '宝马', '奔驰', '特斯拉', '比亚迪', '蔚来', '理想', '问界']
            for brand in brand_list:
                if brand in query:
                    brands.append(brand)
            
            analysis = {
                "intent": intent,
                "core_keywords": keywords[:3],
                "search_terms": keywords[:3],
                "keywords": keywords[:3],  # 兼容字段
                "entities": {
                    "brands": brands,
                    "car_models": [],
                    "features": []
                },
                "search_focus": "content"
            }
            
            logger.info(f"简单查询意图分析: {analysis}")
            return analysis
            
        except Exception as e:
            logger.error(f"简单查询分析失败: {e}")
            # 最终降级
            return {
                "intent": "general_search",
                "core_keywords": [query],
                "search_terms": [query],
                "keywords": [query],
                "entities": {"brands": [], "car_models": [], "features": []},
                "search_focus": "content"
            }
    
    def _generate_cypher_query(self, analysis: Dict[str, Any], k: int) -> tuple:
        """生成Cypher查询（暂时使用模板，避免LLM生成问题）"""
        try:
            # 暂时使用简单可靠的查询模板
            # TODO: 后续优化LLM生成
            keywords = analysis['keywords']
            
            if not keywords:
                keywords = ['汽车']  # 默认关键词
            
            # 根据意图选择查询模板
            if analysis['intent'] == 'car_recommendation':
                cypher_query = """
                MATCH (r:Review)-[m:MENTIONS]->(c:CarModel)
                WHERE r.content CONTAINS $keyword_0 
                AND r.overallSentiment = 'positive'
                AND size(r.content) > 30
                RETURN r.content as content, c.name as carModel, c.brand as brand,
                       r.overallSentiment as sentiment, r.importance as importance,
                       m.sentimentScore as carSentiment
                ORDER BY m.sentimentScore DESC, r.importance DESC
                LIMIT $limit
                """
            else:
                cypher_query = """
                MATCH (r:Review)-[m:MENTIONS]->(c:CarModel)
                WHERE r.content CONTAINS $keyword_0
                AND size(r.content) > 30
                RETURN r.content as content, c.name as carModel, c.brand as brand,
                       r.overallSentiment as sentiment, r.importance as importance,
                       m.sentimentScore as carSentiment
                ORDER BY r.importance DESC, m.sentimentScore DESC
                LIMIT $limit
                """
            
            # 构建查询参数
            params = {
                "keyword_0": keywords[0],
                "limit": k * 2
            }
            
            logger.info(f"使用模板查询，关键词: {keywords[0]}")
            return cypher_query, params
            
        except Exception as e:
            logger.error(f"Cypher查询生成失败: {e}")
            # 返回简单的降级查询
            return self._get_fallback_query(analysis, k)
    
    def _get_fallback_query(self, analysis: Dict[str, Any], k: int) -> tuple:
        """获取降级查询"""
        cypher_query = """
        MATCH (r:Review)-[m:MENTIONS]->(c:CarModel)
        WHERE r.content CONTAINS $keyword_0
        AND size(r.content) > 30
        RETURN r.content as content, c.name as carModel, c.brand as brand,
               r.overallSentiment as sentiment, r.importance as importance,
               m.sentimentScore as carSentiment
        ORDER BY r.importance DESC, m.sentimentScore DESC
        LIMIT $limit
        """
        
        params = {
            "keyword_0": analysis['keywords'][0] if analysis['keywords'] else "汽车",
            "limit": k * 2
        }
        
        return cypher_query, params
    
    def _format_search_result(self, record: Dict[str, Any], intent: str) -> str:
        """格式化搜索结果"""
        if intent == "car_recommendation":
            return f"""车型推荐信息:
车型: {record.get('carModel', 'N/A')}
品牌: {record.get('brand', 'N/A')}
评论: {record.get('content', 'N/A')[:200]}...
情感评价: {record.get('sentiment', 'N/A')}
重要性: {record.get('importance', 0):.3f}
车型满意度: {record.get('carSentiment', 0):.3f}

基于AI分析的智能推荐，综合考虑用户需求和真实评论数据。"""
        
        elif intent == "feature_comparison":
            return f"""特征对比信息:
车型: {record.get('carModel', 'N/A')} ({record.get('brand', 'N/A')})
相关评论: {record.get('content', 'N/A')[:200]}...
特征表现: {record.get('featureScore', 'N/A')}
用户满意度: {record.get('carSentiment', 0):.3f}

基于用户真实反馈的特征对比分析。"""
        
        else:
            return f"""相关评论信息:
车型: {record.get('carModel', 'N/A')} ({record.get('brand', 'N/A')})
评论内容: {record.get('content', 'N/A')[:200]}...
整体情感: {record.get('sentiment', 'N/A')}
重要性评分: {record.get('importance', 0):.3f}
车型满意度: {record.get('carSentiment', 0):.3f}

来自真实用户的评论，反映实际使用体验。"""
    
    def _fallback_simple_search(self, query: str, k: int) -> List[Document]:
        """简单的降级搜索"""
        try:
            graph = neo4j_conn.get_graph()
            
            # 简单的关键词匹配查询
            cypher_query = """
            MATCH (r:Review)-[m:MENTIONS]->(c:CarModel)
            WHERE r.content CONTAINS $query_term
            AND size(r.content) > 30
            RETURN r.content as content, c.name as carModel, c.brand as brand,
                   r.overallSentiment as sentiment, r.importance as importance,
                   m.sentimentScore as carSentiment
            ORDER BY r.importance DESC
            LIMIT $limit
            """
            
            result = graph.query(cypher_query, params={
                "query_term": query.split()[0] if query.split() else query,
                "limit": k
            })
            
            documents = []
            for record in result:
                content = f"""评论信息:
车型: {record['carModel']} ({record['brand']})
评论: {record['content'][:200]}...
情感: {record['sentiment']}
重要性: {record['importance']:.3f}

降级搜索结果，基于简单关键词匹配。"""

                doc = Document(
                    page_content=content,
                    metadata={
                        'type': 'fallback_search',
                        'car_model': record['carModel'],
                        'sentiment': record['sentiment'],
                        'source': 'simple_search'
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"降级搜索也失败: {e}")
            return []
    
    def _graph_search(self, query: str, k: int) -> List[Document]:
        """基于图谱关系的搜索（使用AI生成查询）"""
        try:
            # 使用AI分析查询意图
            query_analysis = self._analyze_query_with_llm(query)
            
            # 根据意图使用AI生成专门的图谱查询
            cypher_query, params = self._generate_graph_cypher_query(query_analysis, k)
            
            # 执行查询
            graph = neo4j_conn.get_graph()
            result = graph.query(cypher_query, params=params)
            
            # 构建文档
            documents = []
            for record in result:
                content = self._format_graph_search_result(record, query_analysis['intent'])
                
                doc = Document(
                    page_content=content,
                    metadata={
                        'type': 'ai_graph_search',
                        'intent': query_analysis['intent'],
                        'source': 'graph_ai_search'
                    }
                )
                documents.append(doc)
            
            logger.info(f"AI图谱搜索返回 {len(documents)} 个结果")
            return documents[:k]
                
        except Exception as e:
            logger.error(f"AI图谱搜索失败: {e}")
            # 降级到简单图谱搜索
            return self._fallback_graph_search(query, k)
    
    def _generate_graph_cypher_query(self, analysis: Dict[str, Any], k: int) -> tuple:
        """为图谱搜索生成专门的Cypher查询"""
        # 创建图谱专用的提示词
        graph_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个Neo4j Cypher查询专家，必须生成语法正确的完整查询。

数据库架构信息：
节点类型：
- Review（评论）：包含content, overallSentiment, importance, userId, reviewId
- CarModel（车型）：包含name, brand, type, priceRange, reviewCount, modelId
- UserProfile（用户画像）：用户相关信息
- Feature（特征）：包含name, description, category

关系类型：
- MENTIONS（评论提及车型）：Review->CarModel，包含sentimentScore属性
- PUBLISHED（用户发布评论）：UserProfile->Review，包含userMatchScore属性
- CONTAINS_ASPECT（评论包含特征）：Review->Feature，包含intensity和sentiment属性
- INTERESTED_IN（用户对特征感兴趣）：UserProfile->Feature

查询意图对应的查询模式：
- car_recommendation: 查找正面评价的车型推荐
- feature_comparison: 比较不同车型在特定特征上的表现
- car_info: 获取特定车型的详细信息和评价
- brand_analysis: 分析品牌相关数据
- general_search: 基于关键词的通用搜索

【🚫 绝对禁止的语法错误】：
❌ 查询不能以 WITH 子句结尾
❌ 不能有未完成的 WITH 语句
❌ 查询必须完整，不能截断

【✅ 正确的查询结构】：
MATCH (节点模式)-[关系]->(节点模式)
WHERE 过滤条件
[可选: WITH 变量, 计算字段]
RETURN 字段列表
ORDER BY 排序字段
LIMIT 数量

【示例正确查询】：
MATCH (r:Review)-[:MENTIONS]->(c:CarModel)
WHERE r.content CONTAINS $keyword_0
RETURN r.content as content, c.name as carModel, c.brand as brand
ORDER BY r.importance DESC
LIMIT $limit

【严格要求】：
1. 查询必须以RETURN子句结尾
2. 如果使用WITH，必须在其后添加RETURN子句
3. 不要生成不完整的查询
4. 只返回Cypher查询代码，不要解释文本
5. 确保查询语法完整且可执行"""),
            ("user", """
意图: {intent}
关键词: {keywords}  
实体: {entities}
返回数量: {limit}

请生成完整的Cypher查询（必须以RETURN结尾）：""")
        ])
        
        try:
            response = self.llm.invoke(graph_prompt.format_messages(
                intent=analysis['intent'],
                keywords=analysis['keywords'],
                entities=analysis['entities'],
                limit=k * 2
            ))
            
            cypher_query = response.content.strip()
            
            # 清理可能的markdown标记
            if cypher_query.startswith('```cypher'):
                cypher_query = cypher_query[9:]
            elif cypher_query.startswith('```'):
                cypher_query = cypher_query[3:]
            
            if cypher_query.endswith('```'):
                cypher_query = cypher_query[:-3]
            
            cypher_query = cypher_query.strip()
            
            # 验证和修复查询语法
            cypher_query = self._validate_and_fix_cypher(cypher_query)
            
            # 构建参数
            params = {"limit": k * 2}
            for i, keyword in enumerate(analysis['keywords'][:3]):
                params[f"keyword_{i}"] = keyword
            
            if analysis['entities']['brands']:
                params['brand'] = analysis['entities']['brands'][0]
            if analysis['entities']['car_models']:
                params['car_model'] = analysis['entities']['car_models'][0]
                
            return cypher_query, params
            
        except Exception as e:
            logger.error(f"图谱查询生成失败: {e}")
            return self._get_fallback_graph_query(analysis, k)
    
    def _validate_and_fix_cypher(self, cypher_query: str) -> str:
        """验证并修复Cypher查询语法"""
        try:
            # 去除多余空行和空格
            lines = [line.strip() for line in cypher_query.split('\n') if line.strip()]
            cypher_query = '\n'.join(lines)
            
            # 打印查询用于调试
            logger.info(f"原始查询: {cypher_query}")
            
            # 检查查询是否以WITH结尾（这是最常见的错误）
            query_lines = [line.strip() for line in cypher_query.split('\n') if line.strip()]
            if not query_lines:
                logger.error("查询为空")
                return self._get_safe_fallback_query()
            
            # 检查最后一行是否以WITH开头
            last_line = query_lines[-1].strip().upper()
            logger.info(f"最后一行: {last_line}")
            
            # 如果最后一行以WITH开头，说明查询不完整
            if last_line.startswith('WITH'):
                logger.warning("❌ 检测到查询以WITH结尾，这是无效语法！自动修复...")
                
                # 分析WITH语句中提到的变量
                with_vars = self._extract_with_variables(last_line)
                logger.info(f"FROM WITH语句提取的变量: {with_vars}")
                
                # 构建合适的RETURN语句
                return_clause = self._build_return_from_with_vars(with_vars)
                
                # 移除最后的WITH行，添加RETURN
                fixed_query = '\n'.join(query_lines[:-1]) + '\n' + return_clause
                logger.info(f"✅ 修复后查询: {fixed_query}")
                return fixed_query
            
            # 检查是否有RETURN子句
            has_return = any('RETURN' in line.upper() for line in query_lines)
            
            if not has_return:
                logger.warning("检测到缺少RETURN子句，自动添加")
                cypher_query += """
RETURN r.content as content, c.name as carModel, c.brand as brand,
       r.overallSentiment as sentiment, r.importance as importance
ORDER BY r.importance DESC
LIMIT $limit"""
            
            # 确保有LIMIT
            has_limit = any('LIMIT' in line.upper() for line in query_lines)
            if not has_limit:
                cypher_query += '\nLIMIT $limit'
            
            logger.info(f"✅ 查询验证完成")
            return cypher_query
            
        except Exception as e:
            logger.error(f"查询验证失败: {e}")
            return self._get_safe_fallback_query()
    
    def _extract_with_variables(self, with_line: str) -> List[str]:
        """从WITH语句中提取变量名"""
        try:
            # 移除"WITH"关键字
            vars_part = with_line[4:].strip()  # 移除"WITH"
            
            # 分割变量（通过逗号）
            variables = []
            for var in vars_part.split(','):
                var = var.strip()
                # 处理别名情况 (如 "c as carModel")
                if ' as ' in var.lower():
                    var = var.split(' as ')[0].strip()
                variables.append(var)
            
            return variables
        except Exception as e:
            logger.error(f"提取WITH变量失败: {e}")
            return ['r', 'c']  # 默认变量
    
    def _build_return_from_with_vars(self, variables: List[str]) -> str:
        """根据WITH变量构建RETURN语句"""
        try:
            return_parts = []
            
            for var in variables:
                if var.lower() in ['r', 'review']:
                    return_parts.extend([
                        f"{var}.content as content",
                        f"{var}.overallSentiment as sentiment", 
                        f"{var}.importance as importance"
                    ])
                elif var.lower() in ['c', 'cm', 'carmodel']:
                    return_parts.extend([
                        f"{var}.name as carModel",
                        f"{var}.brand as brand"
                    ])
                elif var.lower() in ['f', 'feature']:
                    return_parts.extend([
                        f"{var}.name as featureName",
                        f"{var}.description as featureDesc"
                    ])
                else:
                    # 通用处理
                    return_parts.append(f"{var}")
            
            if not return_parts:
                return_parts = ["r.content as content", "c.name as carModel"]
            
            return_clause = "RETURN " + ", ".join(return_parts[:5])  # 限制字段数量
            return_clause += "\nORDER BY r.importance DESC\nLIMIT $limit"
            
            return return_clause
            
        except Exception as e:
            logger.error(f"构建RETURN语句失败: {e}")
            return """RETURN r.content as content, c.name as carModel
ORDER BY r.importance DESC
LIMIT $limit"""
    
    def _get_safe_fallback_query(self) -> str:
        """获取安全的备用查询"""
        return """MATCH (r:Review)-[m:MENTIONS]->(c:CarModel)
            WHERE r.content CONTAINS $keyword_0
            RETURN r.content as content, c.name as carModel, c.brand as brand,
                   r.overallSentiment as sentiment, r.importance as importance
            ORDER BY r.importance DESC
            LIMIT $limit
            """
    
    def _get_fallback_graph_query(self, analysis: Dict[str, Any], k: int) -> tuple:
        """获取降级的图谱查询"""
        cypher_query = """
        MATCH (r:Review)-[m:MENTIONS]->(c:CarModel)
        OPTIONAL MATCH (r)-[ca:CONTAINS_ASPECT]->(f:Feature)
        WHERE r.content CONTAINS $keyword_0
        RETURN r.content as content, c.name as carModel, c.brand as brand,
               r.overallSentiment as sentiment, m.sentimentScore as carSentiment,
               COLLECT(f.name) as features
        ORDER BY m.sentimentScore DESC
        LIMIT $limit
        """
        
        params = {
            "keyword_0": analysis['keywords'][0] if analysis['keywords'] else "汽车",
            "limit": k * 2
        }
        
        return cypher_query, params
    
    def _format_graph_search_result(self, record: Dict[str, Any], intent: str) -> str:
        """格式化图谱搜索结果"""
        if intent == "car_recommendation":
            return f"""智能推荐分析:
车型: {record.get('carModel', 'N/A')} ({record.get('brand', 'N/A')})
推荐理由: {record.get('content', 'N/A')[:150]}...
用户匹配度: {record.get('userMatch', 0):.3f}
综合评分: {record.get('overallScore', 0):.3f}
目标用户群: {record.get('userType', 'N/A')}

基于图谱关系分析的智能推荐。"""
        
        elif intent == "user_analysis":
            return f"""用户画像分析:
用户类型: {record.get('userType', 'N/A')}
关注特征: {', '.join(record.get('features', []))}
偏好强度: {record.get('preferenceStrength', 0):.3f}
代表性评论: {record.get('content', 'N/A')[:150]}...

基于关系图谱的用户行为分析。"""
        
        elif intent == "feature_comparison":
            return f"""特征对比分析:
特征维度: {record.get('feature', 'N/A')}
表现车型: {record.get('carModel', 'N/A')}
用户关注度: {record.get('attention', 0):.3f}
满意度评分: {record.get('satisfaction', 0):.3f}
相关评论: {record.get('content', 'N/A')[:150]}...

基于图谱关系的特征表现对比。"""
        
        else:
            return f"""图谱关系信息:
车型: {record.get('carModel', 'N/A')} ({record.get('brand', 'N/A')})
关联特征: {', '.join(record.get('features', []))}
评论摘要: {record.get('content', 'N/A')[:150]}...
关系强度: {record.get('relationStrength', 0):.3f}

基于知识图谱的关系分析结果。"""
    
    def _fallback_graph_search(self, query: str, k: int) -> List[Document]:
        """降级的图谱搜索"""
        try:
            graph = neo4j_conn.get_graph()
            
            cypher_query = """
            MATCH (r:Review)-[m:MENTIONS]->(c:CarModel)
            WHERE r.content CONTAINS $query_term
            RETURN r.content as content, c.name as carModel, c.brand as brand,
                   r.overallSentiment as sentiment, m.sentimentScore as carSentiment
            ORDER BY m.sentimentScore DESC
            LIMIT $limit
            """
            
            result = graph.query(cypher_query, params={
                "query_term": query.split()[0] if query.split() else query,
                "limit": k
            })
            
            documents = []
            for record in result:
                content = f"""图谱搜索结果:
车型: {record['carModel']} ({record['brand']})
评论: {record['content'][:200]}...
情感: {record['sentiment']}
车型满意度: {record['carSentiment']:.3f}

降级图谱搜索结果。"""

                doc = Document(
                    page_content=content,
                    metadata={
                        'type': 'fallback_graph_search',
                        'car_model': record['carModel'],
                        'sentiment': record['sentiment'],
                        'source': 'simple_graph_search'
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"降级图谱搜索失败: {e}")
            return []
    
    def _hybrid_search(self, query: str, k: int) -> List[Document]:
        """混合搜索：结合向量和图谱"""
        try:
            # 向量搜索结果
            vector_results = self._vector_search(query, k//2)
            
            # 图谱搜索结果
            graph_results = self._graph_search(query, k//2)
            
            # 合并和去重
            combined_results = vector_results + graph_results
            
            # 简单去重（基于内容）
            seen_content = set()
            unique_results = []
            for doc in combined_results:
                content_hash = hash(doc.page_content[:100])  # 使用前100字符作为去重依据
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_results.append(doc)
            
            logger.info(f"混合搜索返回 {len(unique_results)} 个去重结果")
            return unique_results[:k]
            
        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            return []
    
    
    def get_car_comprehensive_info(self, car_model: str) -> Optional[Document]:
        """获取特定车型的综合信息"""
        try:
            graph = neo4j_conn.get_graph()
            
            cypher_query = """
            MATCH (c:CarModel {name: $car_model})
            OPTIONAL MATCH (r:Review)-[m:MENTIONS]->(c)
            OPTIONAL MATCH (r)-[ca:CONTAINS_ASPECT]->(f:Feature)
            OPTIONAL MATCH (u:UserProfile)-[pub:PUBLISHED]->(r)
            
            WITH c, 
                 COUNT(DISTINCT r) as reviewCount,
                 AVG(m.sentimentScore) as avgSentiment,
                 COLLECT(DISTINCT {feature: f.name, intensity: ca.intensity, sentiment: ca.sentiment}) as features,
                 COLLECT(DISTINCT {userType: u.name, userCount: u.userCount}) as userTypes
            
            RETURN c.name as carModel, c.brand as brand, c.type as type, c.priceRange as priceRange,
                   reviewCount, avgSentiment, features, userTypes
            """
            
            result = graph.query(cypher_query, params={"car_model": car_model})
            
            if not result:
                return None
            
            record = result[0]
            
            # 构建综合信息
            features_text = ""
            for feature in record['features']:
                if feature['feature']:
                    features_text += f"- {feature['feature']}: 强度 {feature['intensity']:.3f}, 评价 {feature['sentiment']}\n"
            
            users_text = ""
            for user in record['userTypes']:
                if user['userType']:
                    users_text += f"- {user['userType']}: {user['userCount']} 用户\n"
            
            content = f"""车型综合信息:
车型: {record['carModel']}
品牌: {record['brand']}
类型: {record['type']}
价格区间: {record['priceRange']}
评论数量: {record['reviewCount']}
平均满意度: {record['avgSentiment']:.3f}

特征表现:
{features_text}

主要用户群体:
{users_text}

基于知识图谱的综合分析，该车型具有明确的定位和用户群体特征。"""

            doc = Document(
                page_content=content,
                metadata={
                    'type': 'comprehensive_info',
                    'car_model': record['carModel'],
                    'avg_sentiment': record['avgSentiment'],
                    'source': 'graph_comprehensive'
                }
            )
            
            return doc
            
        except Exception as e:
            logger.error(f"获取车型综合信息失败: {e}")
            return None

# 全局实例
vector_graph_tool = VectorGraphTool()