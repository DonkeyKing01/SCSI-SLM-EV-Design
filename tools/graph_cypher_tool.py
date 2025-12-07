"""
图谱Cypher查询工具模块
专门用于执行Cypher查询和图谱分析
"""
from typing import List, Dict, Any, Optional
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from database.neo4j_connection import neo4j_conn
from config.settings import Settings
import logging

logger = logging.getLogger(__name__)

class GraphCypherTool:
    """图谱Cypher查询工具类"""
    
    def __init__(self):
        """初始化Cypher工具"""
        self.llm = None
        self.cypher_chain = None
        self.graph = None
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
                temperature=0
            )
            
            # 获取Neo4j图谱
            self.graph = neo4j_conn.get_graph()
            
            # 初始化Cypher查询链
            self._setup_cypher_chain()
            
            logger.info("GraphCypherTool初始化成功")
            
        except Exception as e:
            logger.error(f"GraphCypherTool初始化失败: {e}")
            raise
    
    def _setup_cypher_chain(self):
        """设置Cypher查询链"""
        try:
            # 创建专门的Cypher生成提示词
            cypher_prompt = PromptTemplate(
                input_variables=["schema", "question"],
                template="""
你是一个Neo4j Cypher查询专家。根据以下图谱schema和用户问题，生成准确的Cypher查询。

图谱Schema:
{schema}

重要说明：
1. 这是一个新能源汽车推荐系统的知识图谱
2. 主要节点类型：CarModel(车型)、UserProfile(用户画像)、Review(评论)、Feature(特征)
3. 主要关系：PUBLISHED(发布)、MENTIONS(提及)、CONTAINS_ASPECT(包含特征)、INTERESTED_IN(感兴趣)
4. 节点属性请参考schema中的具体定义
5. 生成的查询应该高效且准确
6. 避免返回过多数据，适当使用LIMIT
7. 优先使用索引字段进行查询

用户问题: {question}

Cypher查询:
"""
            )
            
            # 创建Cypher查询链
            self.cypher_chain = GraphCypherQAChain.from_llm(
                llm=self.llm,
                graph=self.graph,
                verbose=True,
                cypher_prompt=cypher_prompt,
                top_k=10,
                allow_dangerous_requests=True
            )
            
            logger.info("Cypher查询链设置成功")
            
        except Exception as e:
            logger.error(f"Cypher查询链设置失败: {e}")
            self.cypher_chain = None
    
    def execute_cypher_query(self, cypher_query: str, params: Dict = None) -> List[Dict]:
        """执行原生Cypher查询"""
        try:
            if not self.graph:
                logger.error("图谱连接未初始化")
                return []
            
            result = self.graph.query(cypher_query, params=params or {})
            logger.info(f"Cypher查询执行成功，返回 {len(result)} 条记录")
            
            return result
            
        except Exception as e:
            logger.error(f"Cypher查询执行失败: {e}")
            return []
    
    def natural_language_query(self, question: str) -> str:
        """自然语言转Cypher查询并执行"""
        try:
            if not self.cypher_chain:
                return "Cypher查询链未初始化，无法处理自然语言查询。"
            
            response = self.cypher_chain.run(question)
            logger.info("自然语言查询执行成功")
            
            return response
            
        except Exception as e:
            logger.error(f"自然语言查询失败: {e}")
            return f"查询执行失败: {str(e)}"
    
    def get_car_models_by_criteria(self, criteria: Dict[str, Any]) -> List[Document]:
        """根据条件查询车型"""
        try:
            # 构建查询条件
            where_clauses = []
            params = {}
            
            if criteria.get('brand'):
                where_clauses.append("c.brand = $brand")
                params['brand'] = criteria['brand']
            
            if criteria.get('price_range'):
                where_clauses.append("c.priceRange = $priceRange")
                params['priceRange'] = criteria['price_range']
            
            if criteria.get('min_reviews'):
                where_clauses.append("c.reviewCount >= $minReviews")
                params['minReviews'] = criteria['min_reviews']
            
            where_clause = " AND ".join(where_clauses) if where_clauses else "true"
            
            cypher_query = f"""
            MATCH (c:CarModel)
            WHERE {where_clause}
            OPTIONAL MATCH (r:Review)-[m:MENTIONS]->(c)
            WITH c, AVG(m.sentimentScore) as avgSentiment, COUNT(r) as reviewCount
            ORDER BY avgSentiment DESC, reviewCount DESC
            LIMIT 10
            
            RETURN c.name as carModel, c.brand as brand, c.type as type, 
                   c.priceRange as priceRange, c.reviewCount as totalReviews,
                   avgSentiment, reviewCount as mentionCount
            """
            
            result = self.execute_cypher_query(cypher_query, params)
            
            documents = []
            for record in result:
                content = f"""车型查询结果:
车型: {record['carModel']}
品牌: {record['brand']}
类型: {record['type']}
价格区间: {record['priceRange']}
总评论数: {record['totalReviews']}
平均情感评分: {record.get('avgSentiment', 0):.3f}
相关评论数: {record.get('mentionCount', 0)}

该车型符合您的筛选条件，具有良好的用户评价。"""

                doc = Document(
                    page_content=content,
                    metadata={
                        'type': 'car_query_result',
                        'car_model': record['carModel'],
                        'brand': record['brand'],
                        'source': 'cypher_query'
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"条件查询车型失败: {e}")
            return []
    
    def get_user_preferences_by_car(self, car_model: str) -> List[Document]:
        """查询特定车型的用户偏好分析"""
        try:
            cypher_query = """
            MATCH (c:CarModel {name: $carModel})
            MATCH (r:Review)-[m:MENTIONS]->(c)
            MATCH (u:UserProfile)-[pub:PUBLISHED]->(r)
            MATCH (r)-[ca:CONTAINS_ASPECT]->(f:Feature)
            
            WITH u, f, AVG(ca.intensity) as avgIntensity, COUNT(r) as reviewCount
            WHERE avgIntensity > 0.2
            ORDER BY u.userCount DESC, avgIntensity DESC
            
            RETURN u.name as userType, u.userCount as userCount, u.percentage as percentage,
                   f.name as feature, avgIntensity, reviewCount
            LIMIT 20
            """
            
            result = self.execute_cypher_query(cypher_query, {"carModel": car_model})
            
            if not result:
                return [Document(
                    page_content=f"未找到车型 {car_model} 的用户偏好数据。",
                    metadata={'type': 'no_data', 'car_model': car_model}
                )]
            
            # 按用户类型分组结果
            user_preferences = {}
            for record in result:
                user_type = record['userType']
                if user_type not in user_preferences:
                    user_preferences[user_type] = {
                        'userCount': record['userCount'],
                        'percentage': record['percentage'],
                        'features': []
                    }
                
                user_preferences[user_type]['features'].append({
                    'feature': record['feature'],
                    'intensity': record['avgIntensity'],
                    'reviews': record['reviewCount']
                })
            
            documents = []
            for user_type, data in user_preferences.items():
                features_text = ""
                for feature in data['features']:
                    features_text += f"- {feature['feature']}: 关注度 {feature['intensity']:.3f} (基于 {feature['reviews']} 条评论)\n"
                
                content = f"""车型用户偏好分析:
车型: {car_model}
用户类型: {user_type}
用户数量: {data['userCount']}
占比: {data['percentage']:.1f}%

特征关注度:
{features_text}

该用户群体对车型的特征偏好明确，为产品定位提供了重要参考。"""

                doc = Document(
                    page_content=content,
                    metadata={
                        'type': 'user_preferences',
                        'car_model': car_model,
                        'user_type': user_type,
                        'source': 'cypher_analysis'
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"查询用户偏好失败: {e}")
            return []
    
    def get_feature_rankings(self, feature_name: str) -> List[Document]:
        """获取特定特征的车型排名"""
        try:
            cypher_query = """
            MATCH (f:Feature {name: $featureName})
            MATCH (r:Review)-[ca:CONTAINS_ASPECT]->(f)
            MATCH (r)-[m:MENTIONS]->(c:CarModel)
            
            WITH c, AVG(ca.intensity) as avgIntensity, 
                 AVG(m.sentimentScore) as avgSentiment,
                 COUNT(r) as reviewCount
            WHERE reviewCount >= 5 AND avgIntensity > 0.1
            
            WITH c, avgIntensity, avgSentiment, reviewCount,
                 (avgIntensity * avgSentiment) as compositeScore
            ORDER BY compositeScore DESC
            LIMIT 10
            
            RETURN c.name as carModel, c.brand as brand, c.priceRange as priceRange,
                   avgIntensity, avgSentiment, reviewCount, compositeScore
            """
            
            result = self.execute_cypher_query(cypher_query, {"featureName": feature_name})
            
            if not result:
                return [Document(
                    page_content=f"未找到特征 {feature_name} 的相关数据。",
                    metadata={'type': 'no_data', 'feature': feature_name}
                )]
            
            # 构建排名内容
            ranking_text = f"特征 '{feature_name}' 车型排名:\n\n"
            
            for i, record in enumerate(result, 1):
                ranking_text += f"{i}. {record['carModel']} ({record['brand']})\n"
                ranking_text += f"   价格区间: {record['priceRange']}\n"
                ranking_text += f"   关注度: {record['avgIntensity']:.3f}\n"
                ranking_text += f"   满意度: {record['avgSentiment']:.3f}\n"
                ranking_text += f"   综合评分: {record['compositeScore']:.3f}\n"
                ranking_text += f"   基于评论数: {record['reviewCount']}\n\n"
            
            ranking_text += "排名基于用户关注度和满意度的综合评分，评论数量不少于5条。"
            
            doc = Document(
                page_content=ranking_text,
                metadata={
                    'type': 'feature_ranking',
                    'feature': feature_name,
                    'ranking_count': len(result),
                    'source': 'cypher_ranking'
                }
            )
            
            return [doc]
            
        except Exception as e:
            logger.error(f"获取特征排名失败: {e}")
            return []
    
    def get_recommendation_by_user_profile(self, user_preferences: Dict[str, float]) -> List[Document]:
        """基于用户偏好推荐车型"""
        try:
            # 构建用户偏好匹配查询
            feature_conditions = []
            params = {}
            
            for feature, weight in user_preferences.items():
                if weight > 0.3:  # 只考虑权重较高的特征
                    feature_conditions.append(f"(f.name = '{feature}' AND ca.intensity >= {weight * 0.8})")
            
            if not feature_conditions:
                return [Document(
                    page_content="用户偏好不明确，无法生成推荐。",
                    metadata={'type': 'no_recommendation'}
                )]
            
            cypher_query = f"""
            MATCH (f:Feature)
            WHERE {' OR '.join(feature_conditions)}
            MATCH (r:Review)-[ca:CONTAINS_ASPECT]->(f)
            MATCH (r)-[m:MENTIONS]->(c:CarModel)
            WHERE ca.sentiment = '正面' OR ca.sentiment = 'positive'
            
            WITH c, COUNT(DISTINCT f) as matchedFeatures, 
                 AVG(ca.intensity) as avgIntensity,
                 AVG(m.sentimentScore) as avgSentiment,
                 COUNT(r) as reviewCount
            WHERE matchedFeatures >= 2 AND reviewCount >= 5
            
            WITH c, matchedFeatures, avgIntensity, avgSentiment, reviewCount,
                 (matchedFeatures * avgIntensity * avgSentiment) as recommendationScore
            ORDER BY recommendationScore DESC
            LIMIT 5
            
            RETURN c.name as carModel, c.brand as brand, c.type as type,
                   c.priceRange as priceRange, matchedFeatures, avgIntensity,
                   avgSentiment, reviewCount, recommendationScore
            """
            
            result = self.execute_cypher_query(cypher_query, params)
            
            if not result:
                return [Document(
                    page_content="根据您的偏好未找到合适的车型推荐。",
                    metadata={'type': 'no_match'}
                )]
            
            documents = []
            for i, record in enumerate(result, 1):
                content = f"""个性化车型推荐 #{i}:
车型: {record['carModel']}
品牌: {record['brand']}
类型: {record['type']}
价格区间: {record['priceRange']}

匹配分析:
- 匹配特征数: {record['matchedFeatures']}
- 平均关注强度: {record['avgIntensity']:.3f}
- 平均满意度: {record['avgSentiment']:.3f}
- 推荐评分: {record['recommendationScore']:.3f}
- 基于评论数: {record['reviewCount']}

推荐理由: 该车型在您关注的特征维度上表现突出，用户满意度较高，是很好的选择。"""

                doc = Document(
                    page_content=content,
                    metadata={
                        'type': 'personalized_recommendation',
                        'car_model': record['carModel'],
                        'recommendation_score': record['recommendationScore'],
                        'rank': i,
                        'source': 'cypher_recommendation'
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"个性化推荐失败: {e}")
            return []
    
    def get_database_statistics(self) -> Document:
        """获取数据库统计信息"""
        try:
            stats_query = """
            CALL apoc.meta.stats() YIELD labelCount, relTypeCount, propertyKeyCount, nodeCount, relCount
            RETURN labelCount, relTypeCount, propertyKeyCount, nodeCount, relCount
            """
            
            try:
                stats_result = self.execute_cypher_query(stats_query)
                if stats_result:
                    stats = stats_result[0]
                else:
                    # 如果APOC不可用，使用基本查询
                    stats = self._get_basic_stats()
            except:
                stats = self._get_basic_stats()
            
            content = f"""数据库统计信息:
节点总数: {stats.get('nodeCount', '未知')}
关系总数: {stats.get('relCount', '未知')}
标签类型数: {stats.get('labelCount', '未知')}
关系类型数: {stats.get('relTypeCount', '未知')}
属性键数: {stats.get('propertyKeyCount', '未知')}

知识图谱包含了丰富的新能源汽车相关数据，支持多维度的查询和分析。"""

            return Document(
                page_content=content,
                metadata={
                    'type': 'database_stats',
                    'source': 'system_query'
                }
            )
            
        except Exception as e:
            logger.error(f"获取数据库统计失败: {e}")
            return Document(
                page_content="无法获取数据库统计信息。",
                metadata={'type': 'stats_error'}
            )
    
    def _get_basic_stats(self) -> Dict[str, int]:
        """获取基本统计信息（不依赖APOC）"""
        try:
            node_query = "MATCH (n) RETURN count(n) as nodeCount"
            rel_query = "MATCH ()-[r]->() RETURN count(r) as relCount"
            
            node_result = self.execute_cypher_query(node_query)
            rel_result = self.execute_cypher_query(rel_query)
            
            return {
                'nodeCount': node_result[0]['nodeCount'] if node_result else 0,
                'relCount': rel_result[0]['relCount'] if rel_result else 0,
                'labelCount': 4,  # 已知的节点类型数
                'relTypeCount': 4,  # 已知的关系类型数
                'propertyKeyCount': '未知'
            }
            
        except Exception as e:
            logger.error(f"获取基本统计失败: {e}")
            return {}

# 全局实例
graph_cypher_tool = GraphCypherTool()