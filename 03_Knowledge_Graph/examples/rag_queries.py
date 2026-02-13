"""
RAG查询示例
展示如何使用构建好的知识图谱进行各种查询
"""
from typing import List, Dict, Any
import logging


class RAGQueryEngine:
    """RAG查询引擎"""
    
    def __init__(self, neo4j_manager):
        self.neo4j_manager = neo4j_manager
    
    def recommend_cars_by_user_profile(self, profile_id: int, price_ranges: List[str] = None, limit: int = 5) -> List[Dict]:
        """
        基于用户画像推荐车型
        
        参数:
        - profile_id: 用户画像ID (0-29)
        - price_ranges: 价格区间过滤
        - limit: 返回结果数量限制
        """
        query = """
        MATCH (p:UserProfile {profileId: $profileId})
        MATCH (p)-[i:INTERESTED_IN]->(c:CarModel)
        """
        
        if price_ranges:
            query += "WHERE c.priceRange IN $priceRanges\n"
        
        query += """
        RETURN c.name as carName, 
               c.brand as brand, 
               c.type as type,
               c.priceRange as priceRange,
               i.correlationScore as score,
               i.positiveMentions as positiveCount,
               i.negativeMentions as negativeCount,
               i.topAspects as topFeatures
        ORDER BY i.correlationScore DESC
        LIMIT $limit
        """
        
        params = {"profileId": profile_id, "limit": limit}
        if price_ranges:
            params["priceRanges"] = price_ranges
        
        try:
            results = self.neo4j_manager.execute_query(query, params)
            return results
        except Exception as e:
            logging.error(f"推荐查询失败: {e}")
            return []
    
    def get_car_feature_analysis(self, car_name: str, min_intensity: float = 0.3) -> List[Dict]:
        """
        获取车型特征维度分析
        
        参数:
        - car_name: 车型名称
        - min_intensity: 最小强度阈值
        """
        query = """
        MATCH (c:CarModel {name: $carName})
        MATCH (r:Review)-[:MENTIONS]->(c)
        MATCH (r)-[ca:CONTAINS_ASPECT]->(f:Feature)
        WHERE ca.intensity >= $minIntensity
        RETURN f.name as featureName,
               f.category as category,
               avg(ca.intensity) as avgIntensity,
               count(*) as mentionCount,
               sum(CASE WHEN ca.aspectSentiment = '正面' THEN 1 ELSE 0 END) as positiveCount,
               sum(CASE WHEN ca.aspectSentiment = '负面' THEN 1 ELSE 0 END) as negativeCount,
               sum(CASE WHEN ca.aspectSentiment = '中性' THEN 1 ELSE 0 END) as neutralCount
        GROUP BY f.name, f.category
        ORDER BY avgIntensity DESC
        """
        
        try:
            results = self.neo4j_manager.execute_query(query, {
                "carName": car_name,
                "minIntensity": min_intensity
            })
            return results
        except Exception as e:
            logging.error(f"特征分析查询失败: {e}")
            return []
    
    def get_competitive_analysis(self, car_names: List[str]) -> Dict[str, Any]:
        """
        获取竞品分析
        
        参数:
        - car_names: 车型名称列表
        """
        analysis = {}
        
        for car_name in car_names:
            # 获取车型基本信息
            car_info_query = """
            MATCH (c:CarModel {name: $carName})
            RETURN c.name as name, c.brand as brand, c.type as type, c.priceRange as priceRange
            """
            
            try:
                car_info = self.neo4j_manager.execute_query(car_info_query, {"carName": car_name})
                if car_info:
                    analysis[car_name] = {
                        "info": car_info[0],
                        "features": self.get_car_feature_analysis(car_name),
                        "sentiment": self.get_car_sentiment_analysis(car_name)
                    }
            except Exception as e:
                logging.error(f"获取车型 {car_name} 信息失败: {e}")
                analysis[car_name] = {"error": str(e)}
        
        return analysis
    
    def get_car_sentiment_analysis(self, car_name: str) -> Dict[str, Any]:
        """获取车型情感分析"""
        query = """
        MATCH (c:CarModel {name: $carName})
        MATCH (r:Review)-[:MENTIONS]->(c)
        RETURN count(r) as totalReviews,
               avg(r.overallSentiment) as avgSentiment,
               sum(CASE WHEN r.overallSentiment > 0 THEN 1 ELSE 0 END) as positiveReviews,
               sum(CASE WHEN r.overallSentiment < 0 THEN 1 ELSE 0 END) as negativeReviews,
               sum(CASE WHEN r.overallSentiment = 0 THEN 1 ELSE 0 END) as neutralReviews
        """
        
        try:
            results = self.neo4j_manager.execute_query(query, {"carName": car_name})
            if results:
                return results[0]
            return {}
        except Exception as e:
            logging.error(f"情感分析查询失败: {e}")
            return {}
    
    def search_reviews_by_keywords(self, keywords: List[str], sentiment_filter: str = None, limit: int = 20) -> List[Dict]:
        """
        基于关键词搜索评论
        
        参数:
        - keywords: 关键词列表
        - sentiment_filter: 情感过滤 (positive/negative/neutral)
        - limit: 返回结果数量限制
        """
        # 构建关键词匹配条件
        keyword_conditions = []
        for i, keyword in enumerate(keywords):
            keyword_conditions.append(f"r.content CONTAINS '{keyword}'")
        
        query = """
        MATCH (r:Review)
        WHERE """ + " OR ".join(keyword_conditions)
        
        if sentiment_filter:
            if sentiment_filter == "positive":
                query += " AND r.overallSentiment > 0"
            elif sentiment_filter == "negative":
                query += " AND r.overallSentiment < 0"
            elif sentiment_filter == "neutral":
                query += " AND r.overallSentiment = 0"
        
        query += """
        MATCH (r)-[:MENTIONS]->(c:CarModel)
        RETURN r.content as content,
               r.overallSentiment as sentiment,
               r.importance as importance,
               c.name as carName,
               c.brand as brand
        ORDER BY r.importance DESC
        LIMIT $limit
        """
        
        try:
            results = self.neo4j_manager.execute_query(query, {"limit": limit})
            return results
        except Exception as e:
            logging.error(f"关键词搜索失败: {e}")
            return []
    
    def get_brand_insights(self, brand_name: str) -> Dict[str, Any]:
        """获取品牌洞察分析"""
        # 品牌车型统计
        brand_stats_query = """
        MATCH (c:CarModel {brand: $brandName})
        RETURN count(c) as totalModels,
               collect(c.name) as modelNames,
               collect(c.type) as modelTypes,
               collect(c.priceRange) as priceRanges
        """
        
        # 品牌情感分析
        brand_sentiment_query = """
        MATCH (c:CarModel {brand: $brandName})
        MATCH (r:Review)-[:MENTIONS]->(c)
        RETURN avg(r.overallSentiment) as avgBrandSentiment,
               count(r) as totalReviews,
               sum(CASE WHEN r.overallSentiment > 0 THEN 1 ELSE 0 END) as positiveCount,
               sum(CASE WHEN r.overallSentiment < 0 THEN 1 ELSE 0 END) as negativeCount
        """
        
        # 品牌特征分析
        brand_features_query = """
        MATCH (c:CarModel {brand: $brandName})
        MATCH (r:Review)-[:MENTIONS]->(c)
        MATCH (r)-[ca:CONTAINS_ASPECT]->(f:Feature)
        WHERE ca.intensity >= 0.3
        RETURN f.name as featureName,
               avg(ca.intensity) as avgIntensity,
               count(*) as mentionCount,
               sum(CASE WHEN ca.aspectSentiment = '正面' THEN 1 ELSE 0 END) as positiveCount
        GROUP BY f.name
        ORDER BY avgIntensity DESC
        LIMIT 5
        """
        
        try:
            brand_stats = self.neo4j_manager.execute_query(brand_stats_query, {"brandName": brand_name})
            brand_sentiment = self.neo4j_manager.execute_query(brand_sentiment_query, {"brandName": brand_name})
            brand_features = self.neo4j_manager.execute_query(brand_features_query, {"brandName": brand_name})
            
            return {
                "brandStats": brand_stats[0] if brand_stats else {},
                "brandSentiment": brand_sentiment[0] if brand_sentiment else {},
                "topFeatures": brand_features
            }
        except Exception as e:
            logging.error(f"品牌洞察查询失败: {e}")
            return {}
    
    def get_user_profile_analysis(self, profile_id: int) -> Dict[str, Any]:
        """获取用户画像分析"""
        # 画像基本信息
        profile_query = """
        MATCH (p:UserProfile {profileId: $profileId})
        RETURN p.name as name,
               p.description as description,
               p.userCount as userCount,
               p.percentage as percentage,
               p.mainFeatures as mainFeatures,
               p.dimensionStrengths as dimensionStrengths
        """
        
        # 画像评论统计
        profile_reviews_query = """
        MATCH (p:UserProfile {profileId: $profileId})
        MATCH (p)-[:PUBLISHED]->(r:Review)
        RETURN count(r) as totalReviews,
               avg(r.overallSentiment) as avgSentiment,
               avg(r.importance) as avgImportance
        """
        
        # 画像最关注的车型
        profile_cars_query = """
        MATCH (p:UserProfile {profileId: $profileId})
        MATCH (p)-[:INTERESTED_IN]->(c:CarModel)
        RETURN c.name as carName,
               c.brand as brand,
               c.type as type,
               c.priceRange as priceRange,
               i.correlationScore as score
        ORDER BY i.correlationScore DESC
        LIMIT 10
        """
        
        try:
            profile_info = self.neo4j_manager.execute_query(profile_query, {"profileId": profile_id})
            profile_reviews = self.neo4j_manager.execute_query(profile_reviews_query, {"profileId": profile_id})
            profile_cars = self.neo4j_manager.execute_query(profile_cars_query, {"profileId": profile_id})
            
            return {
                "profileInfo": profile_info[0] if profile_info else {},
                "reviewStats": profile_reviews[0] if profile_reviews else {},
                "topCars": profile_cars
            }
        except Exception as e:
            logging.error(f"用户画像分析查询失败: {e}")
            return {}
    
    def get_similar_cars(self, car_name: str, similarity_threshold: float = 0.7) -> List[Dict]:
        """获取相似车型推荐"""
        query = """
        MATCH (target:CarModel {name: $carName})
        MATCH (target)<-[:MENTIONS]-(r1:Review)-[:CONTAINS_ASPECT]->(f:Feature)
        MATCH (other:CarModel)<-[:MENTIONS]-(r2:Review)-[:CONTAINS_ASPECT]->(f)
        WHERE target <> other
        WITH target, other, 
             avg(abs(r1.overallSentiment - r2.overallSentiment)) as sentimentSimilarity,
             count(DISTINCT f) as commonFeatures
        WHERE sentimentSimilarity < $threshold AND commonFeatures >= 3
        RETURN other.name as carName,
               other.brand as brand,
               other.type as type,
               other.priceRange as priceRange,
               sentimentSimilarity as similarity,
               commonFeatures as featureOverlap
        ORDER BY sentimentSimilarity ASC, featureOverlap DESC
        LIMIT 10
        """
        
        try:
            results = self.neo4j_manager.execute_query(query, {
                "carName": car_name,
                "threshold": similarity_threshold
            })
            return results
        except Exception as e:
            logging.error(f"相似车型查询失败: {e}")
            return []


def demonstrate_rag_queries(neo4j_manager):
    """演示RAG查询功能"""
    logging.info("开始RAG查询演示...")
    
    query_engine = RAGQueryEngine(neo4j_manager)
    
    # 1. 基于用户画像推荐车型
    logging.info("\n1. 基于用户画像推荐车型 (画像ID: 0)")
    recommendations = query_engine.recommend_cars_by_user_profile(
        profile_id=0, 
        price_ranges=["中端(20-30万)", "中高端(30-50万)"], 
        limit=5
    )
    
    for i, rec in enumerate(recommendations, 1):
        logging.info(f"  {i}. {rec['carName']} ({rec['brand']}) - 评分: {rec['score']:.3f}")
    
    # 2. 车型特征分析
    logging.info("\n2. 小米SU7特征分析")
    features = query_engine.get_car_feature_analysis("小米SU7", min_intensity=0.3)
    
    for feature in features[:5]:
        logging.info(f"  {feature['featureName']}: 强度 {feature['avgIntensity']:.3f}, "
                    f"提及 {feature['mentionCount']} 次")
    
    # 3. 竞品分析
    logging.info("\n3. 竞品分析: 小米SU7 vs 特斯拉Model 3")
    comparison = query_engine.get_competitive_analysis(["小米SU7", "特斯拉Model 3"])
    
    for car_name, analysis in comparison.items():
        if "error" not in analysis:
            logging.info(f"  {car_name}: {analysis['info']['brand']} {analysis['info']['type']}")
            if analysis['sentiment']:
                sentiment = analysis['sentiment']
                logging.info(f"    情感评分: {sentiment.get('avgSentiment', 0):.3f}")
    
    # 4. 关键词搜索
    logging.info("\n4. 关键词搜索: '续航'")
    search_results = query_engine.search_reviews_by_keywords(
        keywords=["续航"], 
        sentiment_filter="positive",
        limit=5
    )
    
    for i, result in enumerate(search_results, 1):
        logging.info(f"  {i}. {result['carName']}: {result['content'][:50]}...")
    
    # 5. 品牌洞察
    logging.info("\n5. 小米品牌洞察")
    brand_insights = query_engine.get_brand_insights("小米")
    
    if brand_insights and "brandStats" in brand_insights:
        stats = brand_insights["brandStats"]
        logging.info(f"  车型数量: {stats.get('totalModels', 0)}")
        logging.info(f"  车型列表: {', '.join(stats.get('modelNames', []))}")
    
    # 6. 用户画像分析
    logging.info("\n6. 用户画像分析 (画像ID: 1)")
    profile_analysis = query_engine.get_user_profile_analysis(1)
    
    if profile_analysis and "profileInfo" in profile_analysis:
        info = profile_analysis["profileInfo"]
        logging.info(f"  画像名称: {info.get('name', 'N/A')}")
        logging.info(f"  用户数量: {info.get('userCount', 0)}")
        logging.info(f"  主要特征: {', '.join(info.get('mainFeatures', []))}")
    
    # 7. 相似车型推荐
    logging.info("\n7. 与小米SU7相似的车型")
    similar_cars = query_engine.get_similar_cars("小米SU7", similarity_threshold=0.5)
    
    for i, car in enumerate(similar_cars[:5], 1):
        logging.info(f"  {i}. {car['carName']} ({car['brand']}) - 相似度: {car['similarity']:.3f}")
    
    logging.info("\nRAG查询演示完成！")


if __name__ == "__main__":
    # 测试RAG查询引擎
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 需要先建立Neo4j连接
    try:
        from src.neo4j_manager import Neo4jManager
        
        neo4j_manager = Neo4jManager(
            uri="bolt://localhost:7688",
            user="neo4j",
            password="neo4j123",
            database="neo4j"
        )
        
        if neo4j_manager.test_connection():
            demonstrate_rag_queries(neo4j_manager)
        else:
            logging.error("Neo4j连接失败，无法运行演示")
            
    except Exception as e:
        logging.error(f"演示失败: {e}")
    finally:
        if 'neo4j_manager' in locals():
            neo4j_manager.close()