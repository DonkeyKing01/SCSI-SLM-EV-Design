"""
Neo4j数据库连接管理器
处理数据库连接、索引创建和基本操作
"""
from neo4j import GraphDatabase, Driver, Session
from typing import Dict, List, Optional, Any
import logging
import time
import json
from contextlib import contextmanager


class Neo4jManager:
    """Neo4j数据库管理器"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None
        self.connection_established = False
        
        self._connect()
    
    def _connect(self):
        """建立数据库连接"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # 测试连接
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                result.single()
                self.connection_established = True
                logging.info("Neo4j连接成功")
        except Exception as e:
            logging.error(f"Neo4j连接失败: {e}")
            self.connection_established = False
            raise
    
    @contextmanager
    def get_session(self):
        """获取数据库会话的上下文管理器"""
        if not self.connection_established:
            raise ConnectionError("Neo4j连接未建立")
        
        session = self.driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()
    
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            self.connection_established = False
            logging.info("Neo4j连接已关闭")
    
    def create_constraints_and_indexes(self):
        """创建约束和索引"""
        constraints_and_indexes = [
            # 约束
            "CREATE CONSTRAINT car_model_id IF NOT EXISTS FOR (c:CarModel) REQUIRE c.modelId IS UNIQUE",
            "CREATE CONSTRAINT user_profile_id IF NOT EXISTS FOR (p:UserProfile) REQUIRE p.profileId IS UNIQUE",
            "CREATE CONSTRAINT review_id IF NOT EXISTS FOR (r:Review) REQUIRE r.reviewId IS UNIQUE",
            "CREATE CONSTRAINT feature_name IF NOT EXISTS FOR (f:Feature) REQUIRE f.name IS UNIQUE",
            
            # 索引
            "CREATE INDEX car_model_brand IF NOT EXISTS FOR (c:CarModel) ON (c.brand)",
            "CREATE INDEX car_model_type IF NOT EXISTS FOR (c:CarModel) ON (c.type)",
            "CREATE INDEX car_model_price IF NOT EXISTS FOR (c:CarModel) ON (c.priceRange)",
            "CREATE INDEX review_sentiment IF NOT EXISTS FOR (r:Review) ON (r.overallSentiment)",
            "CREATE INDEX user_profile_features IF NOT EXISTS FOR (p:UserProfile) ON (p.mainFeatures)",
            
            # 关系索引
            "CREATE INDEX review_car_model IF NOT EXISTS FOR ()-[r:MENTIONS]->() ON (r.sentimentScore)",
            "CREATE INDEX user_profile_review IF NOT EXISTS FOR ()-[r:PUBLISHED]->() ON (r.userMatchScore)",
            "CREATE INDEX review_feature IF NOT EXISTS FOR ()-[r:CONTAINS_ASPECT]->() ON (r.aspectSentiment, r.intensity)"
        ]
        
        with self.get_session() as session:
            for query in constraints_and_indexes:
                try:
                    session.run(query)
                    logging.info(f"成功创建约束/索引: {query[:50]}...")
                except Exception as e:
                    logging.warning(f"创建约束/索引失败: {query[:50]}... - {e}")
    
    def clear_database(self):
        """清空数据库（谨慎使用）"""
        with self.get_session() as session:
            try:
                # 删除所有关系
                session.run("MATCH ()-[r]-() DELETE r")
                # 删除所有节点
                session.run("MATCH (n) DELETE n")
                logging.info("数据库已清空")
            except Exception as e:
                logging.error(f"清空数据库失败: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        stats_queries = {
            "total_nodes": "MATCH (n) RETURN count(n) as count",
            "total_relationships": "MATCH ()-[r]-() RETURN count(r) as count",
            "car_models": "MATCH (c:CarModel) RETURN count(c) as count",
            "user_profiles": "MATCH (p:UserProfile) RETURN count(p) as count",
            "reviews": "MATCH (r:Review) RETURN count(r) as count",
            "features": "MATCH (f:Feature) RETURN count(f) as count",
            "mentions_relationships": "MATCH ()-[r:MENTIONS]->() RETURN count(r) as count",
            "published_relationships": "MATCH ()-[r:PUBLISHED]->() RETURN count(r) as count",
            "contains_aspect_relationships": "MATCH ()-[r:CONTAINS_ASPECT]->() RETURN count(r) as count",
            "interested_in_relationships": "MATCH ()-[r:INTERESTED_IN]->() RETURN count(r) as count"
        }
        
        stats = {}
        with self.get_session() as session:
            for stat_name, query in stats_queries.items():
                try:
                    result = session.run(query)
                    count = result.single()["count"]
                    stats[stat_name] = count
                except Exception as e:
                    logging.warning(f"获取统计信息失败 {stat_name}: {e}")
                    stats[stat_name] = 0
        
        return stats
    
    def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            with self.get_session() as session:
                result = session.run("RETURN 'Connection OK' as status")
                status = result.single()["status"]
                return status == "Connection OK"
        except Exception as e:
            logging.error(f"连接测试失败: {e}")
            return False
    
    def execute_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """执行Cypher查询"""
        if parameters is None:
            parameters = {}
        
        try:
            with self.get_session() as session:
                result = session.run(query, parameters)
                return [dict(record) for record in result]
        except Exception as e:
            logging.error(f"查询执行失败: {e}")
            logging.error(f"查询: {query}")
            logging.error(f"参数: {parameters}")
            raise
    
    def execute_write_query(self, query: str, parameters: Optional[Dict] = None) -> Dict:
        """执行写操作查询"""
        if parameters is None:
            parameters = {}
        
        try:
            with self.get_session() as session:
                result = session.run(query, parameters)
                summary = result.consume()
                return {
                    "nodes_created": summary.counters.nodes_created,
                    "relationships_created": summary.counters.relationships_created,
                    "properties_set": summary.counters.properties_set,
                    "labels_added": summary.counters.labels_added
                }
        except Exception as e:
            logging.error(f"写操作失败: {e}")
            logging.error(f"查询: {query}")
            logging.error(f"参数: {parameters}")
            raise
    
    def batch_create_nodes(self, nodes_data: List[Dict], batch_size: int = 1000) -> Dict[str, int]:
        """批量创建节点"""
        total_created = 0
        total_batches = 0
        
        for i in range(0, len(nodes_data), batch_size):
            batch = nodes_data[i:i + batch_size]
            batch_created = 0
            
            # 按节点类型分组
            nodes_by_type = {}
            for node in batch:
                node_type = node["type"]
                if node_type not in nodes_by_type:
                    nodes_by_type[node_type] = []
                nodes_by_type[node_type].append(node["data"])
            
            # 批量创建每种类型的节点
            for node_type, nodes in nodes_by_type.items():
                if node_type == "CarModel":
                    created = self._batch_create_car_models(nodes)
                elif node_type == "UserProfile":
                    created = self._batch_create_user_profiles(nodes)
                elif node_type == "Feature":
                    created = self._batch_create_features(nodes)
                else:
                    logging.warning(f"未知的节点类型: {node_type}")
                    continue
                
                batch_created += created
            
            total_created += batch_created
            total_batches += 1
            
            if total_batches % 10 == 0:
                logging.info(f"已处理 {total_batches} 批，创建了 {total_created} 个节点")
        
        return {"total_created": total_created, "total_batches": total_batches}
    
    def _batch_create_car_models(self, car_models: List[Dict]) -> int:
        """批量创建车型节点"""
        query = """
        UNWIND $car_models AS car
        MERGE (c:CarModel {modelId: car.modelId})
        SET c.name = car.name,
            c.brand = car.brand,
            c.type = car.type,
            c.priceRange = car.priceRange,
            c.reviewCount = car.reviewCount
        RETURN count(c) as created
        """
        
        try:
            result = self.execute_write_query(query, {"car_models": car_models})
            return result["nodes_created"]
        except Exception as e:
            logging.error(f"批量创建车型节点失败: {e}")
            return 0
    
    def _batch_create_user_profiles(self, user_profiles: List[Dict]) -> int:
        """批量创建用户画像节点"""
        # 将dimensionStrengths转换为JSON字符串，避免Neo4j Map类型错误
        for profile in user_profiles:
            if 'dimensionStrengths' in profile and isinstance(profile['dimensionStrengths'], dict):
                profile['dimensionStrengths'] = json.dumps(profile['dimensionStrengths'], ensure_ascii=False)
        
        query = """
        UNWIND $user_profiles AS profile
        MERGE (p:UserProfile {profileId: profile.profileId})
        SET p.name = profile.name,
            p.description = profile.description,
            p.userCount = profile.userCount,
            p.percentage = profile.percentage,
            p.mainFeatures = profile.mainFeatures,
            p.dimensionStrengths = profile.dimensionStrengths
        RETURN count(p) as created
        """
        
        try:
            result = self.execute_write_query(query, {"user_profiles": user_profiles})
            return result["nodes_created"]
        except Exception as e:
            logging.error(f"批量创建用户画像节点失败: {e}")
            return 0
    
    def _batch_create_features(self, features: List[Dict]) -> int:
        """批量创建特征节点"""
        query = """
        UNWIND $features AS feature
        MERGE (f:Feature {name: feature.name})
        SET f.category = feature.category,
            f.description = feature.description
        RETURN count(f) as created
        """
        
        try:
            result = self.execute_write_query(query, {"features": features})
            return result["nodes_created"]
        except Exception as e:
            logging.error(f"批量创建特征节点失败: {e}")
            return 0
    
    def create_review_nodes(self, reviews_data: List[Dict]) -> int:
        """创建评论节点"""
        query = """
        UNWIND $reviews AS review
        MERGE (r:Review {reviewId: review.reviewId})
        SET r.content = review.content,
            r.userId = review.userId,
            r.overallSentiment = review.overallSentiment,
            r.importance = review.importance
        RETURN count(r) as created
        """
        
        try:
            result = self.execute_write_query(query, {"reviews": reviews_data})
            return result["nodes_created"]
        except Exception as e:
            logging.error(f"创建评论节点失败: {e}")
            return 0
    
    def create_relationships(self, relationships_data: List[Dict]) -> Dict[str, int]:
        """创建关系"""
        total_created = 0
        relationships_by_type = {}
        
        # 按关系类型分组
        for rel in relationships_data:
            rel_type = rel["type"]
            if rel_type not in relationships_by_type:
                relationships_by_type[rel_type] = []
            relationships_by_type[rel_type].append(rel["data"])
        
        # 批量创建每种类型的关系
        for rel_type, rels in relationships_by_type.items():
            if rel_type == "MENTIONS":
                created = self._create_mentions_relationships(rels)
            elif rel_type == "PUBLISHED":
                created = self._create_published_relationships(rels)
            elif rel_type == "CONTAINS_ASPECT":
                created = self._create_contains_aspect_relationships(rels)
            elif rel_type == "INTERESTED_IN":
                created = self._create_interested_in_relationships(rels)
            else:
                logging.warning(f"未知的关系类型: {rel_type}")
                continue
            
            total_created += created
        
        return {"total_created": total_created}
    
    def _create_mentions_relationships(self, relationships: List[Dict]) -> int:
        """创建MENTIONS关系"""
        query = """
        UNWIND $relationships AS rel
        MATCH (r:Review {reviewId: rel.reviewId})
        MATCH (c:CarModel {modelId: rel.carModelId})
        MERGE (r)-[m:MENTIONS]->(c)
        SET m.sentimentScore = rel.sentimentScore,
            m.importance = rel.importance
        RETURN count(m) as created
        """
        
        try:
            result = self.execute_write_query(query, {"relationships": relationships})
            return result["relationships_created"]
        except Exception as e:
            logging.error(f"创建MENTIONS关系失败: {e}")
            return 0
    
    def _create_published_relationships(self, relationships: List[Dict]) -> int:
        """创建PUBLISHED关系"""
        query = """
        UNWIND $relationships AS rel
        MATCH (p:UserProfile {profileId: rel.profileId})
        MATCH (r:Review {reviewId: rel.reviewId})
        MERGE (p)-[pub:PUBLISHED]->(r)
        SET pub.userMatchScore = rel.userMatchScore
        RETURN count(pub) as created
        """
        
        try:
            result = self.execute_write_query(query, {"relationships": relationships})
            return result["relationships_created"]
        except Exception as e:
            logging.error(f"创建PUBLISHED关系失败: {e}")
            return 0
    
    def _create_contains_aspect_relationships(self, relationships: List[Dict]) -> int:
        """创建CONTAINS_ASPECT关系"""
        query = """
        UNWIND $relationships AS rel
        MATCH (r:Review {reviewId: rel.reviewId})
        MATCH (f:Feature {name: rel.featureName})
        MERGE (r)-[ca:CONTAINS_ASPECT]->(f)
        SET ca.aspectSentiment = rel.aspectSentiment,
            ca.intensity = rel.intensity
        RETURN count(ca) as created
        """
        
        try:
            result = self.execute_write_query(query, {"relationships": relationships})
            return result["relationships_created"]
        except Exception as e:
            logging.error(f"创建CONTAINS_ASPECT关系失败: {e}")
            return 0
    
    def _create_interested_in_relationships(self, relationships: List[Dict]) -> int:
        """创建INTERESTED_IN关系"""
        query = """
        UNWIND $relationships AS rel
        MATCH (p:UserProfile {profileId: rel.profileId})
        MATCH (c:CarModel {modelId: rel.carModelId})
        MERGE (p)-[ii:INTERESTED_IN]->(c)
        SET ii.correlationScore = rel.correlationScore,
            ii.positiveMentions = rel.positiveMentions,
            ii.negativeMentions = rel.negativeMentions,
            ii.topAspects = rel.topAspects
        RETURN count(ii) as created
        """
        
        try:
            result = self.execute_write_query(query, {"relationships": relationships})
            return result["relationships_created"]
        except Exception as e:
            logging.error(f"创建INTERESTED_IN关系失败: {e}")
            return 0


if __name__ == "__main__":
    # 测试Neo4j管理器
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 连接参数（需要根据实际情况修改）
    neo4j_manager = Neo4jManager(
        uri="bolt://localhost:7688",
        user="neo4j",
        password="neo4j123",
        database="neo4j"
    )
    
    try:
        # 测试连接
        if neo4j_manager.test_connection():
            print("Neo4j连接测试成功")
            
            # 获取数据库统计
            stats = neo4j_manager.get_database_stats()
            print(f"数据库统计: {stats}")
            
            # 创建约束和索引
            neo4j_manager.create_constraints_and_indexes()
            
        else:
            print("Neo4j连接测试失败")
    
    except Exception as e:
        print(f"测试失败: {e}")
    
    finally:
        neo4j_manager.close() 