"""
Neo4j数据库连接器
"""
import os
import sys
from neo4j import GraphDatabase
from loguru import logger
from typing import Dict, List, Any, Optional

# 添加config路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.database import DatabaseConfig


class Neo4jConnector:
    """Neo4j数据库连接器类"""
    
    def __init__(self):
        """初始化连接器"""
        self.driver = None
        self.connect()
    
    def connect(self):
        """连接到Neo4j数据库"""
        try:
            config = DatabaseConfig.get_connection_params()
            self.driver = GraphDatabase.driver(
                config['uri'], 
                auth=config['auth']
            )
            # 测试连接
            with self.driver.session(database=config['database']) as session:
                session.run("RETURN 1")
            logger.info(f"成功连接到Neo4j数据库: {config['uri']}")
        except Exception as e:
            logger.error(f"连接Neo4j数据库失败: {e}")
            raise
    
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")
    
    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict]:
        """执行Cypher查询"""
        try:
            with self.driver.session(database=DatabaseConfig.NEO4J_DATABASE) as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"查询执行失败: {e}")
            raise
    
    def execute_write_transaction(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict]:
        """执行写事务"""
        try:
            with self.driver.session(database=DatabaseConfig.NEO4J_DATABASE) as session:
                result = session.write_transaction(self._execute_query, query, parameters or {})
                return result
        except Exception as e:
            logger.error(f"写事务执行失败: {e}")
            raise
    
    def execute_read_transaction(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict]:
        """执行读事务"""
        try:
            with self.driver.session(database=DatabaseConfig.NEO4J_DATABASE) as session:
                result = session.read_transaction(self._execute_query, query, parameters or {})
                return result
        except Exception as e:
            logger.error(f"读事务执行失败: {e}")
            raise
    
    @staticmethod
    def _execute_query(tx, query: str, parameters: Dict[str, Any]):
        """事务内部查询执行"""
        result = tx.run(query, parameters)
        return [record.data() for record in result]
    
    def create_constraints_and_indexes(self):
        """创建约束和索引"""
        constraints_and_indexes = [
            # 创建唯一约束
            "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT company_id IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE",
            
            # 创建索引
            "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)",
            "CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name)",
        ]
        
        for query in constraints_and_indexes:
            try:
                self.execute_query(query)
                logger.info(f"成功执行: {query}")
            except Exception as e:
                logger.warning(f"执行失败 (可能已存在): {query}, 错误: {e}")
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库信息"""
        info_queries = {
            "nodes_count": "MATCH (n) RETURN count(n) as count",
            "relationships_count": "MATCH ()-[r]->() RETURN count(r) as count",
            "labels": "CALL db.labels() YIELD label RETURN collect(label) as labels",
            "relationship_types": "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"
        }
        
        result = {}
        for key, query in info_queries.items():
            try:
                data = self.execute_query(query)
                if key in ["labels", "relationship_types"]:
                    result[key] = data[0][key.split('_')[0] if key == "relationship_types" else key] if data else []
                else:
                    result[key] = data[0]["count"] if data else 0
            except Exception as e:
                logger.error(f"获取{key}失败: {e}")
                result[key] = "Error"
        
        return result


# 全局连接器实例
neo4j_connector = Neo4jConnector()