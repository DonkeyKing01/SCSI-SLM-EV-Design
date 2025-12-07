"""
Neo4j数据库连接管理模块
"""
from neo4j import GraphDatabase
try:
    from langchain_neo4j import Neo4jGraph
except ImportError:
    from langchain_community.graphs import Neo4jGraph
from config.settings import Settings
import logging

logger = logging.getLogger(__name__)

class Neo4jConnection:
    """Neo4j连接管理器"""
    
    def __init__(self):
        self.driver = None
        self.graph = None
        self._connect()
    
    def _connect(self):
        """建立Neo4j连接"""
        try:
            config = Settings.get_neo4j_config()
            
            # 创建驱动器
            self.driver = GraphDatabase.driver(
                config['uri'],
                auth=config['auth']
            )
            
            # 测试连接
            self.driver.verify_connectivity()
            
            # 创建LangChain图形对象
            self.graph = Neo4jGraph(
                url=config['uri'],
                username=config['auth'][0],
                password=config['auth'][1],
                database=config['database']
            )
            
            logger.info("成功连接到Neo4j数据库")
            
        except Exception as e:
            logger.error(f"连接Neo4j数据库失败: {e}")
            raise
    
    def get_driver(self):
        """获取Neo4j驱动器"""
        return self.driver
    
    def get_graph(self):
        """获取LangChain图形对象"""
        return self.graph
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")
    
    def test_connection(self):
        """测试数据库连接并返回基本信息"""
        try:
            with self.driver.session() as session:
                result = session.run("CALL db.info()")
                info = result.single()
                
                # 获取节点和关系数量
                node_count_result = session.run("MATCH (n) RETURN count(n) as count")
                node_count = node_count_result.single()['count']
                
                rel_count_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                rel_count = rel_count_result.single()['count']
                
                return {
                    'status': 'connected',
                    'database_info': dict(info) if info else {},
                    'node_count': node_count,
                    'relationship_count': rel_count
                }
                
        except Exception as e:
            logger.error(f"测试连接失败: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

# 全局连接实例
neo4j_conn = Neo4jConnection()