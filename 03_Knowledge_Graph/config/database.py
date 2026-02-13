"""
Neo4j数据库配置模块
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# 加载根目录的环境变量
root_dir = Path(__file__).parent.parent.parent
load_dotenv(root_dir / '.env')

class DatabaseConfig:
    """数据库配置类"""
    
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7688')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'neo4j123')
    NEO4J_DATABASE = os.getenv('NEO4J_DATABASE', 'neo4jfinal')
    
    @classmethod
    def get_connection_params(cls):
        """获取数据库连接参数"""
        return {
            'uri': cls.NEO4J_URI,
            'auth': (cls.NEO4J_USERNAME, cls.NEO4J_PASSWORD),
            'database': cls.NEO4J_DATABASE
        }