"""
RAG系统配置模块
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# 加载根目录的环境变量
root_dir = Path(__file__).parent.parent.parent
load_dotenv(root_dir / '.env')

class Settings:
    """系统配置类"""
    
    # Neo4j配置
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7688')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'neo4j123')
    NEO4J_DATABASE = os.getenv('NEO4J_DATABASE', 'neo4jfinal')
    
    # OpenAI配置
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        raise ValueError(
            "未找到 OPENAI_API_KEY 环境变量。\n"
            "请在项目根目录创建 .env 文件，并配置 OPENAI_API_KEY。\n"
            "参考根目录的 .env.example 文件进行配置。"
        )
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.siliconflow.cn/v1')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'Pro/THUDM/glm-4-9b-chat')
    
    # 应用配置
    APP_TITLE = os.getenv('APP_TITLE', '新能源汽车智能推荐系统')
    APP_ICON = os.getenv('APP_ICON', '🚗')
    DISABLE_ANALYTICS = os.getenv('DISABLE_ANALYTICS', 'true').lower() == 'true'
    
    # 向量数据库配置
    VECTOR_STORE_PATH = os.getenv('VECTOR_STORE_PATH', './vector_store')
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-large-zh-v1.5')
    
    # [新增] 评估日志路径
    EVAL_LOG_PATH = os.getenv('EVAL_LOG_PATH', './logs/rag_eval_logs.jsonl')
    
    # 系统提示词模板
    SYSTEM_PROMPT_TEMPLATE = """
你是一个专业的新能源汽车推荐助手。你拥有以下知识：
- 丰富的车型特征数据（外观设计、内饰质感、智能配置、空间实用、舒适体验、操控性能、续航能耗、价值认知）
- 详细的用户画像分析
- 真实的用户评论数据
- 车型之间的关系图谱

请基于提供的上下文信息回答用户问题，要求：
1. 回答要准确、客观，基于数据说话
2. 如果涉及推荐，要考虑用户的具体需求
3. 可以引用具体的数据和用户评论来支持观点
4. 保持友好和专业的语调
5. 如果不确定答案，诚实说明

上下文信息：
{context}

用户问题：{question}
"""

    @classmethod
    def get_neo4j_config(cls):
        """获取Neo4j连接配置"""
        return {
            'uri': cls.NEO4J_URI,
            'auth': (cls.NEO4J_USERNAME, cls.NEO4J_PASSWORD),
            'database': cls.NEO4J_DATABASE
        }
    
    @classmethod
    def get_openai_config(cls):
        """获取OpenAI配置"""
        return {
            'api_key': cls.OPENAI_API_KEY,
            'base_url': cls.OPENAI_BASE_URL,
            'model': cls.OPENAI_MODEL
        }