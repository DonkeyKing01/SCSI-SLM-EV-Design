"""
搜索配置模块
定义搜索模式、问题类型等配置信息
"""
from typing import Dict, List, Set
from enum import Enum

class SearchMode(Enum):
    """搜索模式枚举"""
    VECTOR = "vector"      # 纯向量语义搜索
    GRAPH = "graph"        # 智能图谱搜索
    CYPHER = "cypher"      # 精确Cypher查询
    AUTO = "auto"          # 自动选择模式

class QuestionType(Enum):
    """问题类型枚举"""
    STATISTICS = "statistics"           # 统计查询
    CAR_RECOMMENDATION = "car_recommendation"  # 车型推荐
    FEATURE_COMPARISON = "feature_comparison"  # 特征对比
    USER_ANALYSIS = "user_analysis"     # 用户分析
    CAR_INFO = "car_info"              # 车型信息
    GENERAL_SEARCH = "general_search"   # 一般搜索

class SearchConfig:
    """搜索配置类"""
    
    # 搜索模式配置
    SEARCH_MODES = {
        SearchMode.VECTOR: {
            "name": "向量搜索",
            "description": "基于语义相似度的向量检索",
            "tool": "vector_tool",
            "method": "search",
            "suitable_for": [QuestionType.GENERAL_SEARCH, QuestionType.CAR_INFO]
        },
        SearchMode.GRAPH: {
            "name": "图谱搜索", 
            "description": "基于知识图谱关系的智能搜索",
            "tool": "vector_graph_tool",
            "method": "hybrid_search",
            "suitable_for": [
                QuestionType.CAR_RECOMMENDATION, 
                QuestionType.FEATURE_COMPARISON,
                QuestionType.USER_ANALYSIS,
                QuestionType.GENERAL_SEARCH
            ]
        },
        SearchMode.CYPHER: {
            "name": "Cypher查询",
            "description": "直接使用Cypher查询语言进行精确搜索",
            "tool": "graph_cypher_tool", 
            "method": "natural_language_query",
            "suitable_for": [QuestionType.STATISTICS]
        }
    }
    
    # 问题类型识别关键词
    QUESTION_TYPE_KEYWORDS = {
        QuestionType.STATISTICS: {
            "keywords": ["几款", "多少", "统计", "数量", "总共", "count", "有多少", "几个", "几种"],
            "priority": 10  # 最高优先级，统计问题需要精确答案
        },
        QuestionType.CAR_RECOMMENDATION: {
            "keywords": ["推荐", "选择", "买", "哪个好", "适合", "建议", "推荐一下"],
            "priority": 8
        },
        QuestionType.FEATURE_COMPARISON: {
            "keywords": ["对比", "比较", "区别", "vs", "差异", "哪个更好"],
            "priority": 7
        },
        QuestionType.USER_ANALYSIS: {
            "keywords": ["用户", "画像", "喜欢", "偏好", "用户群体", "用户类型"],
            "priority": 6
        },
        QuestionType.CAR_INFO: {
            "keywords": ["信息", "详情", "介绍", "参数", "怎么样", "如何"],
            "priority": 5
        },
        QuestionType.GENERAL_SEARCH: {
            "keywords": [],  # 默认类型，无特定关键词
            "priority": 1
        }
    }
    
    # 自动模式选择策略
    AUTO_MODE_STRATEGY = {
        QuestionType.STATISTICS: SearchMode.CYPHER,
        QuestionType.CAR_RECOMMENDATION: SearchMode.GRAPH,
        QuestionType.FEATURE_COMPARISON: SearchMode.GRAPH,
        QuestionType.USER_ANALYSIS: SearchMode.GRAPH,
        QuestionType.CAR_INFO: SearchMode.GRAPH,
        QuestionType.GENERAL_SEARCH: SearchMode.GRAPH  # 默认使用图谱搜索
    }
    
    # 搜索参数配置
    SEARCH_PARAMS = {
        "default_k": 5,           # 默认返回结果数
        "max_results": 10,        # 最大结果数
        "min_content_length": 30, # 最小内容长度
        "max_context_length": 3000 # 最大上下文长度
    }
    
    # Cypher查询模板（用于降级场景）
    CYPHER_TEMPLATES = {
        QuestionType.STATISTICS: {
            "car_count": """
                MATCH (c:CarModel)
                WHERE c.name CONTAINS $keyword OR c.type CONTAINS $keyword
                RETURN count(DISTINCT c) as total_count, 
                       collect(DISTINCT c.name)[0..10] as sample_models
            """,
            "total_cars": """
                MATCH (c:CarModel)
                RETURN count(c) as total_count,
                       collect(c.name)[0..10] as sample_models
            """
        }
    }
    
    @classmethod
    def get_search_mode_info(cls, mode: SearchMode) -> Dict:
        """获取搜索模式信息"""
        return cls.SEARCH_MODES.get(mode, {})
    
    @classmethod
    def get_question_types(cls) -> List[QuestionType]:
        """获取所有问题类型"""
        return list(cls.QUESTION_TYPE_KEYWORDS.keys())
    
    @classmethod
    def get_auto_mode(cls, question_type: QuestionType) -> SearchMode:
        """根据问题类型获取推荐的搜索模式"""
        return cls.AUTO_MODE_STRATEGY.get(question_type, SearchMode.GRAPH)
    
    @classmethod
    def get_cypher_template(cls, question_type: QuestionType, template_name: str) -> str:
        """获取Cypher查询模板"""
        return cls.CYPHER_TEMPLATES.get(question_type, {}).get(template_name, "")