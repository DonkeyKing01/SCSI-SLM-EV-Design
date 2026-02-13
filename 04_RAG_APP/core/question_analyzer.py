"""
问题分析器模块
用于分析用户问题类型并选择合适的搜索策略
"""
from typing import Dict, List, Tuple, Optional
import re
import jieba
import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config.search_config import SearchConfig, QuestionType, SearchMode
from config.settings import Settings

logger = logging.getLogger(__name__)

class QuestionAnalyzer:
    """问题分析器类"""
    
    def __init__(self):
        """初始化问题分析器"""
        self.config = SearchConfig()
        self.llm = None
        self._load_stopwords()
        self._initialize_llm()
    
    def _initialize_llm(self):
        """初始化LLM"""
        try:
            openai_config = Settings.get_openai_config()
            self.llm = ChatOpenAI(
                openai_api_key=openai_config['api_key'],
                openai_api_base=openai_config['base_url'],
                model_name=openai_config['model'],
                temperature=0
            )
            logger.info("问题分析器LLM初始化成功")
        except Exception as e:
            logger.error(f"问题分析器LLM初始化失败: {e}")
            self.llm = None
    
    def _load_stopwords(self):
        """加载停用词"""
        self.stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', 
            '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会',
            '着', '没有', '看', '好', '自己', '这', '呢', '吗', '啊', '吧', '呀'
        }
    
    def analyze_question(self, question: str) -> Dict[str, any]:
        """
        分析问题，返回问题类型和相关信息
        
        Args:
            question: 用户问题
            
        Returns:
            Dict: 包含问题类型、关键词、推荐搜索模式等信息
        """
        try:
            # 预处理问题
            cleaned_question = self._preprocess_question(question)
            
            # 提取关键词
            keywords = self._extract_keywords(cleaned_question)
            
            # AI驱动的问题类型识别
            if self.llm:
                question_type, confidence, recommended_mode = self._ai_analyze_question(cleaned_question)
            else:
                # 降级到关键词匹配
                question_type, confidence = self._identify_question_type(cleaned_question)
                recommended_mode = self.config.get_auto_mode(question_type)
            
            # 提取实体
            entities = self._extract_entities(cleaned_question)
            
            analysis_result = {
                'question': question,
                'cleaned_question': cleaned_question,
                'question_type': question_type,
                'confidence': confidence,
                'keywords': keywords,
                'entities': entities,
                'recommended_mode': recommended_mode,
                'search_params': self._get_search_params(question_type)
            }
            
            logger.info(f"问题分析结果: 类型={question_type.value}, 置信度={confidence:.2f}, 推荐模式={recommended_mode.value}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"问题分析失败: {e}")
            # 返回默认分析结果
            return self._get_default_analysis(question)
    
    def _preprocess_question(self, question: str) -> str:
        """预处理问题文本"""
        # 转换为小写
        question = question.lower().strip()
        
        # 移除特殊字符（保留中文、英文、数字）
        question = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', question)
        
        return question
    
    def _extract_keywords(self, question: str) -> List[str]:
        """提取关键词"""
        try:
            # 使用jieba分词
            words = list(jieba.cut(question))
            
            # 过滤停用词和单字符词
            keywords = [
                word for word in words 
                if len(word) > 1 and word not in self.stopwords
            ]
            
            return keywords[:5]  # 最多返回5个关键词
            
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            return [question]  # 降级返回原问题
    
    def _identify_question_type(self, question: str) -> Tuple[QuestionType, float]:
        """
        识别问题类型
        
        Returns:
            Tuple[QuestionType, float]: 问题类型和置信度
        """
        type_scores = {}
        
        # 遍历所有问题类型的关键词
        for question_type, config in self.config.QUESTION_TYPE_KEYWORDS.items():
            score = 0
            keywords = config['keywords']
            priority = config['priority']
            
            if keywords:  # 如果有关键词配置
                for keyword in keywords:
                    if keyword in question:
                        # 计算分数：优先级 * 关键词权重
                        score += priority * (1.0 + len(keyword) * 0.1)
                
                type_scores[question_type] = score
        
        # 如果没有匹配到任何类型，返回默认类型
        if not type_scores:
            return QuestionType.GENERAL_SEARCH, 0.5
        
        # 找到得分最高的类型
        best_type = max(type_scores, key=type_scores.get)
        max_score = type_scores[best_type]
        
        # 计算置信度（0-1之间）
        confidence = min(max_score / 20.0, 1.0)  # 最大分数约20分
        
        return best_type, confidence
    
    def _extract_entities(self, question: str) -> Dict[str, List[str]]:
        """提取实体信息"""
        entities = {
            'brands': [],
            'car_models': [],
            'features': []
        }
        
        # 品牌识别
        brand_patterns = [
            '小米', '智界', '享界', '极氪', '宝马', '奔驰', '特斯拉', 
            '比亚迪', '蔚来', '理想', '问界', '零跑', '哪吒', '威马',
            '小鹏', '岚图', '高合', '红旗', '吉利', '长城'
        ]
        
        for brand in brand_patterns:
            if brand in question:
                entities['brands'].append(brand)
        
        # 特征识别
        feature_patterns = [
            '外观', '内饰', '智能', '空间', '舒适', '操控', '续航', '能耗',
            '价格', '性价比', '安全', '质量', '服务', '充电', '动力'
        ]
        
        for feature in feature_patterns:
            if feature in question:
                entities['features'].append(feature)
        
        return entities
    
    def _get_search_params(self, question_type: QuestionType) -> Dict[str, any]:
        """根据问题类型获取搜索参数"""
        base_params = self.config.SEARCH_PARAMS.copy()
        
        # 根据问题类型调整参数
        if question_type == QuestionType.STATISTICS:
            base_params['default_k'] = 1  # 统计问题只需要少量精确结果
            base_params['max_results'] = 3
        elif question_type == QuestionType.CAR_RECOMMENDATION:
            base_params['default_k'] = 8  # 推荐问题需要更多选择
            base_params['max_results'] = 10
        elif question_type == QuestionType.FEATURE_COMPARISON:
            base_params['default_k'] = 6  # 对比问题需要平衡的结果
            base_params['max_results'] = 8
        
        return base_params
    
    def _ai_analyze_question(self, question: str) -> Tuple[QuestionType, float, SearchMode]:
        """使用AI智能分析问题类型和推荐搜索模式"""
        try:
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是新能源汽车推荐系统的问题分析专家。分析用户问题，返回问题类型、置信度和推荐搜索模式。

问题类型（必须选择其中一个）：
- statistics: 统计查询（如"有多少种电车"、"总共几款车型"）
- car_recommendation: 车型推荐（如"推荐一款车"、"哪个好"）
- feature_comparison: 特征对比（如"对比续航"、"比较性能"）  
- user_analysis: 用户分析（如"用户喜好"、"用户画像"）
- car_info: 车型信息（如"特斯拉怎么样"、"车型参数"）
- general_search: 一般搜索（其他问题）

搜索模式（必须选择其中一个）：
- cypher: 精确查询，适合统计问题
- graph: 图谱搜索，适合推荐、对比、信息查询
- vector: 向量搜索，适合语义相似性搜索

重要：只返回JSON格式，不要任何其他文字。示例格式：
{{"question_type": "statistics", "confidence": 0.95, "search_mode": "cypher", "reasoning": "这是统计问题"}}"""),
                ("user", f"请分析这个问题：{question}")
            ])
            
            response = self.llm.invoke(analysis_prompt.format_messages())
            result_text = response.content.strip()
            
            # 记录原始响应用于调试
            logger.debug(f"AI原始响应: {result_text}")
            
            # 更强的清理逻辑
            # 移除markdown代码块标记
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            elif result_text.startswith('```'):
                result_text = result_text[3:]
            
            if result_text.endswith('```'):
                result_text = result_text[:-3]
            
            # 移除可能的前导文字
            json_start = result_text.find('{')
            json_end = result_text.rfind('}')
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                result_text = result_text[json_start:json_end+1]
            
            result_text = result_text.strip()
            
            # 解析JSON结果
            import json
            try:
                result = json.loads(result_text)
                
                # 验证必需字段
                required_fields = ['question_type', 'confidence', 'search_mode']
                for field in required_fields:
                    if field not in result:
                        raise KeyError(f"缺少必需字段: {field}")
                
                question_type = QuestionType(result['question_type'])
                confidence = float(result['confidence'])
                search_mode = SearchMode(result['search_mode'])
                
                logger.info(f"AI分析成功: 类型={question_type.value}, 置信度={confidence:.2f}, 模式={search_mode.value}")
                return question_type, confidence, search_mode
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"AI分析JSON解析失败: {e}")
                logger.error(f"清理后文本: '{result_text}'")
                return self._fallback_analysis(question)
                
        except Exception as e:
            logger.error(f"AI问题分析失败: {e}")
            return self._fallback_analysis(question)
    
    def _fallback_analysis(self, question: str) -> Tuple[QuestionType, float, SearchMode]:
        """备用分析方案"""
        question_type, confidence = self._identify_question_type(question)
        recommended_mode = self.config.get_auto_mode(question_type)
        return question_type, confidence, recommended_mode
    
    def _get_default_analysis(self, question: str) -> Dict[str, any]:
        """获取默认分析结果（降级方案）"""
        return {
            'question': question,
            'cleaned_question': question.lower().strip(),
            'question_type': QuestionType.GENERAL_SEARCH,
            'confidence': 0.3,
            'keywords': [question],
            'entities': {'brands': [], 'car_models': [], 'features': []},
            'recommended_mode': SearchMode.GRAPH,
            'search_params': self.config.SEARCH_PARAMS.copy()
        }
    
    def get_search_mode_description(self, mode: SearchMode) -> str:
        """获取搜索模式描述"""
        mode_info = self.config.get_search_mode_info(mode)
        return mode_info.get('description', '未知搜索模式')
    
    def validate_search_mode(self, mode_str: str) -> SearchMode:
        """验证并转换搜索模式字符串"""
        try:
            return SearchMode(mode_str.lower())
        except ValueError:
            logger.warning(f"无效的搜索模式: {mode_str}, 使用默认模式")
            return SearchMode.GRAPH

# 全局实例
question_analyzer = QuestionAnalyzer()