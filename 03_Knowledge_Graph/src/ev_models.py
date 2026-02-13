"""
新能源汽车知识图谱数据模型
基于用户画像和8维特征的EV推荐系统数据结构
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


@dataclass
class CarModel:
    """车型节点模型"""
    model_id: str                    # 车型唯一标识符
    name: str                        # 车型名称 (如: "小米SU7 Ultra")
    brand: str                       # 品牌名称 (如: "小米")
    type: str                        # 车型类型 (如: "纯电动轿车")
    price_range: str                 # 价格区间 (如: "50-60万")
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为Neo4j节点属性字典"""
        return {
            'modelId': self.model_id,
            'name': self.name,
            'brand': self.brand,
            'type': self.type,
            'priceRange': self.price_range,
            'createdAt': self.created_at.isoformat() if self.created_at else datetime.now().isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CarModel':
        """从字典创建实例"""
        created_at = None
        if data.get('createdAt'):
            created_at = datetime.fromisoformat(data['createdAt'])
        
        return cls(
            model_id=data['modelId'],
            name=data['name'],
            brand=data['brand'],
            type=data['type'],
            price_range=data['priceRange'],
            created_at=created_at
        )


@dataclass
class UserProfile:
    """用户画像节点模型 (30个聚类)"""
    profile_id: int                           # 画像ID (1-30)
    name: str                                 # 画像名称 (如: "性能追求者")
    description: str                          # 画像描述
    user_count: int                          # 该画像下的用户数量
    main_features: List[str]                 # 主要特征标签
    dimension_strengths: Dict[str, float]    # 8维特征的强度分布
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为Neo4j节点属性字典"""
        return {
            'profileId': self.profile_id,
            'name': self.name,
            'description': self.description,
            'userCount': self.user_count,
            'mainFeatures': self.main_features,
            'dimensionStrengths': self.dimension_strengths,
            'createdAt': self.created_at.isoformat() if self.created_at else datetime.now().isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """从字典创建实例"""
        created_at = None
        if data.get('createdAt'):
            created_at = datetime.fromisoformat(data['createdAt'])
        
        return cls(
            profile_id=data['profileId'],
            name=data['name'],
            description=data['description'],
            user_count=data['userCount'],
            main_features=data.get('mainFeatures', []),
            dimension_strengths=data.get('dimensionStrengths', {}),
            created_at=created_at
        )


@dataclass
class Review:
    """评论节点模型 (13,682条)"""
    review_id: str                    # 评论唯一标识符
    content: str                      # 原始评论内容
    user_id: str                      # 用户标识符
    car_model: str                    # 评论的车型
    overall_sentiment: float          # 整体情感倾向 (-1到1)
    dimension_data: Dict[str, Dict[str, float]]  # 8维度的情感和强度数据
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为Neo4j节点属性字典"""
        return {
            'reviewId': self.review_id,
            'content': self.content,
            'userId': self.user_id,
            'carModel': self.car_model,
            'overallSentiment': self.overall_sentiment,
            'dimensionData': json.dumps(self.dimension_data),
            'createdAt': self.created_at.isoformat() if self.created_at else datetime.now().isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Review':
        """从字典创建实例"""
        created_at = None
        if data.get('createdAt'):
            created_at = datetime.fromisoformat(data['createdAt'])
        
        dimension_data = {}
        if data.get('dimensionData'):
            if isinstance(data['dimensionData'], str):
                dimension_data = json.loads(data['dimensionData'])
            else:
                dimension_data = data['dimensionData']
        
        return cls(
            review_id=data['reviewId'],
            content=data['content'],
            user_id=data['userId'],
            car_model=data['carModel'],
            overall_sentiment=data['overallSentiment'],
            dimension_data=dimension_data,
            created_at=created_at
        )


@dataclass
class Feature:
    """特征维度节点模型 (8个固定维度)"""
    name: str                         # 特征名称
    category: str                     # 特征分类
    description: str                  # 特征详细说明
    keywords: List[str] = field(default_factory=list)  # 相关关键词
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为Neo4j节点属性字典"""
        return {
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'keywords': self.keywords
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Feature':
        """从字典创建实例"""
        return cls(
            name=data['name'],
            category=data['category'],
            description=data['description'],
            keywords=data.get('keywords', [])
        )


# 预定义的8个核心特征维度
PREDEFINED_FEATURES = [
    Feature(
        name="外观设计",
        category="视觉感知",
        description="车型外观造型、设计美学、视觉冲击力",
        keywords=["外观", "设计", "造型", "颜值", "美观", "好看", "漂亮", "时尚"]
    ),
    Feature(
        name="内饰质感",
        category="内部体验",
        description="车内装饰材质、做工品质、豪华感",
        keywords=["内饰", "质感", "材质", "做工", "豪华", "精致", "档次", "品质"]
    ),
    Feature(
        name="智能配置",
        category="科技功能",
        description="自动驾驶、智能车机、科技配置",
        keywords=["智能", "自动驾驶", "车机", "科技", "配置", "功能", "系统", "辅助"]
    ),
    Feature(
        name="空间实用",
        category="功能性",
        description="乘坐空间、储物能力、实用性",
        keywords=["空间", "实用", "储物", "乘坐", "座椅", "后排", "后备箱", "装载"]
    ),
    Feature(
        name="舒适体验",
        category="乘坐感受",
        description="座椅舒适性、噪音控制、悬挂调校",
        keywords=["舒适", "噪音", "悬挂", "减震", "座椅", "静音", "平稳", "柔软"]
    ),
    Feature(
        name="操控性能",
        category="驾驶体验",
        description="动力输出、加速性能、操控感受",
        keywords=["操控", "性能", "动力", "加速", "提速", "驾驶", "灵活", "响应"]
    ),
    Feature(
        name="续航能耗",
        category="续航能力",
        description="电池续航、充电速度、能耗表现",
        keywords=["续航", "能耗", "电池", "充电", "里程", "电量", "省电", "耐用"]
    ),
    Feature(
        name="价值认知",
        category="性价比",
        description="价格合理性、品牌价值、投资回报",
        keywords=["价格", "性价比", "值得", "便宜", "贵", "划算", "品牌", "保值"]
    )
]


class RelationshipTypes:
    """关系类型常量"""
    PUBLISHED = "PUBLISHED"           # 用户画像发布评论
    MENTIONS = "MENTIONS"             # 评论提及车型
    CONTAINS_ASPECT = "CONTAINS_ASPECT"  # 评论包含特征维度
    INTERESTED_IN = "INTERESTED_IN"   # 用户画像对车型的兴趣


class NodeLabels:
    """节点标签常量"""
    CAR_MODEL = "CarModel"
    USER_PROFILE = "UserProfile"
    REVIEW = "Review"
    FEATURE = "Feature"


# 维度名称映射 (CSV字段名 -> 标准名称)
DIMENSION_MAPPING = {
    "外观设计": "外观设计",
    "内饰质感": "内饰质感", 
    "智能配置": "智能配置",
    "空间实用": "空间实用",
    "舒适体验": "舒适体验",
    "操控性能": "操控性能",
    "续航能耗": "续航能耗",
    "价值认知": "价值认知"
}


def get_dimension_names() -> List[str]:
    """获取所有维度名称列表"""
    return list(DIMENSION_MAPPING.values())


def validate_sentiment_score(score: float) -> bool:
    """验证情感分数是否在有效范围内"""
    return -1.0 <= score <= 1.0


def validate_intensity_score(score: float) -> bool:
    """验证强度分数是否在有效范围内"""
    return 0.0 <= score <= 1.0


def normalize_car_model_name(raw_name: str) -> str:
    """标准化车型名称"""
    # 移除多余空格，统一格式
    name = raw_name.strip()
    
    # 可以在这里添加更多的标准化规则
    # 例如：品牌名称统一、型号格式统一等
    
    return name


def extract_brand_from_model(model_name: str) -> str:
    """从车型名称中提取品牌"""
    # 常见品牌列表
    brands = ["小米", "特斯拉", "比亚迪", "蔚来", "理想", "小鹏", "极氪", "岚图", "问界", "智界"]
    
    for brand in brands:
        if brand in model_name:
            return brand
    
    # 如果没有匹配到已知品牌，取第一个词作为品牌
    parts = model_name.split()
    return parts[0] if parts else "未知品牌"


def determine_car_type(model_name: str) -> str:
    """根据车型名称判断车型类型"""
    name_lower = model_name.lower()
    
    if any(keyword in name_lower for keyword in ['suv', 'x', 'u']):
        return "SUV"
    elif any(keyword in name_lower for keyword in ['sedan', 's', '轿车']):
        return "轿车"
    elif any(keyword in name_lower for keyword in ['mpv', 'm']):
        return "MPV"
    else:
        return "其他"


def estimate_price_range(model_name: str) -> str:
    """根据车型名称估算价格区间"""
    name_lower = model_name.lower()
    
    # 这里可以根据已知的车型信息进行价格估算
    # 目前使用简单的规则，实际应用中可以从数据库或API获取
    
    if any(keyword in name_lower for keyword in ['ultra', '旗舰', 'max']):
        return "50万以上"
    elif any(keyword in name_lower for keyword in ['pro', 'plus', '高配']):
        return "30-50万"
    elif any(keyword in name_lower for keyword in ['标准', 'base', '入门']):
        return "20-30万"
    else:
        return "20-40万"