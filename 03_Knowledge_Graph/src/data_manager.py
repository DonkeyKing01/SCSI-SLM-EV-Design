"""
数据管理器模块
处理CSV数据、车型信息和评论数据
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Iterator
import re
from pathlib import Path
import logging


class DataManager:
    """数据管理器"""
    
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        self.data: Optional[pd.DataFrame] = None
        self.car_models: Dict[str, Dict] = {}
        self.feature_dimensions = [
            "外观设计", "内饰质感", "智能配置", "空间实用", 
            "舒适体验", "操控性能", "续航能耗", "价值认知"
        ]
        
        self._load_data()
        self._extract_car_models()
    
    def _load_data(self):
        """加载CSV数据"""
        try:
            self.data = pd.read_csv(self.csv_file_path)
            logging.info(f"成功加载数据: {len(self.data)} 条记录")
        except Exception as e:
            logging.error(f"加载CSV文件失败: {e}")
            raise
    
    def _extract_car_models(self):
        """提取车型信息"""
        if self.data is None:
            return
        
        # 从car_model字段提取车型信息
        car_model_counts = self.data['car_model'].value_counts()
        
        for model_name, count in car_model_counts.items():
            if pd.isna(model_name) or model_name == '':
                continue
            
            # 解析车型信息
            model_info = self._parse_car_model(model_name)
            
            # 获取该车型的评论数据
            model_reviews = self.data[self.data['car_model'] == model_name]
            
            # 计算车型特征统计
            feature_stats = self._calculate_model_feature_stats(model_reviews)
            
            self.car_models[model_name] = {
                "modelId": model_name,
                "name": model_info.get("name", model_name),
                "brand": model_info.get("brand", "未知"),
                "type": model_info.get("type", "未知"),
                "priceRange": model_info.get("priceRange", "未知"),
                "reviewCount": int(count),
                "featureStats": feature_stats
            }
        
        logging.info(f"提取了 {len(self.car_models)} 个车型")
    
    def _parse_car_model(self, model_name: str) -> Dict[str, str]:
        """解析车型名称，提取品牌、类型等信息"""
        # 常见的品牌映射
        brand_patterns = {
            r'小米': '小米',
            r'特斯拉|Tesla': '特斯拉',
            r'比亚迪|BYD': '比亚迪',
            r'蔚来|NIO': '蔚来',
            r'理想|Li': '理想',
            r'小鹏|XPeng': '小鹏',
            r'问界|AITO': '问界',
            r'极氪|Zeekr': '极氪',
            r'岚图|VOYAH': '岚图',
            r'阿维塔|Avatr': '阿维塔',
            r'智己|IM': '智己',
            r'飞凡|R': '飞凡',
            r'深蓝|SL': '深蓝',
            r'零跑|Leapmotor': '零跑',
            r'哪吒|Neta': '哪吒',
            r'威马|WM': '威马',
            r'高合|HiPhi': '高合',
            r'天际|ENOVATE': '天际',
            r'爱驰|Aiways': '爱驰',
            r'云度|YUDO': '云度'
        }
        
        # 车型类型映射
        type_patterns = {
            r'SUV|suv': 'SUV',
            r'轿车|轿车型': '轿车',
            r'跑车|超跑': '跑车',
            r'MPV|mpv': 'MPV',
            r'皮卡': '皮卡',
            r'旅行车': '旅行车'
        }
        
        # 价格区间映射
        price_patterns = {
            r'Ultra|ultra|旗舰': '高端(50万+)',
            r'Pro|pro|专业': '中高端(30-50万)',
            r'Plus|plus|增强': '中端(20-30万)',
            r'标准|基础': '入门(15-20万)',
            r'入门|经济': '经济(10-15万)'
        }
        
        brand = "未知"
        car_type = "未知"
        price_range = "未知"
        
        # 识别品牌
        for pattern, brand_name in brand_patterns.items():
            if re.search(pattern, model_name):
                brand = brand_name
                break
        
        # 识别车型类型
        for pattern, type_name in type_patterns.items():
            if re.search(pattern, model_name):
                car_type = type_name
                break
        
        # 识别价格区间
        for pattern, price_name in price_patterns.items():
            if re.search(pattern, model_name):
                price_range = price_name
                break
        
        return {
            "name": model_name,
            "brand": brand,
            "type": car_type,
            "priceRange": price_range
        }
    
    def _calculate_model_feature_stats(self, model_reviews: pd.DataFrame) -> Dict[str, Dict]:
        """计算车型的特征统计"""
        stats = {}
        
        for dim in self.feature_dimensions:
            intensity_col = f"{dim}_强度"
            sentiment_col = f"{dim}_情感"
            
            if intensity_col in model_reviews.columns and sentiment_col in model_reviews.columns:
                # 强度统计
                intensities = model_reviews[intensity_col].dropna()
                if len(intensities) > 0:
                    stats[dim] = {
                        "avgIntensity": float(intensities.mean()),
                        "maxIntensity": float(intensities.max()),
                        "minIntensity": float(intensities.min()),
                        "stdIntensity": float(intensities.std())
                    }
                
                # 情感统计
                sentiments = model_reviews[sentiment_col].dropna()
                if len(sentiments) > 0:
                    sentiment_counts = sentiments.value_counts()
                    stats[dim]["sentimentDistribution"] = {
                        "positive": int(sentiment_counts.get("正面", 0)),
                        "negative": int(sentiment_counts.get("负面", 0)),
                        "neutral": int(sentiment_counts.get("中性", 0))
                    }
        
        return stats
    
    def get_car_models(self) -> List[Dict]:
        """获取所有车型信息"""
        return list(self.car_models.values())
    
    def get_car_model(self, model_name: str) -> Optional[Dict]:
        """获取指定车型信息"""
        return self.car_models.get(model_name)
    
    def get_reviews_for_model(self, model_name: str) -> pd.DataFrame:
        """获取指定车型的所有评论"""
        if self.data is None:
            return pd.DataFrame()
        
        return self.data[self.data['car_model'] == model_name]
    
    def get_reviews_batch(self, batch_size: int = 1000) -> Iterator[pd.DataFrame]:
        """分批获取评论数据"""
        if self.data is None:
            return
        
        total_reviews = len(self.data)
        for start_idx in range(0, total_reviews, batch_size):
            end_idx = min(start_idx + batch_size, total_reviews)
            yield self.data.iloc[start_idx:end_idx]
    
    def get_user_vector_from_review(self, review_row: pd.Series) -> Dict[str, float]:
        """从评论行提取用户向量"""
        user_vector = {}
        
        for dim in self.feature_dimensions:
            intensity_col = f"{dim}_强度"
            if intensity_col in review_row:
                value = review_row[intensity_col]
                if pd.notna(value) and value != '':
                    try:
                        user_vector[dim] = float(value)
                    except (ValueError, TypeError):
                        user_vector[dim] = 0.0
                else:
                    user_vector[dim] = 0.0
            else:
                user_vector[dim] = 0.0
        
        return user_vector
    
    def get_sentiment_scores_from_review(self, review_row: pd.Series) -> Dict[str, Dict]:
        """从评论行提取情感分数"""
        sentiment_scores = {}
        
        for dim in self.feature_dimensions:
            intensity_col = f"{dim}_强度"
            sentiment_col = f"{dim}_情感"
            keywords_col = f"{dim}_关键词"
            
            sentiment_data = {
                "intensity": 0.0,
                "sentiment": "neutral",
                "keywords": []
            }
            
            # 强度
            if intensity_col in review_row:
                value = review_row[intensity_col]
                if pd.notna(value) and value != '':
                    try:
                        sentiment_data["intensity"] = float(value)
                    except (ValueError, TypeError):
                        sentiment_data["intensity"] = 0.0
            
            # 情感
            if sentiment_col in review_row:
                value = review_row[sentiment_col]
                if pd.notna(value) and value != '':
                    sentiment_data["sentiment"] = str(value)
            
            # 关键词
            if keywords_col in review_row:
                value = review_row[keywords_col]
                if pd.notna(value) and value != '':
                    try:
                        # 尝试解析关键词列表
                        if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                            # 简单的字符串解析
                            keywords = value.strip('[]').replace('"', '').split(',')
                            sentiment_data["keywords"] = [k.strip() for k in keywords if k.strip()]
                        else:
                            sentiment_data["keywords"] = [str(value)]
                    except:
                        sentiment_data["keywords"] = [str(value)]
            
            sentiment_scores[dim] = sentiment_data
        
        return sentiment_scores
    
    def calculate_review_importance(self, review_row: pd.Series) -> float:
        """计算评论重要性分数"""
        # 基于评论长度和关键词密度
        content = str(review_row.get('original_comment', ''))
        content_length = len(content)
        
        # 关键词总数
        total_keywords = 0
        for dim in self.feature_dimensions:
            keywords_col = f"{dim}_关键词"
            if keywords_col in review_row:
                keywords = review_row[keywords_col]
                if pd.notna(keywords) and keywords != '':
                    try:
                        if isinstance(keywords, str) and keywords.startswith('['):
                            keywords_list = keywords.strip('[]').replace('"', '').split(',')
                            total_keywords += len([k for k in keywords_list if k.strip()])
                        else:
                            total_keywords += 1
                    except:
                        total_keywords += 1
        
        # 计算重要性分数 (0-1)
        length_score = min(content_length / 500.0, 1.0)  # 500字为满分
        keyword_score = min(total_keywords / 20.0, 1.0)  # 20个关键词为满分
        
        importance = (length_score * 0.6 + keyword_score * 0.4)
        return round(importance, 4)
    
    def get_data_summary(self) -> Dict:
        """获取数据摘要"""
        if self.data is None:
            return {}
        
        summary = {
            "totalReviews": len(self.data),
            "totalCarModels": len(self.car_models),
            "featureDimensions": self.feature_dimensions,
            "dataColumns": list(self.data.columns),
            "carModelDistribution": {},
            "featureCoverage": {}
        }
        
        # 车型分布
        for model_name, model_info in self.car_models.items():
            summary["carModelDistribution"][model_name] = {
                "reviewCount": model_info["reviewCount"],
                "brand": model_info["brand"],
                "type": model_info["type"]
            }
        
        # 特征覆盖
        for dim in self.feature_dimensions:
            intensity_col = f"{dim}_强度"
            if intensity_col in self.data.columns:
                non_zero_count = (self.data[intensity_col] > 0).sum()
                summary["featureCoverage"][dim] = {
                    "totalMentions": int(non_zero_count),
                    "coverageRate": round(non_zero_count / len(self.data), 4)
                }
        
        return summary


if __name__ == "__main__":
    # 测试数据管理器
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 使用相对路径
    csv_path = "../02_User_Modeling/User_Preference_Clustering/outputs/user_dimension_vectors.csv"
    
    try:
        data_manager = DataManager(csv_path)
        
        # 显示数据摘要
        summary = data_manager.get_data_summary()
        print(f"数据摘要:")
        print(f"总评论数: {summary['totalReviews']}")
        print(f"车型数: {summary['totalCarModels']}")
        
        # 显示前几个车型
        car_models = data_manager.get_car_models()
        print(f"\n前5个车型:")
        for i, model in enumerate(car_models[:5]):
            print(f"{i+1}. {model['name']} - {model['brand']} {model['type']} ({model['reviewCount']}条评论)")
        
        # 测试评论处理
        first_review = data_manager.data.iloc[0]
        user_vector = data_manager.get_user_vector_from_review(first_review)
        sentiment_scores = data_manager.get_sentiment_scores_from_review(first_review)
        importance = data_manager.calculate_review_importance(first_review)
        
        print(f"\n第一条评论分析:")
        print(f"用户向量: {user_vector}")
        print(f"重要性分数: {importance}")
        
    except Exception as e:
        print(f"测试失败: {e}") 