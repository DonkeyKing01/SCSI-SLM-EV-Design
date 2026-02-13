import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jieba
import re
import os
import json
import time
import sys
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 设置标准输出为UTF-8，避免Windows控制台编码问题
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# BERT相关导入
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import pipeline
    BERT_AVAILABLE = True
except ImportError:
    print(" 警告: transformers库未安装，将使用词典基础情感分析")
    BERT_AVAILABLE = False

# 设置中文字体 - 使用多个备选字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class FeatureSentimentAnalyzer:
    def __init__(self):
        """初始化特征情感分析器"""
        

        self.feature_dimensions = {
            '外观设计': {
                'description': '车辆外观造型、颜值、设计风格相关',
                'keywords': ['外观', '颜值', '造型', '设计', '前脸', '尾部', '车身', '外形', 
                           '大灯', '尾灯', '轮毂', '车漆', '线条', '风格', '时尚', '运动', '优雅', '霸气'],
                'weight': 0.125  # 8个维度均等权重
            },
            '内饰质感': {
                'description': '内饰材质、做工、豪华感、精致度相关',
                'keywords': ['内饰', '座椅', '方向盘', '中控', '仪表', '材质', '做工', '精致', 
                           '豪华', '质感', '皮质', '软硬', '包裹', '支撑', '档次'],
                'weight': 0.125
            },
            '智能配置': {
                'description': '智能科技、辅助驾驶、车机系统相关',
                'keywords': ['智能', '科技', '配置', '功能', '辅助', '导航', '音响', '车机', 
                           '语音', '自动', '雷达', '摄像头', '系统', '软件', '升级', '互联'],
                'weight': 0.125
            },
            '空间实用': {
                'description': '车内空间、储物、实用性相关',
                'keywords': ['空间', '座椅', '腿部', '头部', '后排', '前排', '乘坐', '储物', 
                           '后备箱', '装载', '实用', '宽敞', '紧凑', '够用'],
                'weight': 0.125
            },
            '舒适体验': {
                'description': '乘坐舒适性、静音性、减震效果相关',
                'keywords': ['舒适', '静音', '噪音', '减震', '悬挂', '滤震', '平顺', '稳定', 
                           '颠簸', '震动', '隔音', '风噪', '胎噪', '异响'],
                'weight': 0.125
            },
            '操控性能': {
                'description': '驾驶操控、动力性能、驾驶感受相关',
                'keywords': ['操控', '驾驶', '动力', '加速', '推背', '性能', '手感', '指向', 
                           '转向', '刹车', '油门', '制动', '精准', '轻松', '灵活', '响应'],
                'weight': 0.125
            },
            '续航能耗': {
                'description': '续航里程、能耗表现、充电相关',
                'keywords': ['续航', '能耗', '充电', '电池', '里程', '电量', '快充', '慢充', 
                           '省电', '耗电', '充电桩', '电费'],
                'weight': 0.125
            },
            '价值认知': {
                'description': '性价比、价格、经济性相关认知',
                'keywords': ['性价比', '价格', '便宜', '值得', '划算', '经济', '实惠', '贵', 
                           '超值', '物有所值', '成本', '保值'],
                'weight': 0.125
            }
        }
        
        # 情感词典（基础版本）
        self.positive_words = set(['好', '棒', '赞', '优秀', '满意', '喜欢', '不错', '完美', '舒服', '舒适', 
                                  '漂亮', '美观', '精致', '高级', '豪华', '实用', '方便', '快', '强', '稳'])
        self.negative_words = set(['差', '烂', '糟', '垃圾', '失望', '不满', '问题', '毛病', '缺点', '不好',
                                  '难看', '粗糙', '简陋', '不便', '慢', '弱', '不稳', '噪音', '异响'])
        
        # 初始化BERT模型
        self.bert_analyzer = None
        self.init_bert_model()
        
        # 创建输出目录
        self.output_dir = './outputs'
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f" 特征情感分析器初始化完成")
        print(f" 定义了 {len(self.feature_dimensions)} 个特征维度")
        print(f" BERT模型: {'已加载' if self.bert_analyzer else '使用基础词典'}")
        print(f" 输出目录: {self.output_dir}")
    
    def init_bert_model(self):
        """初始化BERT情感分析模型"""
        if not BERT_AVAILABLE:
            return
        
        try:
            print(" 正在加载BERT情感分析模型...")
            # 使用中文情感分析模型
            model_name = "uer/roberta-base-finetuned-dianping-chinese"
            self.bert_analyzer = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=-1  # CPU模式，如果有GPU可以设置为0
            )
            print(" BERT模型加载成功")
        except Exception as e:
            print(f" BERT模型加载失败，使用基础情感分析: {e}")
            self.bert_analyzer = None
    
    def clean_text(self, text: str) -> str:
        """清洗文本数据"""
        if pd.isna(text) or text == '':
            return ""
        
        text = str(text)
        # 移除URL  
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # 移除特殊字符，保留中文、英文、数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_feature_segments(self, comment: str) -> Dict[str, List[str]]:
        """从评论中提取各特征相关的文本片段"""
        feature_segments = defaultdict(list)
        
        # 分词
        words = jieba.lcut(comment)
        comment_lower = comment.lower()
        
        # 为每个特征维度查找相关片段
        for feature_name, feature_info in self.feature_dimensions.items():
            keywords = feature_info['keywords']
            
            # 查找包含关键词的句子片段
            for keyword in keywords:
                if keyword in comment_lower:
                    # 找到关键词前后的上下文
                    keyword_positions = []
                    start = 0
                    while True:
                        pos = comment_lower.find(keyword, start)
                        if pos == -1:
                            break
                        keyword_positions.append(pos)
                        start = pos + 1
                    
                    for pos in keyword_positions:
                        # 提取关键词前后各10个字符作为上下文
                        start_pos = max(0, pos - 10)
                        end_pos = min(len(comment), pos + len(keyword) + 10)
                        segment = comment[start_pos:end_pos].strip()
                        
                        if len(segment) >= 5:  # 至少5个字符的片段
                            feature_segments[feature_name].append(segment)
        
        return dict(feature_segments)
    
    def analyze_sentiment_bert(self, text: str) -> float:
        """使用BERT模型分析情感，返回0-1的分数"""
        if not self.bert_analyzer or not text.strip():
            return 0.5  # 中性分数
        
        try:
            # 限制文本长度，BERT有最大token限制
            if len(text) > 200:
                text = text[:200]
            
            result = self.bert_analyzer(text)[0]
            
            # 转换为0-1分数
            if result['label'].upper() == 'POSITIVE':
                score = 0.5 + (result['score'] * 0.5)  # 0.5-1.0
            else:
                score = 0.5 - (result['score'] * 0.5)  # 0.0-0.5
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"BERT分析错误: {e}")
            return self.analyze_sentiment_basic(text)
    
    def analyze_sentiment_basic(self, text: str) -> float:
        """基础词典情感分析，返回0-1的分数"""
        if not text.strip():
            return 0.5
        
        words = jieba.lcut(text.lower())
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.5  # 中性
        
        # 计算情感分数
        sentiment_ratio = positive_count / total_sentiment_words
        return sentiment_ratio
    
    def analyze_comment_features(self, comment: str) -> Dict[str, Dict]:
        """分析单条评论的各特征情感分数"""
        if not comment or pd.isna(comment):
            return {feature: {'score': 0.5, 'segments': [], 'has_mention': False} 
                   for feature in self.feature_dimensions.keys()}
        
        # 清洗评论
        cleaned_comment = self.clean_text(comment)
        
        # 提取特征相关片段
        feature_segments = self.extract_feature_segments(cleaned_comment)
        
        feature_scores = {}
        
        for feature_name in self.feature_dimensions.keys():
            segments = feature_segments.get(feature_name, [])
            
            if segments:
                # 对每个片段进行情感分析
                segment_scores = []
                for segment in segments:
                    if self.bert_analyzer:
                        score = self.analyze_sentiment_bert(segment)
                    else:
                        score = self.analyze_sentiment_basic(segment)
                    segment_scores.append(score)
                
                # 计算平均分数
                avg_score = np.mean(segment_scores) if segment_scores else 0.5
                
                feature_scores[feature_name] = {
                    'score': avg_score,
                    'segments': segments[:3],  # 保留前3个片段
                    'has_mention': True,
                    'segment_count': len(segments)
                }
            else:
                # 没有提及该特征，给予较低的默认分数（避免影响数据分布）
                feature_scores[feature_name] = {
                    'score': 0.1,  # 使用较低的默认值而不是0.5
                    'segments': [],
                    'has_mention': False,
                    'segment_count': 0
                }
        
        return feature_scores
    
    def load_comment_data(self, data_source: str) -> pd.DataFrame:
        """加载评论数据"""
        print(f"\n=== 步骤1：加载评论数据 ===")
        
        try:
            # 尝试从多个可能的数据源加载
            possible_sources = [
                data_source,
                "../../01_SSE_Analysis/1_Data_Preprocessing/outputs/cleaned_comments.csv",
                "./outputs",  
            ]
            possible_sources = [s for s in possible_sources if isinstance(s, str) and len(s) > 0]
            
            df = None
            for source in possible_sources:
                if os.path.exists(source):
                    if os.path.isfile(source) and source.endswith('.csv'):
                        df = pd.read_csv(source, encoding='utf-8')
                        print(f"从文件加载数据: {source}")
                        break
                    elif os.path.isdir(source):
                        csv_files = [f for f in os.listdir(source) if f.endswith('.csv')]
                        if csv_files:
                            all_dfs = []
                            for csv_file in csv_files[:5]:  
                                file_path = os.path.join(source, csv_file)
                                temp_df = pd.read_csv(file_path, encoding='utf-8')
                                temp_df['source_file'] = csv_file
                                all_dfs.append(temp_df)
                            df = pd.concat(all_dfs, ignore_index=True)
                            print(f"从目录加载数据: {source} ({len(csv_files)} 文件)")
                            break
            
            if df is None or df.empty:
                print(f"无法加载数据，请检查数据源")
                return pd.DataFrame()
            
            print(f"原始数据量: {len(df)} 条")
            
            # 数据清洗和标准化
            if 'cleaned_content' in df.columns:
                df['comment'] = df['cleaned_content']
            elif 'cleaned_comment' in df.columns:
                df['comment'] = df['cleaned_comment']
            elif 'original_comment' in df.columns:
                df['comment'] = df['original_comment']
            elif '评价内容' in df.columns:
                df['comment'] = df['评价内容']
            else:
                print("未找到评论内容列")
                return pd.DataFrame()
            
            # 车型信息标准化
            if 'car_model' not in df.columns:
                if 'source_file' in df.columns:
                    df['car_model'] = df['source_file'].str.replace('.csv', '')
                else:
                    df['car_model'] = 'Unknown'
            
            # 过滤有效评论
            df = df[df['comment'].str.len() >= 10].copy()
            df = df.dropna(subset=['comment'])
            
            print(f"清洗后数据量: {len(df)} 条有效评论")
            print(f"涉及车型: {df['car_model'].nunique()} 个")
            
            return df
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return pd.DataFrame()
    
    def batch_analyze_features(self, df: pd.DataFrame, batch_size: int = 100) -> pd.DataFrame:
        """批量分析评论的特征情感分数"""
        print(f"\n=== 步骤2：批量特征情感分析 ===")
        
        results = []
        total_comments = len(df)
        
        for i in range(0, total_comments, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_comments + batch_size - 1) // batch_size
            
            print(f"正在处理第 {batch_num}/{total_batches} 批 ({len(batch_df)} 条评论)...")
            
            batch_results = []
            for idx, row in batch_df.iterrows():
                comment = row['comment']
                
                # 分析特征情感
                feature_analysis = self.analyze_comment_features(comment)
                
                # 构建结果记录
                result = {
                    'comment_id': idx,
                    'original_comment': comment,
                    'car_model': row['car_model'],
                }
                
                # 添加各特征的分数和相关信息
                for feature_name, analysis in feature_analysis.items():
                    result[f'{feature_name}_score'] = analysis['score']
                    result[f'{feature_name}_has_mention'] = analysis['has_mention']
                    result[f'{feature_name}_segments'] = json.dumps(analysis['segments'], ensure_ascii=False)
                
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # 进度显示
            processed = min(i + batch_size, total_comments)
            print(f"  进度: {processed}/{total_comments} ({processed/total_comments*100:.1f}%)")
            
            # 避免过快调用BERT
            if self.bert_analyzer:
                time.sleep(0.1)
        
        # 创建结果DataFrame
        result_df = pd.DataFrame(results)
        
        print(f"特征情感分析完成")
        print(f"分析了 {len(result_df)} 条评论")
        
        # 统计各特征的提及情况
        print(f"\n特征提及统计：")
        for feature_name in self.feature_dimensions.keys():
            mention_col = f'{feature_name}_has_mention'
            if mention_col in result_df.columns:
                mention_count = result_df[mention_col].sum()
                mention_rate = mention_count / len(result_df) * 100
                avg_score = result_df[f'{feature_name}_score'].mean()
                print(f"  {feature_name}: 提及率 {mention_rate:.1f}% ({mention_count}条), 平均分 {avg_score:.3f}")
        
        return result_df
    
    def calculate_importance_weights(self, result_df: pd.DataFrame) -> Dict[str, float]:
        """基于用户提及频率计算特征重要度权重"""
        print(f"\n=== 步骤3：计算特征重要度权重 ===")
        
        importance_weights = {}
        mention_counts = {}
        
        # 统计各特征的提及次数
        for feature_name in self.feature_dimensions.keys():
            mention_col = f'{feature_name}_has_mention'
            if mention_col in result_df.columns:
                mention_count = result_df[mention_col].sum()
                mention_counts[feature_name] = mention_count
        
        total_mentions = sum(mention_counts.values())
        
        if total_mentions == 0:
            # 如果没有提及数据，使用均等权重
            for feature_name in self.feature_dimensions.keys():
                importance_weights[feature_name] = 1.0 / len(self.feature_dimensions)
        else:
            # 基于提及频率计算权重，但避免权重过于极端
            min_weight = 0.05  # 最小权重5%
            max_weight = 0.25  # 最大权重25%
            
            raw_weights = {}
            for feature_name, count in mention_counts.items():
                raw_weights[feature_name] = count / total_mentions
            
            # 调整权重范围
            adjusted_weights = {}
            for feature_name, weight in raw_weights.items():
                adjusted_weight = max(min_weight, min(max_weight, weight))
                adjusted_weights[feature_name] = adjusted_weight
            
            # 重新归一化
            total_adjusted = sum(adjusted_weights.values())
            for feature_name in adjusted_weights:
                importance_weights[feature_name] = adjusted_weights[feature_name] / total_adjusted
        
        print(f"特征重要度权重计算完成")
        print(f"权重分布：")
        for feature_name, weight in sorted(importance_weights.items(), key=lambda x: x[1], reverse=True):
            mention_count = mention_counts.get(feature_name, 0)
            print(f"  {feature_name}: {weight:.3f} (提及{mention_count}次)")
        
        return importance_weights
    
    def aggregate_car_model_scores(self, result_df: pd.DataFrame, importance_weights: Dict[str, float]) -> pd.DataFrame:
        """聚合车型级别的特征分数"""
        print(f"\n=== 步骤4：聚合车型特征分数 ===")
        
        car_scores = []
        
        for car_model in result_df['car_model'].unique():
            if pd.isna(car_model):
                continue
                
            car_data = result_df[result_df['car_model'] == car_model]
            
            car_score_record = {
                'car_model': car_model,
                'review_count': len(car_data)
            }
            
            # 计算各特征的平均分数和重要度
            for feature_name in self.feature_dimensions.keys():
                score_col = f'{feature_name}_score'
                mention_col = f'{feature_name}_has_mention'
                
                if score_col in car_data.columns:
                    # 只计算有提及的评论的平均分
                    mentioned_data = car_data[car_data[mention_col] == True]
                    
                    if len(mentioned_data) > 0:
                        avg_score = mentioned_data[score_col].mean()
                        mention_rate = len(mentioned_data) / len(car_data)
                    else:
                        avg_score = 0.5  # 没有提及时给中性分
                        mention_rate = 0.0
                    
                    car_score_record[f'{feature_name}_performance'] = avg_score
                    car_score_record[f'{feature_name}_importance'] = importance_weights.get(feature_name, 0.125)
                    car_score_record[f'{feature_name}_mention_rate'] = mention_rate
            
            car_scores.append(car_score_record)
        
        car_scores_df = pd.DataFrame(car_scores)
        
        print(f"车型特征分数聚合完成")
        print(f"处理了 {len(car_scores_df)} 个车型")
        
        # 显示部分统计信息
        if len(car_scores_df) > 0:
            print(f"\n车型数据统计：")
            print(f"  平均每车型评论数: {car_scores_df['review_count'].mean():.1f}")
            print(f"  评论数最多的车型: {car_scores_df.loc[car_scores_df['review_count'].idxmax(), 'car_model']}")
            print(f"  评论数最少的车型: {car_scores_df.loc[car_scores_df['review_count'].idxmin(), 'car_model']}")
        
        return car_scores_df
    
    def generate_ipa_analysis(self, car_scores_df: pd.DataFrame):
        """生成IPA分析图表"""
        print(f"\n=== 步骤5：生成IPA分析 ===")
        
        # 为每个车型生成IPA图
        for _, car_row in car_scores_df.iterrows():
            car_model = car_row['car_model']
            self.plot_ipa_for_car(car_model, car_row)
        
        # 生成综合IPA报告
        self.generate_ipa_report(car_scores_df)
        
        print(f"IPA分析完成，图片和报告已保存到 {self.output_dir}")
    
    def plot_ipa_for_car(self, car_model: str, car_data: pd.Series):
        """为单个车型绘制IPA图"""
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # 提取重要度和绩效数据
        importance_data = []
        performance_data = []
        feature_names = []
        
        for feature_name in self.feature_dimensions.keys():
            importance = car_data[f'{feature_name}_importance']
            performance = car_data[f'{feature_name}_performance']
            
            importance_data.append(importance)
            performance_data.append(performance)
            feature_names.append(feature_name)
        
        # 计算中位数作为象限分割线
        importance_median = np.median(importance_data)
        performance_median = np.median(performance_data)
        
        # 绘制散点图
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#6C5CE7', '#A29BFE']
        
        for i, (imp, perf, feature) in enumerate(zip(importance_data, performance_data, feature_names)):
            ax.scatter(perf, imp, s=300, c=colors[i], alpha=0.7, label=feature, edgecolors='white', linewidth=2)
            ax.annotate(feature, (perf, imp), xytext=(8, 8), textcoords='offset points', 
                       fontsize=10, fontweight='bold')
        
        # 绘制象限分割线
        ax.axhline(y=importance_median, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
        ax.axvline(x=performance_median, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
        
        # 计算象限标签位置
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # 添加象限标签
        ax.text(x_min + (performance_median - x_min) * 0.5, 
               importance_median + (y_max - importance_median) * 0.5, 
               'Q2\n集中改进区\n(高重要度,低绩效)', ha='center', va='center', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFE5E5', alpha=0.8))
        
        ax.text(performance_median + (x_max - performance_median) * 0.5, 
               importance_median + (y_max - importance_median) * 0.5, 
               'Q1\n优势保持区\n(高重要度,高绩效)', ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='#E5F6E5', alpha=0.8))
        
        ax.text(x_min + (performance_median - x_min) * 0.5, 
               y_min + (importance_median - y_min) * 0.5, 
               'Q3\n低优先级区\n(低重要度,低绩效)', ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='#F0F0F0', alpha=0.8))
        
        ax.text(performance_median + (x_max - performance_median) * 0.5, 
               y_min + (importance_median - y_min) * 0.5, 
               'Q4\n过度投入区\n(低重要度,高绩效)', ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='#E5E5FF', alpha=0.8))
        
        # 设置图表属性
        ax.set_xlabel('绩效分数 (Performance Score)', fontsize=12, fontweight='bold')
        ax.set_ylabel('重要度 (Importance Weight)', fontsize=12, fontweight='bold')
        ax.set_title(f'{car_model} - IPA分析\n(基于BERT情感分析)', fontsize=14, fontweight='bold')
        
        # 设置坐标轴范围 - 基于数据自动调整
        x_margin = (max(performance_data) - min(performance_data)) * 0.1
        y_margin = (max(importance_data) - min(importance_data)) * 0.1
        
        ax.set_xlim(min(performance_data) - x_margin, max(performance_data) + x_margin)
        ax.set_ylim(min(importance_data) - y_margin, max(importance_data) + y_margin)
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # 添加图例
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{car_model}_IPA.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   已生成 {car_model} IPA图")
    
    def generate_ipa_report(self, car_scores_df: pd.DataFrame):
        """生成IPA分析报告"""
        
        report_lines = [
            "# 新版车型特征IPA分析报告",
            "",
            "## 分析概述",
            "",
            "本报告基于BERT模型对用户评论进行情感分析，采用IPA（重要度-绩效分析）方法，",
            f"对 {len(car_scores_df)} 个车型在 {len(self.feature_dimensions)} 个核心特征维度上的表现进行分析。",
            "",
            "### 分析维度",
            "",
        ]
        
        for i, (feature_name, feature_info) in enumerate(self.feature_dimensions.items(), 1):
            report_lines.append(f"{i}. **{feature_name}**: {feature_info['description']}")
        
        report_lines.extend([
            "",
            "### 方法说明",
            "",
            "- **情感分析**: 使用BERT模型对评论进行情感分析，生成0-1的情感分数",
            "- **重要度计算**: 基于用户评论中各特征的提及频率计算",
            "- **绩效分数**: 该特征相关评论的平均情感分数",
            "- **IPA象限**: 根据重要度和绩效的中位数划分四个象限",
            "",
            "### 四象限说明",
            "",
            "- **Q1 优势保持区**（高重要度，高绩效）：继续保持优势",
            "- **Q2 集中改进区**（高重要度，低绩效）：重点投入资源改进", 
            "- **Q3 低优先级区**（低重要度，低绩效）：维持现状",
            "- **Q4 过度投入区**（低重要度，高绩效）：可适当减少投入",
            "",
            "## 车型分析结果",
            ""
        ])
        
        # 按评论数量排序，分析主要车型
        top_cars = car_scores_df.nlargest(15, 'review_count')
        
        for _, car_row in top_cars.iterrows():
            car_model = car_row['car_model']
            review_count = car_row['review_count']
            
            report_lines.extend([
                f"### {car_model}",
                "",
                f"**评论数量**: {review_count} 条",
                "",
                "**各维度表现**:",
                ""
            ])
            
            # 收集该车型的特征表现
            feature_performance = []
            for feature_name in self.feature_dimensions.keys():
                performance = car_row[f'{feature_name}_performance']
                importance = car_row[f'{feature_name}_importance']
                mention_rate = car_row[f'{feature_name}_mention_rate']
                
                feature_performance.append({
                    'feature': feature_name,
                    'performance': performance,
                    'importance': importance,
                    'mention_rate': mention_rate
                })
            
            # 按绩效分数排序
            feature_performance.sort(key=lambda x: x['performance'], reverse=True)
            
            for fp in feature_performance:
                report_lines.append(f"- **{fp['feature']}**: 绩效 {fp['performance']:.3f}, "
                                  f"重要度 {fp['importance']:.3f}, 提及率 {fp['mention_rate']:.1%}")
            
            # 分析象限分布
            q1_features = []
            q2_features = []
            for fp in feature_performance:
                if fp['importance'] > np.median([f['importance'] for f in feature_performance]) and \
                   fp['performance'] > np.median([f['performance'] for f in feature_performance]):
                    q1_features.append(fp['feature'])
                elif fp['importance'] > np.median([f['importance'] for f in feature_performance]) and \
                     fp['performance'] <= np.median([f['performance'] for f in feature_performance]):
                    q2_features.append(fp['feature'])
            
            if q1_features:
                report_lines.append(f"\n**优势特征**: {', '.join(q1_features)}")
            if q2_features:
                report_lines.append(f"**需改进特征**: {', '.join(q2_features)}")
            
            report_lines.append("")
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'IPA_Analysis_Report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"   已生成IPA分析报告: {report_path}")
    
    def save_results(self, result_df: pd.DataFrame, car_scores_df: pd.DataFrame, importance_weights: Dict[str, float]):
        """保存所有分析结果"""
        print(f"\n=== 步骤6：保存分析结果 ===")
        
        # 1. 保存评论级别的分析结果
        comments_path = os.path.join(self.output_dir, 'comment_feature_scores.csv')
        result_df.to_csv(comments_path, index=False, encoding='utf-8-sig')
        print(f" 评论特征分数已保存: {comments_path}")
        
        # 2. 保存车型级别的聚合结果
        cars_path = os.path.join(self.output_dir, 'car_model_scores.csv')
        car_scores_df.to_csv(cars_path, index=False, encoding='utf-8-sig')
        print(f" 车型特征分数已保存: {cars_path}")
        
        # 3. 保存重要度权重
        weights_path = os.path.join(self.output_dir, 'feature_importance_weights.json')
        with open(weights_path, 'w', encoding='utf-8') as f:
            json.dump(importance_weights, f, ensure_ascii=False, indent=2)
        print(f" 特征重要度权重已保存: {weights_path}")
        
        # 4. 生成维度统计报告
        stats_report = self.generate_feature_statistics(result_df, car_scores_df, importance_weights)
        stats_path = os.path.join(self.output_dir, 'feature_statistics.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_report, f, ensure_ascii=False, indent=2)
        print(f" 特征统计报告已保存: {stats_path}")
        
        print(f"\n 所有结果已保存到: {self.output_dir}")
    
    def generate_feature_statistics(self, result_df: pd.DataFrame, car_scores_df: pd.DataFrame, 
                                   importance_weights: Dict[str, float]) -> Dict:
        """生成特征统计报告"""
        
        stats = {
            'analysis_summary': {
                'total_comments': len(result_df),
                'total_cars': len(car_scores_df),
                'feature_dimensions': len(self.feature_dimensions),
                'bert_enabled': self.bert_analyzer is not None
            },
            'feature_statistics': {},
            'car_rankings': {},
            'dimension_correlations': {}
        }
        
        # 特征维度统计
        for feature_name in self.feature_dimensions.keys():
            score_col = f'{feature_name}_score'
            mention_col = f'{feature_name}_has_mention'
            
            if score_col in result_df.columns:
                scores = result_df[score_col]
                mentions = result_df[mention_col].sum()
                
                stats['feature_statistics'][feature_name] = {
                    'importance_weight': importance_weights.get(feature_name, 0.125),
                    'mention_count': int(mentions),
                    'mention_rate': float(mentions / len(result_df)),
                    'avg_score': float(scores.mean()),
                    'score_std': float(scores.std()),
                    'score_min': float(scores.min()),
                    'score_max': float(scores.max())
                }
        
        # 车型排名（按各维度）
        for feature_name in self.feature_dimensions.keys():
            perf_col = f'{feature_name}_performance'
            if perf_col in car_scores_df.columns:
                top_cars = car_scores_df.nlargest(5, perf_col)
                stats['car_rankings'][feature_name] = [
                    {
                        'car_model': row['car_model'],
                        'score': float(row[perf_col]),
                        'review_count': int(row['review_count'])
                    }
                    for _, row in top_cars.iterrows()
                ]
        
        return stats
    
    def run_complete_analysis(self, data_source: str = None, sample_size: int = None) -> bool:
        """运行完整的特征情感分析流程"""
        print("=" * 80)
        print(" 新版车型特征分析系统 - 基于BERT的情感评分")
        print("=" * 80)
        
        try:
            # 1. 加载数据
            if data_source is None:
                data_source = "../../01_SSE_Analysis/1_Data_Preprocessing/outputs/cleaned_comments.csv"
            
            df = self.load_comment_data(data_source)
            if df.empty:
                print(" 数据加载失败，流程终止")
                return False
            
            # 2. 数据采样（如果需要）
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                print(f" 随机采样 {sample_size} 条评论进行分析")
            
            # 3. 批量特征情感分析
            result_df = self.batch_analyze_features(df, batch_size=50)
            
            if result_df.empty:
                print(" 特征情感分析失败，流程终止")
                return False
            
            # 4. 计算特征重要度权重
            importance_weights = self.calculate_importance_weights(result_df)
            
            # 5. 聚合车型特征分数
            car_scores_df = self.aggregate_car_model_scores(result_df, importance_weights)
            
            # 6. 生成IPA分析
            self.generate_ipa_analysis(car_scores_df)
            
            # 7. 保存所有结果
            self.save_results(result_df, car_scores_df, importance_weights)
            
            print("\n" + "=" * 80)
            print(" 新版特征分析流程完成！")
            print(f" 所有结果已保存到: {self.output_dir}")
            print(" 核心产出:")
            print(f"   - 评论特征分数: {len(result_df)} 条评论")
            print(f"   - 车型IPA分析: {len(car_scores_df)} 个车型")
            print(f"   - 特征维度: {len(self.feature_dimensions)} 个维度")
            print(f"   - BERT情感分析: {'已启用' if self.bert_analyzer else '使用基础词典'}")
            print("=" * 80)
            
            return True
            
        except Exception as e:
            print(f"❌ 流程执行失败: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    print(" 车型特征分析系统 - 基于BERT的情感评分")
    print("=" * 60)
    
    # 创建分析器
    analyzer = FeatureSentimentAnalyzer()
    
    # 运行完整分析流程
    success = analyzer.run_complete_analysis(
        data_source=None,  # 使用默认数据源
        sample_size=None   # 不采样，分析全部数据
    )
    
    if success:
        print("\n 特征分析成功完成！")
        print(" 下一步可以:")
        print("   1. 查看生成的IPA分析图表")
        print("   2. 与Persona_New的用户向量进行匹配分析")
        print("   3. 为Neo4j图谱构建提供特征数据")
    else:
        print("\n 特征分析失败，请检查错误信息")

if __name__ == "__main__":
    main()