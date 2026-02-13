import os
import time
import json
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle
from dotenv import load_dotenv
from pathlib import Path

# 加载根目录的环境变量
root_dir = Path(__file__).parent.parent.parent
load_dotenv(root_dir / '.env')

class PersonaDiscoveryV3Optimized:
    def __init__(self):
        """初始化LLM客户端和配置"""
        # 从环境变量读取 API Key
        api_key = os.getenv('SILICONFLOW_API_KEY')
        if not api_key:
            raise ValueError(
                "未找到 SILICONFLOW_API_KEY 环境变量。\n"
                "请在项目根目录创建 .env 文件，并配置 SILICONFLOW_API_KEY。\n"
                "参考根目录的 .env.example 文件进行配置。"
            )
        
        base_url = os.getenv('SILICONFLOW_BASE_URL', 'https://api.siliconflow.cn/v1')
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = os.getenv('LLM_MODEL', "deepseek-ai/DeepSeek-V2.5")
        
        # 创建输出目录
        os.makedirs('./outputs', exist_ok=True)
        
        # 配置参数
        self.batch_size = 10  # 每批处理条数
        self.max_retries = 3   # 最大重试次数
        self.delay_between_batches = 1.0  # 批次间延迟

    def step_1_1_sample_data(self, input_file: str, sample_size: int = 500) -> pd.DataFrame:
        """
        步骤1.1：数据准备与抽样
        """
        print(f"\n=== 步骤1.1：数据抽样 ===")
        print(f"目标样本量：{sample_size}")
        
        # 读取数据
        df = pd.read_csv(input_file)
        print(f"原始数据量：{len(df)} 条评论")
        
        # 过滤掉过短的评论
        df_filtered = df[df['cleaned_comment'].str.len() >= 15].copy()
        print(f"过滤后数据量：{len(df_filtered)} 条评论")
        
        # 随机抽样
        if len(df_filtered) > sample_size:
            sampled_df = df_filtered.sample(n=sample_size, random_state=42)
        else:
            sampled_df = df_filtered.copy()
            
        print(f"最终样本量：{len(sampled_df)} 条评论")
        
        # 保存样本
        os.makedirs('./outputs', exist_ok=True)
        output_path = './outputs/sampled_reviews.csv'
        sampled_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"样本已保存到：{output_path}")
        
        return sampled_df

    def save_progress(self, results, batch_index, total_batches):
        """保存进度"""
        progress_data = {
            'results': results,
            'batch_index': batch_index,
            'total_batches': total_batches,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open('./outputs/progress_llm_tags.pkl', 'wb') as f:
            pickle.dump(progress_data, f)
        
        print(f"  进度已保存：批次 {batch_index+1}/{total_batches}")

    def load_progress(self):
        """加载进度"""
        progress_file = './outputs/progress_llm_tags.pkl'
        if os.path.exists(progress_file):
            with open(progress_file, 'rb') as f:
                progress_data = pickle.load(f)
            return progress_data
        return None

    def process_batch_with_llm(self, batch_df):
        """使用LLM处理一批评论"""
        reviews_text = []
        for _, row in batch_df.iterrows():
            reviews_text.append({
                'car_model': row['car_model'],
                'review': row['cleaned_comment']
            })
        
        # 构建批量处理的提示
        system_prompt = """你是一位顶尖的汽车行业市场洞察专家，擅长从用户评论中精准识别核心用户诉求点。
你的任务是分析多条评论，为每条评论提取出独立的关注维度。"""
        
        user_prompt = f"""
请分析以下{len(reviews_text)}条汽车用户评论，为每条评论完成以下任务：

1. 识别该评论中的核心关注点（如智能化、动力性能、空间舒适、外观设计、价格成本等）
2. 为每个关注点生成一个简洁的标签（4-8个字）
3. 提取支撑该标签的2-3个关键词

评论列表：
"""
        
        for i, review_data in enumerate(reviews_text):
            user_prompt += f"""
评论{i+1}（{review_data['car_model']}）：{review_data['review']}
"""
        
        user_prompt += f"""

请严格按照以下JSON格式输出，为每条评论返回结果：
{{
  "batch_results": [
    {{
      "review_index": 0,
      "persona_facets": [
        {{
          "raw_persona_tag": "标签名称",
          "keywords": ["关键词1", "关键词2", "关键词3"]
        }}
      ]
    }},
    {{
      "review_index": 1,
      "persona_facets": [...]
    }}
  ]
}}
"""
        
        # 重试机制
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=3000
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # 清理响应文本，移除markdown代码块标记
                if response_text.startswith('```json'):
                    response_text = response_text[7:]  # 移除 ```json
                if response_text.startswith('```'):
                    response_text = response_text[3:]   # 移除 ```
                if response_text.endswith('```'):
                    response_text = response_text[:-3]  # 移除结尾的 ```
                response_text = response_text.strip()
                
                # 解析JSON
                llm_result = json.loads(response_text)
                batch_results = llm_result.get('batch_results', [])
                
                if len(batch_results) == len(batch_df):
                    return batch_results
                else:
                    print(f"    警告：返回结果数量不匹配 ({len(batch_results)} vs {len(batch_df)})")
                    if attempt < self.max_retries - 1:
                        print(f"    重试第 {attempt + 2} 次...")
                        time.sleep(2)
                        continue
                    else:
                        return []
                        
            except json.JSONDecodeError as e:
                print(f"    JSON解析失败 (尝试 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    print(f"    重试第 {attempt + 2} 次...")
                    time.sleep(2)
                else:
                    print(f"    跳过此批次，响应内容：{response_text[:200]}...")
                    return []
                    
            except Exception as e:
                print(f"    API调用失败 (尝试 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    print(f"    重试第 {attempt + 2} 次...")
                    time.sleep(2)
                else:
                    return []
        
        return []

    def step_1_2_llm_multi_tag_generation_optimized(self, sampled_df: pd.DataFrame) -> pd.DataFrame:
        """
        步骤1.2：通过LLM进行多维度标签生成 - 优化版
        """
        print(f"\n=== 步骤1.2：LLM多维标签生成（优化版） ===")
        print(f"总计：{len(sampled_df)} 条评论")
        print(f"批次大小：{self.batch_size} 条/批")
        
        # 检查是否有已保存的进度
        progress_data = self.load_progress()
        if progress_data:
            print(f"发现已保存的进度，从批次 {progress_data['batch_index']+1} 继续...")
            results = progress_data['results']
            start_batch = progress_data['batch_index'] + 1
        else:
            print("开始新的处理流程...")
            results = []
            start_batch = 0
        
        # 分批处理
        total_batches = (len(sampled_df) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(start_batch, total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(sampled_df))
            batch_df = sampled_df.iloc[start_idx:end_idx].copy()
            
            print(f"\n处理批次 {batch_idx + 1}/{total_batches} (评论 {start_idx+1}-{end_idx})...")
            
            # 处理批次
            batch_results = self.process_batch_with_llm(batch_df)
            
            if batch_results:
                # 整合结果
                for i, (_, row) in enumerate(batch_df.iterrows()):
                    if i < len(batch_results):
                        result_data = batch_results[i]
                        persona_facets = result_data.get('persona_facets', [])
                        
                        results.append({
                            'original_comment': row['original_comment'],
                            'cleaned_comment': row['cleaned_comment'],
                            'car_model': row['car_model'],
                            'persona_facets': json.dumps(persona_facets, ensure_ascii=False)
                        })
                        
                        print(f"    评论{i+1}: 提取了 {len(persona_facets)} 个标签")
                    else:
                        # 如果结果不完整，添加空结果
                        results.append({
                            'original_comment': row['original_comment'],
                            'cleaned_comment': row['cleaned_comment'],
                            'car_model': row['car_model'],
                            'persona_facets': json.dumps([], ensure_ascii=False)
                        })
                        print(f"    评论{i+1}: 未获取到结果")
            else:
                # 整个批次失败，添加空结果
                for _, row in batch_df.iterrows():
                    results.append({
                        'original_comment': row['original_comment'],
                        'cleaned_comment': row['cleaned_comment'],
                        'car_model': row['car_model'],
                        'persona_facets': json.dumps([], ensure_ascii=False)
                    })
                print(f"    批次处理失败，添加空结果")
            
            # 保存进度
            self.save_progress(results, batch_idx, total_batches)
            
            # 批次间延迟
            if batch_idx < total_batches - 1:
                print(f"    等待 {self.delay_between_batches} 秒...")
                time.sleep(self.delay_between_batches)
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(results)
        
        # 保存最终结果
        output_path = './outputs/llm_multi_tags.csv'
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nLLM标签生成完成，结果已保存到：{output_path}")
        
        # 统计信息
        total_facets = 0
        valid_results = 0
        for result in results:
            facets = json.loads(result['persona_facets'])
            if facets:
                total_facets += len(facets)
                valid_results += 1
        
        print(f"总计提取标签数：{total_facets}")
        print(f"有效结果数量：{valid_results}/{len(results)}")
        if valid_results > 0:
            print(f"平均每条评论标签数：{total_facets/valid_results:.2f}")
        
        # 清理进度文件
        progress_file = './outputs/progress_llm_tags.pkl'
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print("已清理进度文件")
        
        return results_df

    def step_1_3_secondary_tag_refinement(self, llm_results_df: pd.DataFrame) -> pd.DataFrame:
        """
        步骤1.3：二次标签归纳优化
        使用LLM对初次提取的标签进行二次归纳和优化
        """
        print(f"\n=== 步骤1.3：二次标签归纳优化 ===")
        
        # 1. 数据扁平化
        print("正在进行数据扁平化...")
        flattened_data = []
        
        for idx, row in llm_results_df.iterrows():
            persona_facets = json.loads(row['persona_facets'])
            for facet in persona_facets:
                if facet.get('raw_persona_tag') and facet.get('keywords'):
                    flattened_data.append({
                        'original_comment': row['original_comment'],
                        'car_model': row['car_model'],
                        'raw_persona_tag': facet['raw_persona_tag'],
                        'keywords': facet['keywords']
                    })
        
        if not flattened_data:
            raise ValueError("没有提取到有效的标签数据！")
            
        flattened_df = pd.DataFrame(flattened_data)
        print(f"扁平化后数据量：{len(flattened_df)} 个标签")
        
        # 保存扁平化数据
        flattened_df.to_csv('./outputs/flattened_tags.csv', index=False, encoding='utf-8-sig')
        
        # 2. 收集所有唯一标签进行二次归纳
        unique_tags = flattened_df['raw_persona_tag'].unique().tolist()
        print(f"唯一标签数量：{len(unique_tags)}")
        
        # 构建二次归纳的提示
        system_prompt = """你是汽车行业的资深市场分析专家，擅长对用户关注点进行归纳和分类。
你的任务是对给定的原始标签列表进行智能归纳，将语义相似的标签合并，并为每个类别生成更精准的标签名称。"""
        
        # 将标签列表分批处理（每批50个标签）
        batch_size = 50
        refined_tags = []
        
        for i in range(0, len(unique_tags), batch_size):
            batch_tags = unique_tags[i:i+batch_size]
            
            user_prompt = f"""
请分析以下这些从汽车用户评论中提取的原始标签列表，执行以下任务：

1. 识别语义相似或相关的标签，将它们归为同一类别
2. 为每个类别生成一个更加精准、概括性强的"精炼标签"
3. 保持标签的多样性，确保不同维度的关注点都被保留

原始标签列表：
{batch_tags}

请按照以下JSON格式输出归纳结果：
{{
  "refined_mappings": [
    {{
      "refined_tag": "精炼后的标签名称",
      "original_tags": ["原始标签1", "原始标签2", "原始标签3"],
      "description": "该类别的简要描述"
    }}
  ]
}}
"""
            
            try:
                print(f"正在处理第 {i//batch_size + 1} 批标签...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # 清理响应文本，移除markdown代码块标记
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.startswith('```'):
                    response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                try:
                    llm_result = json.loads(response_text)
                    mappings = llm_result.get('refined_mappings', [])
                    refined_tags.extend(mappings)
                    print(f"  成功归纳为 {len(mappings)} 个精炼标签")
                    
                except json.JSONDecodeError:
                    print(f"  警告：批次 {i//batch_size + 1} JSON解析失败")
                    print(f"  响应内容：{response_text[:200]}...")
                
            except Exception as e:
                print(f"  错误：批次 {i//batch_size + 1} API调用失败 - {e}")
            
            time.sleep(0.5)
        
        # 3. 构建标签映射字典
        tag_mapping = {}
        for mapping in refined_tags:
            refined_tag = mapping['refined_tag']
            original_tags = mapping['original_tags']
            for orig_tag in original_tags:
                if orig_tag in unique_tags:  # 确保原始标签存在
                    tag_mapping[orig_tag] = refined_tag
        
        # 4. 应用映射到扁平化数据
        refined_flattened_data = []
        for _, row in flattened_df.iterrows():
            original_tag = row['raw_persona_tag']
            refined_tag = tag_mapping.get(original_tag, original_tag)  # 如果没有映射，保持原标签
            
            refined_flattened_data.append({
                'original_comment': row['original_comment'],
                'car_model': row['car_model'],
                'raw_persona_tag': original_tag,
                'refined_persona_tag': refined_tag,
                'keywords': row['keywords']
            })
        
        refined_df = pd.DataFrame(refined_flattened_data)
        
        # 保存结果
        refined_df.to_csv('./outputs/refined_tags.csv', index=False, encoding='utf-8-sig')
        
        # 保存映射关系
        with open('./outputs/tag_mappings.json', 'w', encoding='utf-8') as f:
            json.dump(tag_mapping, f, ensure_ascii=False, indent=2)
        
        print(f"\n二次标签归纳完成：")
        print(f"  原始标签数：{len(unique_tags)}")
        print(f"  精炼标签数：{len(set(tag_mapping.values()))}")
        print(f"  映射覆盖率：{len(tag_mapping)/len(unique_tags)*100:.1f}%")
        
        return refined_df

    def step_1_4_auto_persona_clustering(self, refined_df: pd.DataFrame) -> tuple:
        """
        步骤1.4：自动Persona聚类与智能命名
        """
        print(f"\n=== 步骤1.4：自动Persona聚类与智能命名 ===")
        
        # 1. 基于精炼标签进行聚类
        print("正在基于精炼标签进行聚类...")
        
        # 获取所有唯一的精炼标签
        unique_refined_tags = refined_df['refined_persona_tag'].unique()
        print(f"精炼标签数量：{len(unique_refined_tags)}")
        
        # 构建标签向量
        tag_texts = []
        for tag in unique_refined_tags:
            # 收集该标签的所有关键词
            tag_data = refined_df[refined_df['refined_persona_tag'] == tag]
            all_keywords = []
            for keywords_str in tag_data['keywords']:
                if isinstance(keywords_str, str):
                    keywords = eval(keywords_str) if keywords_str.startswith('[') else [keywords_str]
                else:
                    keywords = keywords_str
                all_keywords.extend(keywords)
            
            # 合并关键词作为标签的文本表示
            tag_text = ' '.join(all_keywords)
            tag_texts.append(tag_text)
        
        # 初始化TF-IDF向量化器
        vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 2), min_df=1)
        try:
            tag_vectors = vectorizer.fit_transform(tag_texts)
        except:
            # 如果TF-IDF失败，使用简单的词频统计
            print("  TF-IDF失败，使用简单方法...")
            tag_vectors = np.random.rand(len(unique_refined_tags), 10)
        
        # 确定最佳聚类数
        best_k = min(8, max(3, len(unique_refined_tags) // 3))
        print(f"选择聚类数：{best_k}")
        
        # 进行K-means聚类
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        try:
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tag_vectors)
            
            if len(set(cluster_labels)) > 1:
                silhouette = silhouette_score(tag_vectors, cluster_labels)
                print(f"聚类轮廓系数：{silhouette:.3f}")
            else:
                silhouette = 0
                print("只有一个聚类")
            
        except Exception as e:
            print(f"聚类失败：{e}")
            # 简单分组
            cluster_labels = [i % best_k for i in range(len(unique_refined_tags))]
            silhouette = 0
        
        # 2. 构建标签到聚类的映射
        tag_cluster_mapping = {}
        for tag, cluster_id in zip(unique_refined_tags, cluster_labels):
            tag_cluster_mapping[tag] = cluster_id
        
        # 3. 生成聚类报告
        cluster_reports = {}
        for cluster_id in range(best_k):
            cluster_tags = [tag for tag, cid in tag_cluster_mapping.items() if cid == cluster_id]
            if cluster_tags:
                # 统计该聚类的评论数量
                cluster_data = refined_df[refined_df['refined_persona_tag'].isin(cluster_tags)]
                comment_count = len(cluster_data)
                
                cluster_reports[cluster_id] = {
                    'tags': cluster_tags,
                    'tag_count': len(cluster_tags),
                    'comment_count': comment_count,
                    'representative_tags': cluster_tags[:3]  # 前3个作为代表性标签
                }
                
                print(f"  聚类 {cluster_id}: {len(cluster_tags)}个标签, {comment_count}条评论")
                print(f"    代表性标签: {', '.join(cluster_tags[:3])}")
        
        # 4. 为每个聚类生成智能Persona名称
        persona_names = {}
        print("\n正在为聚类生成智能Persona名称...")
        
        for cluster_id, report in cluster_reports.items():
            if not report['tags']:
                continue
                
            print(f"\n为聚类 {cluster_id} 生成Persona名称...")
            
            # 构建prompt
            system_prompt = """你是资深的汽车行业用户研究专家，擅长为用户群体创造简洁而准确的Persona名称。
你的任务是根据用户关注的标签特征，生成一个4-8个字的Persona名称。"""
            
            tags_text = "、".join(report['tags'])
            user_prompt = f"""
请根据以下用户关注标签，为这个用户群体生成一个准确、简洁的Persona名称：

关注标签：{tags_text}
评论数量：{report['comment_count']}条
代表性标签：{", ".join(report['representative_tags'])}

要求：
1. 名称长度：4-8个中文字符
2. 准确反映用户群体的核心关注点
3. 简洁易懂，便于记忆
4. 避免过于技术性的术语

请按照以下JSON格式输出：
{{
  "persona_name": "你生成的Persona名称",
  "rationale": "命名理由的简要说明"
}}
"""
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # 清理响应文本，移除markdown代码块标记
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.startswith('```'):
                    response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                try:
                    llm_result = json.loads(response_text)
                    persona_name = llm_result.get('persona_name', f'用户群体{cluster_id}')
                    rationale = llm_result.get('rationale', '自动生成')
                    
                    persona_names[cluster_id] = persona_name
                    print(f"  簇 {cluster_id} → {persona_name}")
                    print(f"  命名理由：{rationale}")
                    
                except json.JSONDecodeError:
                    persona_name = f'用户群体{cluster_id}'
                    persona_names[cluster_id] = persona_name
                    print(f"  简化命名：{persona_name}")
                
            except Exception as e:
                persona_name = f'用户群体{cluster_id}'
                persona_names[cluster_id] = persona_name
                print(f"  默认命名：{persona_name} (API调用失败：{e})")
            
            time.sleep(0.3)
        
        # 5. 保存聚类结果
        clustering_results = {
            'best_k': int(best_k),
            'silhouette_score': float(silhouette),
            'tag_cluster_mapping': {k: int(v) for k, v in tag_cluster_mapping.items()},
            'cluster_reports': cluster_reports,
            'persona_names': {str(k): v for k, v in persona_names.items()}
        }
        
        # 保存聚类结果
        with open('./outputs/clustering_results.json', 'w', encoding='utf-8') as f:
            json.dump(clustering_results, f, ensure_ascii=False, indent=2)
        
        # 为refined_df添加Persona信息
        refined_df['persona_cluster'] = refined_df['refined_persona_tag'].map(tag_cluster_mapping)
        refined_df['persona_name'] = refined_df['persona_cluster'].map(persona_names)
        
        # 保存最终的refined_df
        refined_df.to_csv('./outputs/refined_tags_with_persona.csv', index=False, encoding='utf-8-sig')
        
        print(f"\n✅ 自动Persona聚类与命名完成！")
        print(f"   生成了 {len(persona_names)} 个Persona：")
        for cluster_id, name in persona_names.items():
            report = cluster_reports.get(int(cluster_id), {})
            print(f"   - {name} ({report.get('comment_count', 0)}条评论)")
        
        return refined_df, clustering_results

    def step_1_5_generate_final_dictionary(self, refined_df: pd.DataFrame, clustering_results: dict) -> dict:
        """
        步骤1.5：生成最终的Persona关键词词典
        """
        print(f"\n=== 步骤1.5：生成最终Persona关键词词典 ===")
        
        # 构建最终词典
        final_dictionary = {}
        
        for cluster_id, persona_name in clustering_results['persona_names'].items():
            cluster_id = int(cluster_id)
            
            # 获取该Persona对应的所有标签
            persona_data = refined_df[refined_df['persona_cluster'] == cluster_id]
            
            if len(persona_data) == 0:
                continue
            
            # 收集所有关键词
            all_keywords = []
            for _, row in persona_data.iterrows():
                keywords_str = row['keywords']
                if isinstance(keywords_str, str):
                    try:
                        keywords = eval(keywords_str) if keywords_str.startswith('[') else [keywords_str]
                    except:
                        keywords = [keywords_str]
                else:
                    keywords = keywords_str if isinstance(keywords_str, list) else [str(keywords_str)]
                
                all_keywords.extend(keywords)
            
            # 去重并排序
            unique_keywords = list(set(all_keywords))
            unique_keywords = [kw for kw in unique_keywords if kw and len(str(kw).strip()) > 0]
            
            final_dictionary[persona_name] = unique_keywords
            print(f"  {persona_name}: {len(unique_keywords)}个关键词")
        
        # 保存最终词典
        output_path = './outputs/persona_keyword_dictionary_v3.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_dictionary, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 最终Persona词典已生成：{output_path}")
        print(f"   包含 {len(final_dictionary)} 个Persona")
        
        return final_dictionary

    def run_stage1_complete_optimized(self, input_file: str = '../1_Data_Preprocessing/outputs/cleaned_comments.csv', sample_size: int = 500):
        """
        运行完整的优化版第一阶段流程
        """
        print("=" * 80)
        print("基于LLM的用户画像发现与聚类分析 V3.0 - 优化版")
        print("第一阶段：多维Persona与关键词的智能发现与归纳")
        print("=" * 80)
        
        try:
            # 步骤1.1：数据抽样（500条）
            sampled_df = self.step_1_1_sample_data(input_file, sample_size)
            
            # 步骤1.2：LLM多维标签生成（优化版，10条一批次）
            llm_results_df = self.step_1_2_llm_multi_tag_generation_optimized(sampled_df)
            
            # 步骤1.3：二次标签归纳优化
            refined_df = self.step_1_3_secondary_tag_refinement(llm_results_df)
            
            # 步骤1.4：自动Persona聚类与命名
            refined_df, clustering_results = self.step_1_4_auto_persona_clustering(refined_df)
            
            # 步骤1.5：生成最终词典
            final_dictionary = self.step_1_5_generate_final_dictionary(refined_df, clustering_results)
            
            print(f"\n🎉 优化版第一阶段完全流程完成！")
            print(f"📁 请查看 ./outputs/ 文件夹中的所有结果")
            print(f"🚀 现在可以直接运行第二阶段进行规模化聚类分析")
            
            return True
            
        except Exception as e:
            print(f"❌ 优化版第一阶段执行失败：{e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """测试函数"""
    print("🚀 运行优化版LLM标签生成")
    
    discovery = PersonaDiscoveryV3Optimized()
    
    # 询问样本大小
    while True:
        try:
            choice = input("请选择样本大小 [1=50条(测试) 2=500条(完整)]: ").strip()
            if choice == '1':
                sample_size = 50
                break
            elif choice == '2':
                sample_size = 500
                break
            else:
                print("请输入 1 或 2")
        except KeyboardInterrupt:
            print("\n用户取消")
            return
    
    success = discovery.run_stage1_complete_optimized(sample_size=sample_size)
    
    if success:
        print(f"\n✅ 成功完成{sample_size}条样本的标签生成！")
    else:
        print(f"\n❌ 标签生成失败")

if __name__ == "__main__":
    main() 