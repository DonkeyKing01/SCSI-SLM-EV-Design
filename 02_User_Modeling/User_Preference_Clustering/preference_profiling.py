"""
新版用户画像系统 - 用户向量提取模块
基于原有Persona-pro思路，但停止在向量提取阶段，不进行聚类分析

核心功能：
1. 使用LLM提取用户评论的多维度标签
2. 构建统一的用户画像维度词典
3. 将用户评论映射为多维向量
4. 输出用户向量数据供后续分析

维度设计（与Model_New保持一致）：
- 外观设计：外观、颜值、造型等
- 内饰质感：内饰、座椅、材质等  
- 智能配置：科技、智能、辅助驾驶等
- 空间实用：空间、储物、实用性等
- 舒适体验：舒适、静音、减震等
- 操控性能：操控、动力、驾驶感受等
- 续航能耗：续航、充电、能耗等
- 价值认知：性价比、价格、经济性等
"""

import pandas as pd
import numpy as np
import json
import time
import os
import random
import threading
import signal
import queue
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from openai import OpenAI
from collections import defaultdict, Counter
from dotenv import load_dotenv
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 加载根目录的环境变量
root_dir = Path(__file__).parent.parent.parent
load_dotenv(root_dir / '.env')

class UserVectorExtractor:
    def __init__(self):
        """初始化用户向量提取器"""
        # 从环境变量读取 API Key
        api_key = os.getenv('SILICONFLOW_API_KEY')
        if not api_key:
            raise ValueError(
                "未找到 SILICONFLOW_API_KEY 环境变量。\n"
                "请在项目根目录创建 .env 文件，并配置 SILICONFLOW_API_KEY。\n"
                "参考根目录的 .env.example 文件进行配置。"
            )
                "参考 .env.example 文件进行配置。"
            )
        
        # 保存API配置供并发线程使用
        self.base_url = os.getenv('SILICONFLOW_BASE_URL', 'https://api.siliconflow.cn/v1')
        self.api_key = api_key
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.model = os.getenv('LLM_MODEL', "deepseek-ai/DeepSeek-V3")
        
        # 统一的用户画像维度定义（与Model_New保持一致）
        self.persona_dimensions = {
            '外观设计': {
                'description': '车辆外观造型、颜值、设计风格相关',
                'keywords': ['外观', '颜值', '造型', '设计', '前脸', '尾部', '车身', '外形', 
                           '大灯', '尾灯', '轮毂', '车漆', '线条', '风格', '时尚', '运动', '优雅', '霸气']
            },
            '内饰质感': {
                'description': '内饰材质、做工、豪华感、精致度相关', 
                'keywords': ['内饰', '座椅', '方向盘', '中控', '仪表', '材质', '做工', '精致', 
                           '豪华', '质感', '皮质', '软硬', '包裹', '支撑', '档次']
            },
            '智能配置': {
                'description': '智能科技、辅助驾驶、车机系统相关',
                'keywords': ['智能', '科技', '配置', '功能', '辅助', '导航', '音响', '车机', 
                           '语音', '自动', '雷达', '摄像头', '系统', '软件', '升级', '互联']
            },
            '空间实用': {
                'description': '车内空间、储物、实用性相关',
                'keywords': ['空间', '座椅', '腿部', '头部', '后排', '前排', '乘坐', '储物', 
                           '后备箱', '装载', '实用', '宽敞', '紧凑', '够用']
            },
            '舒适体验': {
                'description': '乘坐舒适性、静音性、减震效果相关',
                'keywords': ['舒适', '静音', '噪音', '减震', '悬挂', '滤震', '平顺', '稳定', 
                           '颠簸', '震动', '隔音', '风噪', '胎噪', '异响']
            },
            '操控性能': {
                'description': '驾驶操控、动力性能、驾驶感受相关',
                'keywords': ['操控', '驾驶', '动力', '加速', '推背', '性能', '手感', '指向', 
                           '转向', '刹车', '油门', '制动', '精准', '轻松', '灵活', '响应']
            },
            '续航能耗': {
                'description': '续航里程、能耗表现、充电相关',
                'keywords': ['续航', '能耗', '充电', '电池', '里程', '电量', '快充', '慢充', 
                           '省电', '耗电', '充电桩', '电费']
            },
            '价值认知': {
                'description': '性价比、价格、经济性相关认知',
                'keywords': ['性价比', '价格', '便宜', '值得', '划算', '经济', '实惠', '贵', 
                           '超值', '物有所值', '成本', '保值']
            }
        }
        
        # 创建输出目录
        self.output_dir = './outputs'
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"初始化完成")
        print(f"维度数量: {len(self.persona_dimensions)} 个")
        print(f"输出目录: {self.output_dir}")
    
    def load_comment_data(self, data_source: str = "../../01_SSE_Analysis/1_Data_Preprocessing/outputs/cleaned_comments.csv") -> pd.DataFrame:
        """加载评论数据"""
        print(f"\n=== 步骤1：加载评论数据 ===")
        
        try:
            df = pd.read_csv(data_source, encoding='utf-8')
            print(f"成功加载数据: {len(df)} 条评论")
            
            # 数据清洗
            if 'cleaned_comment' in df.columns:
                df = df[df['cleaned_comment'].str.len() >= 15]  # 过滤太短的评论
                print(f"过滤后数据: {len(df)} 条有效评论")
            
            return df
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return pd.DataFrame()
    
    def extract_dimension_tags_batch(self, comments: List[str], batch_size: int = 20, max_workers: int = None) -> List[Dict]:
        """优化的并发提取评论维度标签，支持超时控制、实时保存、断点续跑与失败重试全量覆盖"""
        print(f"\n=== 步骤2：LLM维度标签提取（优化版） ===")

        # 初始化配置（硬编码稳定参数，不依赖环境变量）
        total_batches = (len(comments) + batch_size - 1) // batch_size
        concurrency = max_workers or 4  # 并发线程固定为4
        task_timeout = 180  # 单个任务超时（秒）
        save_every = 5  # 每5批保存一次
        
        print(f"配置参数:")
        print(f"   - 并发线程: {concurrency}")
        print(f"   - 批大小: {batch_size}")
        print(f"   - 总批数: {total_batches}")
        print(f"   - 任务超时: {task_timeout}秒")
        print(f"   - 保存频率: 每{save_every}批")

        # 文件路径
        partial_path = os.path.join(self.output_dir, 'dimension_results_partial.jsonl')
        progress_path = os.path.join(self.output_dir, 'progress.log')
        status_path = os.path.join(self.output_dir, 'task_status.json')

        # 结果存储和锁
        results: List[Dict] = []
        results_lock = threading.Lock()
        # comment_index -> result 的映射，便于去重与覆盖
        results_map: Dict[int, Dict] = {}
        
        # 任务状态跟踪
        task_status = {
            'submitted': 0,
            'completed': 0,
            'failed': 0,
            'start_time': time.time(),
            'last_save_time': time.time()
        }
        status_lock = threading.Lock()

        # 断点续跑：启用（硬编码），从已有文件恢复已完成的条目
        completed_batches_at_start = 0
        resume = True

        if resume and os.path.exists(partial_path):
            try:
                print("检测到断点文件，正在恢复...")
                with open(partial_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            item = json.loads(line)
                            if isinstance(item, dict) and 'comment_index' in item:
                                idx = int(item['comment_index'])
                                results_map[idx] = item
                                results.append(item)

                if results_map:
                    completed_indices = set(results_map.keys())
                    max_index = max(completed_indices)
                    completed_batches_at_start = (max_index + 1) // batch_size
                    print(f"恢复了 {len(results_map)} 条结果")

            except Exception as e:
                print(f"恢复失败，将从头开始: {e}")
                results.clear()
                results_map.clear()
                completed_batches_at_start = 0

        def save_progress_snapshot(current_results: List[Dict], current_status: Dict) -> None:
            """保存进度快照（线程安全）"""
            try:
                # 保存结果数据
                sorted_results = sorted(current_results, key=lambda r: r.get('comment_index', 0))
                tmp_path = partial_path + '.tmp'
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    for item in sorted_results:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
                # 原子替换
                if os.path.exists(partial_path):
                    try:
                        os.remove(partial_path)
                    except Exception:
                        pass
                os.replace(tmp_path, partial_path)
                
                # 保存进度状态
                with open(progress_path, 'w', encoding='utf-8') as pf:
                    pf.write(f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    pf.write(f"completed_batches={current_status['completed']}/{total_batches}\n")
                    pf.write(f"results_count={len(sorted_results)}\n")
                    pf.write(f"failed_count={current_status['failed']}\n")
                    pf.write(f"success_rate={current_status['completed']/(current_status['completed']+current_status['failed'])*100:.1f}%\n" if (current_status['completed']+current_status['failed']) > 0 else "success_rate=0.0%\n")
                
                # 保存详细状态
                with open(status_path, 'w', encoding='utf-8') as sf:
                    json.dump(current_status, sf, ensure_ascii=False, indent=2)
                
                elapsed = time.time() - current_status['start_time']
                rate = current_status['completed'] / elapsed if elapsed > 0 else 0
                eta = (total_batches - current_status['completed']) / rate if rate > 0 else 0
                
                print(f"  进度保存: {current_status['completed']}/{total_batches} 批 ({current_status['completed']/total_batches*100:.1f}%), "
                      f"结果 {len(sorted_results)} 条, 失败 {current_status['failed']} 批")
                print(f"  速度: {rate:.2f} 批/秒, 预计剩余: {eta/60:.1f} 分钟")
                
                current_status['last_save_time'] = time.time()
                
            except Exception as e:
                print(f"  保存失败: {e}")

        # 预构建提示模板
        system_prompt = (
            "你是汽车行业的专业分析师，擅长从用户评论中提取用户关注的各个维度特征。\n"
            "你需要分析每条评论，识别用户关注的具体维度，并提取相关关键词。\n"
            "请保持分析的准确性和一致性。"
        )
        dimension_desc = "\n".join([
            f"- {dim}: {info['description']}" for dim, info in self.persona_dimensions.items()
        ])

        def call_api_with_retry_and_timeout(global_indices: List[int], batch_comments: List[str], batch_no: int) -> List[Dict]:
            """带超时和重试的API调用。global_indices 与 batch_comments 一一对应。"""
            
            def local_fallback() -> List[Dict]:
                """本地关键词匹配回退方案"""
                fallback_results: List[Dict] = []
                for i, text in enumerate(batch_comments):
                    dims: Dict[str, Dict] = {}
                    text_l = str(text).lower()
                    for dim, info in self.persona_dimensions.items():
                        hit = []
                        for kw in info['keywords']:
                            if kw.lower() in text_l:
                                hit.append(kw)
                            if len(hit) >= 5:
                                break
                        if hit:
                            dims[dim] = {"keywords": hit, "sentiment": "中性"}
                    fallback_results.append({
                        "comment_index": global_indices[i],
                        "dimensions": dims
                    })
                return fallback_results

            # 每个线程创建独立客户端
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            max_retries = 3  # 固定重试次数
            backoff = 2.0  # 固定退避起始值

            for attempt in range(max_retries):
                try:
                    print(f"  批次 {batch_no}: 尝试 {attempt+1}/{max_retries}")
                    
                    user_prompt = (
                        f"请分析以下汽车用户评论，为每条评论提取用户关注的维度及相关关键词。\n\n"
                        f"可选维度类型：\n{dimension_desc}\n\n"
                        f"评论列表：\n{json.dumps(batch_comments, ensure_ascii=False, indent=2)}\n\n"
                        "对于每条评论，请：\n"
                        "1. 识别评论中体现的用户关注维度\n"
                        "2. 提取每个维度对应的关键词\n"
                        "3. 评估用户对该维度的情感倾向（正面/负面/中性）\n\n"
                        "请严格按照以下JSON格式输出：\n"
                        "{\n  \"results\": [\n    {\n      \"comment_index\": 0,\n      \"dimensions\": {\n        \"外观设计\": {\"keywords\": [\"关键词1\", \"关键词2\"], \"sentiment\": \"正面\"},\n        \"内饰质感\": {\"keywords\": [\"关键词3\"], \"sentiment\": \"中性\"}\n      }\n    }\n  ]\n}\n"
                    )
                    
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=3000,
                        timeout=120  # 增加API调用超时时间
                    )
                    
                    response_text = response.choices[0].message.content.strip()
                    
                    # 清理响应文本
                    for prefix in ['```json', '```']:
                        if response_text.startswith(prefix):
                            response_text = response_text[len(prefix):]
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()

                    # 解析结果
                    try:
                        parsed = json.loads(response_text)
                        if 'results' in parsed and isinstance(parsed['results'], list):
                            batch_results = []
                            for result in parsed['results']:
                                if isinstance(result, dict) and 'comment_index' in result:
                                    local_idx = int(result['comment_index'])
                                    if 0 <= local_idx < len(global_indices):
                                        result['comment_index'] = global_indices[local_idx]
                                    else:
                                        continue
                                    batch_results.append(result)
                            
                            if batch_results:
                                print(f"  批次 {batch_no}: 成功提取 {len(batch_results)} 条")
                                return batch_results
                        
                        print(f"  批次 {batch_no}: 响应格式异常，重试...")
                        raise RuntimeError('Invalid response format')
                        
                    except json.JSONDecodeError as e:
                        print(f"  批次 {batch_no}: JSON解析失败 - {e}")
                        raise RuntimeError(f'JSON parse error: {e}')

                except Exception as e:
                    error_msg = str(e)
                    
                    # 余额不足直接使用本地回退
                    if any(keyword in error_msg.lower() for keyword in ['403', 'insufficient', 'balance', '30001']):
                        print(f"  批次 {batch_no}: 余额不足，使用本地规则回退")
                        return local_fallback()
                    
                    # 429错误（速率限制）- 增加更长等待时间但继续重试
                    if '429' in error_msg or 'rate limit' in error_msg.lower() or 'too many requests' in error_msg.lower():
                        if attempt == max_retries - 1:
                            print(f"  批次 {batch_no}: 速率限制重试耗尽，使用本地规则回退")
                            return local_fallback()
                        sleep_time = min(60, backoff * (3 ** attempt)) + random.uniform(5, 15)  # 429错误等待更久
                        print(f"  批次 {batch_no}: 速率限制 [{error_msg[:50]}...] 等待 {sleep_time:.1f}s 后重试")
                        time.sleep(sleep_time)
                        continue
                    
                    # 超时错误 - 快速重试，减少等待时间
                    if 'timeout' in error_msg.lower() or 'timed out' in error_msg.lower():
                        if attempt == max_retries - 1:
                            print(f"  批次 {batch_no}: 超时重试耗尽，使用本地规则回退")
                            return local_fallback()
                        sleep_time = min(2.0, backoff * (1.2 ** attempt)) + random.uniform(0.5, 1.5)  # 大幅减少超时等待时间
                        print(f"  批次 {batch_no}: 超时 [{error_msg[:50]}...] 等待 {sleep_time:.1f}s 后重试")
                        time.sleep(sleep_time)
                        continue
                    
                    # 网络连接错误 - 快速重试
                    if any(keyword in error_msg.lower() for keyword in ['connection', 'network', 'ssl', 'certificate']):
                        if attempt == max_retries - 1:
                            print(f"  批次 {batch_no}: 网络错误重试耗尽，使用本地规则回退")
                            return local_fallback()
                        sleep_time = min(4.0, backoff * (1.4 ** attempt)) + random.uniform(1, 2)
                        print(f"  批次 {batch_no}: 网络错误 [{error_msg[:50]}...] 等待 {sleep_time:.1f}s 后重试")
                        time.sleep(sleep_time)
                        continue
                    
                    # 最后一次尝试失败，使用本地回退
                    if attempt == max_retries - 1:
                        print(f"  批次 {batch_no}: 达到最大重试次数，使用本地规则回退")
                        return local_fallback()
                    
                    # 其他错误，快速重试，减少等待时间
                    sleep_time = min(3.0, backoff * (1.3 ** attempt)) + random.uniform(0.5, 1.5)
                    print(f"  批次 {batch_no}: 错误 [{error_msg[:50]}...] 等待 {sleep_time:.1f}s 后重试")
                    time.sleep(sleep_time)

            # 理论上不会到达这里，但作为保险
            return local_fallback()

        # 批量任务处理
        print(f"\n开始处理，从批次 {completed_batches_at_start + 1} 开始...")
        
        # 准备任务队列（仅加入尚未处理的索引以避免重复调用）
        batch_queue = queue.Queue()
        processed_indices_set = set(results_map.keys())
        for batch_idx in range(0, len(comments), batch_size):
            global_indices = list(range(batch_idx, min(batch_idx + batch_size, len(comments))))
            to_do_indices = [i for i in global_indices if i not in processed_indices_set]
            batch_no = batch_idx // batch_size + 1
            if not to_do_indices:
                # 该批次已完全处理过，仅用于日志统计
                with status_lock:
                    task_status['completed'] += 1
                continue
            batch_comments = [comments[i] for i in to_do_indices]
            batch_queue.put((to_do_indices, batch_comments, batch_no))

        # 使用线程池处理任务
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # 提交初始任务
            active_futures = {}
            failed_batches: List[Dict] = []
            
            # 提交第一轮任务
            for _ in range(min(concurrency, batch_queue.qsize())):
                if not batch_queue.empty():
                    to_do_indices, batch_comments, batch_no = batch_queue.get()
                    future = executor.submit(call_api_with_retry_and_timeout, to_do_indices, batch_comments, batch_no)
                    active_futures[future] = (batch_no, time.time(), to_do_indices)
                    
                    with status_lock:
                        task_status['submitted'] += 1

            # 处理完成的任务
            while active_futures:
                try:
                    # 等待任务完成，带超时
                    completed_futures = []
                    for future in list(active_futures.keys()):
                        try:
                            batch_results = future.result(timeout=1)  # 非阻塞检查
                            completed_futures.append((future, batch_results, None))
                        except TimeoutError:
                            # 检查任务是否超时
                            batch_no, submit_time, to_do_indices = active_futures[future]
                            if time.time() - submit_time > task_timeout:
                                print(f"  批次 {batch_no}: 任务超时，取消")
                                future.cancel()
                                completed_futures.append((future, [], f"timeout after {task_timeout}s"))
                        except Exception as e:
                            batch_no, _, to_do_indices = active_futures[future]
                            print(f"  批次 {batch_no}: 任务异常 - {e}")
                            completed_futures.append((future, [], str(e)))

                    # 处理完成的任务
                    for future, batch_results, error in completed_futures:
                        batch_no, _, to_do_indices = active_futures.pop(future)
                        
                        with status_lock:
                            if error:
                                task_status['failed'] += 1
                                failed_batches.append({
                                    'batch_no': batch_no,
                                    'indices': to_do_indices,
                                    'reason': error
                                })
                            else:
                                task_status['completed'] += 1
                                
                                # 添加结果
                                with results_lock:
                                    for item in batch_results:
                                        idx = int(item.get('comment_index', -1))
                                        if idx >= 0 and idx not in results_map:
                                            results_map[idx] = item
                                            results.append(item)

                        # 提交新任务
                        if not batch_queue.empty():
                            to_do_indices, batch_comments, new_batch_no = batch_queue.get()
                            new_future = executor.submit(call_api_with_retry_and_timeout, to_do_indices, batch_comments, new_batch_no)
                            active_futures[new_future] = (new_batch_no, time.time(), to_do_indices)
                            
                            with status_lock:
                                task_status['submitted'] += 1

                        # 定期保存进度
                        with status_lock:
                            current_time = time.time()
                            should_save = (
                                task_status['completed'] % save_every == 0 or 
                                current_time - task_status['last_save_time'] > 60 or  # 每分钟强制保存一次
                                len(active_futures) == 0  # 最后一批
                            )
                            
                            if should_save:
                                with results_lock:
                                    save_progress_snapshot(list(results_map.values()), task_status.copy())

                    # 短暂休眠避免忙等待
                    if active_futures:
                        time.sleep(0.1)
                        
                except KeyboardInterrupt:
                    print("\n⚠️ 检测到中断信号，正在保存当前进度...")
                    with status_lock, results_lock:
                        save_progress_snapshot(list(results_map.values()), task_status.copy())
                    raise

        # 第一阶段完成后保存
        with status_lock, results_lock:
            save_progress_snapshot(list(results_map.values()), task_status.copy())

        # 第二阶段：对失败批次与遗漏索引进行集中重试（不影响之前主流程的并发效率）
        print("\n进入失败与遗漏重试阶段...")
        missing_indices = [i for i in range(len(comments)) if i not in results_map]
        retry_indices = set()
        for fb in failed_batches:
            for i in fb.get('indices', []):
                retry_indices.add(i)
        retry_indices.update(missing_indices)

        if retry_indices:
            retry_indices = sorted(list(retry_indices))
            print(f"需要重试的条目: {len(retry_indices)} 条")

            def run_retry(indices: List[int], retry_batch_size: int = min(batch_size, 12)) -> None:
                q = queue.Queue()
                for start in range(0, len(indices), retry_batch_size):
                    chunk_indices = indices[start:start + retry_batch_size]
                    chunk_comments = [comments[i] for i in chunk_indices]
                    q.put((chunk_indices, chunk_comments))

                with ThreadPoolExecutor(max_workers=max_workers or 4) as pool:
                    retry_futures = {}
                    while not q.empty() and len(retry_futures) < (max_workers or 4):
                        gi, gc = q.get()
                        fut = pool.submit(call_api_with_retry_and_timeout, gi, gc, -1)
                        retry_futures[fut] = gi

                    while retry_futures:
                        done = []
                        for fut in list(retry_futures.keys()):
                            try:
                                res = fut.result(timeout=1)
                                done.append((fut, res))
                            except TimeoutError:
                                continue
                            except Exception:
                                done.append((fut, []))

                        for fut, res in done:
                            gi = retry_futures.pop(fut)
                            if res:
                                with results_lock:
                                    for item in res:
                                        idx = int(item.get('comment_index', -1))
                                        if idx >= 0:
                                            # 重试结果覆盖先前的结果（包括本地回退）
                                            results_map[idx] = item
                            if not q.empty():
                                gi2, gc2 = q.get()
                                nf = pool.submit(call_api_with_retry_and_timeout, gi2, gc2, -1)
                                retry_futures[nf] = gi2

            run_retry(retry_indices)

        # 第三阶段：仍有缺失则使用本地回退一次性补齐
        remaining_missing = [i for i in range(len(comments)) if i not in results_map]
        if remaining_missing:
            print(f"剩余未覆盖条目 {len(remaining_missing)} 条，使用本地回退补齐")

            def local_fallback_bulk(indices: List[int]) -> List[Dict]:
                out = []
                for i in indices:
                    text = comments[i]
                    dims: Dict[str, Dict] = {}
                    text_l = str(text).lower()
                    for dim, info in self.persona_dimensions.items():
                        hit = []
                        for kw in info['keywords']:
                            if kw.lower() in text_l:
                                hit.append(kw)
                            if len(hit) >= 5:
                                break
                        if hit:
                            dims[dim] = {"keywords": hit, "sentiment": "中性"}
                    out.append({"comment_index": i, "dimensions": dims})
                return out

            bulk_results = local_fallback_bulk(remaining_missing)
            with results_lock:
                for item in bulk_results:
                    results_map[item['comment_index']] = item

        # 统一最终结果列表
        final_results = [results_map[i] for i in sorted(results_map.keys())]

        # 最终保存（统一提交）
        with status_lock:
            task_status['completed'] = total_batches
        save_progress_snapshot(final_results, task_status.copy())

        # 输出最终统计
        elapsed = time.time() - task_status['start_time']
        print(f"\n维度标签提取完成!")
        print(f"统计信息:")
        print(f"   - 总批次数: {total_batches}")
        print(f"   - 成功批次: {task_status['completed']}")
        print(f"   - 失败批次: {task_status['failed']}")
        print(f"   - 提取结果: {len(final_results)} 条")
        print(f"   - 总耗时: {elapsed/60:.1f} 分钟")
        print(f"   - 平均速度: {task_status['completed']/elapsed:.2f} 批/秒")

        return final_results
    
    def build_dimension_vectors(self, df: pd.DataFrame, dimension_results: List[Dict]) -> pd.DataFrame:
        """构建用户维度向量"""
        print(f"\n=== 步骤3：构建用户维度向量 ===")
        
        # 创建向量数据结构
        vector_data = []
        
        for idx, row in df.iterrows():
            # 查找对应的维度提取结果
            dimension_data = None
            for result in dimension_results:
                if result['comment_index'] == idx:
                    dimension_data = result
                    break
            
            # 初始化向量（每个维度的强度值）
            user_vector = {
                'comment_id': idx,
                'original_comment': row.get('original_comment', ''),
                'cleaned_comment': row.get('cleaned_comment', ''),
                'car_model': row.get('car_model', ''),
                'user_name': row.get('user_name', ''),
            }
            
            # 为每个维度计算强度值
            for dimension in self.persona_dimensions.keys():
                intensity = 0.0
                keywords_found = []
                sentiment = 'neutral'
                
                if dimension_data and 'dimensions' in dimension_data:
                    dim_info = dimension_data['dimensions'].get(dimension, {})
                    if dim_info:
                        keywords_found = dim_info.get('keywords', [])
                        sentiment = dim_info.get('sentiment', 'neutral')
                        
                        # 计算强度值（基于关键词数量和情感）
                        keyword_count = len(keywords_found)
                        if keyword_count > 0:
                            base_intensity = min(keyword_count / 3.0, 1.0)  # 基础强度
                            
                            # 情感调整
                            sentiment_multiplier = {
                                '正面': 1.2,
                                '负面': 0.8, 
                                '中性': 1.0
                            }.get(sentiment, 1.0)
                            
                            intensity = base_intensity * sentiment_multiplier
                            intensity = min(intensity, 1.0)  # 限制最大值
                
                user_vector[f'{dimension}_强度'] = intensity
                user_vector[f'{dimension}_关键词'] = json.dumps(keywords_found, ensure_ascii=False)
                user_vector[f'{dimension}_情感'] = sentiment
        
            vector_data.append(user_vector)
        
        # 创建向量DataFrame
        vector_df = pd.DataFrame(vector_data)
        
        print(f"构建用户向量完成")
        print(f"向量维度: {len(self.persona_dimensions)} 个")
        print(f"用户数量: {len(vector_df)} 个")
        
        # 统计各维度的分布
        print(f"\n各维度强度分布：")
        for dimension in self.persona_dimensions.keys():
            intensity_col = f'{dimension}_强度'
            if intensity_col in vector_df.columns:
                mean_intensity = vector_df[intensity_col].mean()
                non_zero_count = (vector_df[intensity_col] > 0).sum()
                print(f"  {dimension}: 平均强度 {mean_intensity:.3f}, 有效用户 {non_zero_count} 个")
        
        return vector_df
    
    def analyze_dimension_insights(self, vector_df: pd.DataFrame) -> Dict:
        """分析维度洞察"""
        print(f"\n=== 步骤4：维度洞察分析 ===")
        
        insights = {
            'dimension_stats': {},
            'user_profiles': {},
            'car_model_analysis': {},
            'correlation_analysis': {}
        }
        
        # 1. 维度统计分析
        for dimension in self.persona_dimensions.keys():
            intensity_col = f'{dimension}_强度'
            
            if intensity_col in vector_df.columns:
                intensities = vector_df[intensity_col]
                
                insights['dimension_stats'][dimension] = {
                    '平均强度': float(intensities.mean()),
                    '标准差': float(intensities.std()),
                    '最大值': float(intensities.max()),
                    '有效用户数': int((intensities > 0).sum()),
                    '覆盖率': float((intensities > 0).sum() / len(vector_df))
                }
        
        # 2. 用户画像分析（找出每个用户的主导维度）
        dimension_cols = [f'{dim}_强度' for dim in self.persona_dimensions.keys()]
        for idx, row in vector_df.iterrows():
            user_intensities = {col.replace('_强度', ''): row[col] for col in dimension_cols}
            
            # 找出用户的top3关注维度
            sorted_dims = sorted(user_intensities.items(), key=lambda x: x[1], reverse=True)
            top_dimensions = [dim for dim, intensity in sorted_dims[:3] if intensity > 0.1]
            
            if top_dimensions:
                profile_key = ' + '.join(top_dimensions)
                if profile_key not in insights['user_profiles']:
                    insights['user_profiles'][profile_key] = {
                        'user_count': 0,
                        'avg_total_intensity': 0,
                        'users': []
                    }
                
                insights['user_profiles'][profile_key]['user_count'] += 1
                insights['user_profiles'][profile_key]['users'].append({
                    'comment_id': row['comment_id'],
                    'car_model': row['car_model'],
                    'total_intensity': sum(user_intensities.values())
                })
        
        # 3. 车型分析
        for car_model in vector_df['car_model'].unique():
            if pd.notna(car_model):
                car_data = vector_df[vector_df['car_model'] == car_model]
                
                car_dimension_means = {}
                for dimension in self.persona_dimensions.keys():
                    intensity_col = f'{dimension}_强度'
                    if intensity_col in car_data.columns:
                        car_dimension_means[dimension] = float(car_data[intensity_col].mean())
                
                # 找出该车型用户最关注的维度
                top_dimensions = sorted(car_dimension_means.items(), key=lambda x: x[1], reverse=True)[:3]
                
                insights['car_model_analysis'][car_model] = {
                    'user_count': len(car_data),
                    'dimension_preferences': dict(top_dimensions),
                    'overall_intensity': sum(car_dimension_means.values())
                }
        
        # 4. 维度相关性分析
        dimension_matrix = vector_df[dimension_cols].corr()
        correlations = {}
        for i, dim1 in enumerate(self.persona_dimensions.keys()):
            for j, dim2 in enumerate(self.persona_dimensions.keys()):
                if i < j:  # 避免重复
                    corr_value = dimension_matrix.iloc[i, j]
                    if not pd.isna(corr_value):
                        correlations[f'{dim1} vs {dim2}'] = float(corr_value)
        
        insights['correlation_analysis'] = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
        
        print(f"维度洞察分析完成")
        print(f"识别了 {len(insights['user_profiles'])} 种用户画像组合")
        print(f"分析了 {len(insights['car_model_analysis'])} 个车型")
        
        return insights
    
    def save_results(self, vector_df: pd.DataFrame, insights: Dict, sample_size: int = None):
        """保存分析结果"""
        print(f"\n=== 步骤5：保存分析结果 ===")
        
        # 1. 保存用户向量数据
        vector_path = os.path.join(self.output_dir, 'user_dimension_vectors.csv')
        vector_df.to_csv(vector_path, index=False, encoding='utf-8-sig')
        print(f"用户向量数据已保存: {vector_path}")
        
        # 2. 保存维度洞察报告
        insights_path = os.path.join(self.output_dir, 'dimension_insights.json')
        with open(insights_path, 'w', encoding='utf-8') as f:
            json.dump(insights, f, ensure_ascii=False, indent=2)
        print(f"维度洞察报告已保存: {insights_path}")
        
        # 3. 生成简化的向量矩阵（仅强度值）
        dimension_cols = [f'{dim}_强度' for dim in self.persona_dimensions.keys()]
        simple_matrix = vector_df[['comment_id', 'car_model'] + dimension_cols].copy()
        
        # 重命名列以简化
        column_mapping = {f'{dim}_强度': dim for dim in self.persona_dimensions.keys()}
        simple_matrix = simple_matrix.rename(columns=column_mapping)
        
        matrix_path = os.path.join(self.output_dir, 'user_vector_matrix.csv')
        simple_matrix.to_csv(matrix_path, index=False, encoding='utf-8-sig')
        print(f"简化向量矩阵已保存: {matrix_path}")
        
        # 4. 生成总结报告
        summary_report = self.generate_summary_report(vector_df, insights, sample_size)
        report_path = os.path.join(self.output_dir, 'analysis_summary.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        print(f"分析总结报告已保存: {report_path}")
        
        print(f"\n所有结果已保存到: {self.output_dir}")
    
    def generate_summary_report(self, vector_df: pd.DataFrame, insights: Dict, sample_size: int = None) -> str:
        """生成分析总结报告"""
        
        report = f"""# 新版用户画像分析报告

## 分析概述

本报告基于用户评论数据，使用LLM技术提取了 {len(self.persona_dimensions)} 个核心用户画像维度，
并将用户偏好转换为多维向量数据。

### 数据统计
- 分析评论数量: {len(vector_df)} 条
{f"- 样本数据: {sample_size} 条" if sample_size else ""}
- 用户画像维度: {len(self.persona_dimensions)} 个
- 涉及车型数量: {len(vector_df['car_model'].unique())} 个

### 用户画像维度定义

"""
        
        for dimension, info in self.persona_dimensions.items():
            stats = insights['dimension_stats'].get(dimension, {})
            report += f"#### {dimension}\n"
            report += f"- 描述: {info['description']}\n"
            if stats:
                report += f"- 平均关注强度: {stats.get('平均强度', 0):.3f}\n"
                report += f"- 用户覆盖率: {stats.get('覆盖率', 0)*100:.1f}%\n"
            report += f"- 关键词示例: {', '.join(info['keywords'][:8])}\n\n"
        
        report += "## 核心发现\n\n"
        
        # 1. 最受关注的维度
        dim_stats = insights['dimension_stats']
        top_dimensions = sorted(dim_stats.items(), key=lambda x: x[1].get('平均强度', 0), reverse=True)[:5]
        
        report += "### 用户最关注的维度（按平均强度排序）\n\n"
        for i, (dim, stats) in enumerate(top_dimensions, 1):
            report += f"{i}. **{dim}**: 平均强度 {stats.get('平均强度', 0):.3f}, 覆盖率 {stats.get('覆盖率', 0)*100:.1f}%\n"
        
        # 2. 主要用户画像类型
        user_profiles = insights['user_profiles']
        top_profiles = sorted(user_profiles.items(), key=lambda x: x[1]['user_count'], reverse=True)[:8]
        
        report += "\n### 主要用户画像类型（按用户数量排序）\n\n"
        for i, (profile, data) in enumerate(top_profiles, 1):
            report += f"{i}. **{profile}型用户**: {data['user_count']} 位用户\n"
        
        # 3. 车型用户偏好特征
        car_analysis = insights['car_model_analysis']
        top_cars = sorted(car_analysis.items(), key=lambda x: x[1]['user_count'], reverse=True)[:10]
        
        report += "\n### 车型用户偏好特征（Top 10）\n\n"
        for car, data in top_cars:
            top_dims = list(data['dimension_preferences'].keys())[:2]
            report += f"- **{car}** ({data['user_count']} 位用户): 主要关注 {', '.join(top_dims)}\n"
        
        # 4. 维度相关性
        correlations = insights['correlation_analysis']
        
        report += "\n### 维度相关性分析（Top 5 强相关）\n\n"
        for i, (pair, corr) in enumerate(list(correlations.items())[:5], 1):
            report += f"{i}. {pair}: 相关系数 {corr:.3f}\n"
        
        report += f"""

## 数据说明

### 向量构建方法
1. 使用LLM分析每条评论，提取各维度的关键词和情感倾向
2. 基于关键词数量和情感计算维度强度值（0-1范围）
3. 构建每个用户的{len(self.persona_dimensions)}维向量表示

### 输出文件说明
- `user_dimension_vectors.csv`: 完整的用户向量数据（包含关键词和情感）
- `user_vector_matrix.csv`: 简化的向量矩阵（仅强度值）
- `dimension_insights.json`: 详细的维度洞察分析数据
- `analysis_summary.md`: 本分析总结报告

### 后续应用方向
1. **个性化推荐**: 基于用户向量进行车型推荐
2. **市场细分**: 识别不同的用户群体特征
3. **产品改进**: 了解用户对各维度的关注程度
4. **营销策略**: 针对不同用户画像制定差异化策略

---
*报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report
    
    def run_extraction_pipeline(self, sample_size: int = None, data_source: str = None) -> bool:
        """运行完整的用户向量提取流程"""
        print("=" * 80)
        print("新版用户画像系统 - 用户向量提取流程（优化版）")
        print("=" * 80)
        
        try:
            # 1. 加载数据
            if data_source is None:
                data_source = r"D:\code\PRP\Cleaned\cleaned_comments.csv"
            
            df = self.load_comment_data(data_source)
            if df.empty:
                print("数据加载失败，流程终止")
                return False
            
            # 2. 数据采样（如果需要）
            if sample_size and sample_size > 0 and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                print(f"随机采样 {sample_size} 条评论进行分析")
            else:
                print(f"处理全部 {len(df)} 条评论")
            
            # 3. 提取评论内容
            if 'cleaned_comment' in df.columns:
                comments = df['cleaned_comment'].tolist()
            elif 'original_comment' in df.columns:
                comments = df['original_comment'].tolist()
            else:
                print("未找到评论内容列，流程终止")
                return False
            
            # 4. LLM维度标签提取（使用硬编码批大小，不依赖环境变量）
            batch_size = 12
            dimension_results = self.extract_dimension_tags_batch(comments, batch_size=batch_size)
            
            if not dimension_results:
                print("维度标签提取失败，流程终止")
                return False
            
            # 5. 构建用户向量
            vector_df = self.build_dimension_vectors(df, dimension_results)
            
            # 6. 分析维度洞察
            insights = self.analyze_dimension_insights(vector_df)
            
            # 7. 保存结果
            self.save_results(vector_df, insights, sample_size)
            
            print("\n" + "=" * 80)
            print("用户向量提取流程完成！")
            print(f"所有结果已保存到: {self.output_dir}")
            print("核心产出:")
            print(f"   - 用户向量矩阵: {len(vector_df)} 个用户 × {len(self.persona_dimensions)} 个维度")
            print(f"   - 维度洞察分析: {len(insights['user_profiles'])} 种用户画像类型")
            print(f"   - 车型偏好分析: {len(insights['car_model_analysis'])} 个车型")
            print("=" * 80)
            
            return True
            
        except Exception as e:
            print(f"流程执行失败: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    print("新版用户画像系统 - 用户向量提取")
    print("=" * 60)
    
    # 创建提取器
    extractor = UserVectorExtractor()
    
    # 运行提取流程（不从环境变量读取采样大小）
    sample_size = None  # 处理全部数据
    
    success = extractor.run_extraction_pipeline(
        sample_size=sample_size,  # None表示处理全部数据
        data_source=r"D:\code\PRP\Cleaned\cleaned_comments.csv"
    )
    
    if success:
        print("\n用户向量提取成功完成！")
        print("下一步可以:")
        print("   1. 查看生成的向量数据进行分析")
        print("   2. 使用向量数据训练推荐模型")
        print("   3. 与Model_New的特征分析结果进行对比")
    else:
        print("\n用户向量提取失败，请检查错误信息")

if __name__ == "__main__":
    main()