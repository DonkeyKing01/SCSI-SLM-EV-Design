import os
import re
import sys
import pandas as pd
import jieba

# 尝试将标准输出设置为UTF-8，避免Windows控制台编码问题
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

DATA_DIR = "../../00_Raw_Data"
OUTPUT_DIR = "./outputs"


def clean_text(text: str) -> str:
    if pd.isna(text) or text == "":
        return ""

    text = str(text)
    # 移除URL
    text = re.sub(r'http[s]?/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # 保留中英文与数字及常见中文标点，将其它替换为空格
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？；：""''（）【】]', ' ', text)
    # 压缩多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(words):
    stopwords = set([
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '里', '比', '这个', '来', '个', '或', '可以', '但是', '这样', '还是', '什么', '这么', '只是', '觉得', '还', '他', '她', '它', '用', '让', '像', '而且', '如果', '所以', '因为', '虽然', '但', '从', '把', '为', '被', '给', '对', '于', '以', '及', '与', '或', '而', '又', '却', '只', '就', '才', '更', '最', '非常', '特别', '确实', '真的', '实在', '比较', '相当', '挺', '蛮', '太', '还有', '另外', '其实', '当然', '应该', '可能', '大概', '也许', '或许', '估计', '差不多', '基本上', '一般', '通常', '经常', '总是', '永远', '从来', '几乎', '完全', '根本', '简直', '十分', '尤其', '格外', '更加'
    ])
    return [w for w in words if w not in stopwords and len(w) > 1]


def discover_comment_column(df: pd.DataFrame) -> str | None:
    candidates = ['评价内容', '评论内容', '内容', '评价', '评论', 'cleaned_content', 'cleaned_comment', 'original_comment', 'content', 'comment']
    for col in candidates:
        if col in df.columns:
            return col
    return None


def process_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows = []
    processed_files = 0

    for filename in os.listdir(DATA_DIR):
        if not filename.endswith('.xlsx') or filename.startswith('~$'):
            continue

        file_path = os.path.join(DATA_DIR, filename)
        car_model = filename.replace('.xlsx', '')

        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            print(f"❌ 无法读取: {filename}: {e}")
            continue

        comment_col = discover_comment_column(df)
        if not comment_col:
            print(f"⚠️ 跳过（未找到评价列）: {filename}")
            continue

        df = df.dropna(subset=[comment_col])
        df[comment_col] = df[comment_col].astype(str)

        cleaned_records = []
        for text in df[comment_col].tolist():
            cleaned = clean_text(text)
            if len(cleaned) < 10:
                continue
            tokens = remove_stopwords(list(jieba.cut(cleaned)))
            if len(tokens) < 3:
                continue
            record = {
                'car_model': car_model,
                'original_comment': text,
                'cleaned_comment': cleaned,
                'words': ' '.join(tokens),
                'word_list': tokens,
            }
            cleaned_records.append(record)
            all_rows.append(record)

        # 保存车型级清洗结果
        if cleaned_records:
            per_model_path = os.path.join(OUTPUT_DIR, f"{car_model}.csv")
            pd.DataFrame(cleaned_records).to_csv(per_model_path, index=False, encoding='utf-8-sig')
            processed_files += 1
            print(f"已处理 {car_model}: {len(cleaned_records)} 条")
        else:
            print(f"警告 {car_model}: 无有效评论")

    # 保存汇总文件
    if all_rows:
        agg_path = os.path.join(OUTPUT_DIR, 'cleaned_comments.csv')
        pd.DataFrame(all_rows).to_csv(agg_path, index=False, encoding='utf-8-sig')
        print(f"\n已保存汇总: {agg_path} （{len(all_rows)} 条，{processed_files} 个车型）")
    else:
        print("错误：未生成任何清洗数据，请检查源数据目录与列名")


if __name__ == '__main__':
    print(f"源数据: {DATA_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    process_all()

