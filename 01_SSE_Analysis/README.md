# Module 01: Structured Semantic Encoding (SSE)

## Manuscript Reference
This module implements **Section 3.1**, transforming unstructured text into engineering semantic tokens.

## Pipeline Stages
1. **Data Preprocessing**: Regex denoising and short-text filtering.
2. **LLM Tag Extraction**: Extracting "Raw Feature Tags" from user narratives using Large Language Models.
3. **Expert Calibration**: Mapping raw tags to **8 Core Engineering Dimensions** (Table 2 in paper).

## Key Files
* `1_Data_Preprocessing/cleaning_pipeline.py`: Data cleaning and preprocessing script.
* `2_Dimension_Construction/tag_extraction_refinement.py`: LLM-based tag extraction and refinement.
* `2_Dimension_Construction/dimension_mapping.json`: **[Critical]** The expert-calibrated dictionary mapping user vocabulary to engineering dimensions.
* `1_Data_Preprocessing/outputs/`: Contains sample processed data.
    * *Note: `cleaned_comments.csv` is provided as a compressed .zip file to reduce repository size.*

## Input & Output
* **Input**: Raw CSVs from `../00_Raw_Data/`.
* **Output**: Structured datasets with `Refined_Tag`, `Sentiment_Score`, and `Engineering_Dimension` columns.

---

# 模块 01：结构化语义编码 (SSE)

## 论文对应
本模块实现了论文 **3.1 节** 的方法，将非结构化文本转化为工程语义 Token。

## 流程阶段
1. **数据预处理**: 正则去噪与短文本过滤。
2. **LLM标签提取**: 利用大语言模型从用户叙述中提取“原始特征标签”。
3. **专家校准**: 将原始标签映射至论文表2定义的 **8个核心工程维度**。

## 核心文件
* `main_pipeline.py`: 执行完整清洗流程的总控脚本。
* `2_Dimension_Construction/dimension_mapping.json`: **[关键文件]** 经过专家校准的字典，用于将用户词汇映射到工程维度。
* `outputs/`: 包含处理后的样本数据。
    * *注：为减小仓库体积，`cleaned_comments.csv` 以 .zip 压缩包形式提供。*

## 输入与输出
* **输入**: 来自 `../00_Raw_Data/` 的原始 CSV 文件。
* **输出**: 包含 `Refined_Tag`（优化标签）、`Sentiment_Score`（情感得分）和 `Engineering_Dimension`（工程维度）列的结构化数据集。