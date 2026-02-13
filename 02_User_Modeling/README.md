# Module 02: Product-User Dynamic Mapping (PUDM)

## Manuscript Reference
This module implements **Section 3.2**, decoupling the corpus into Product-side and User-side analytical streams.

## Analytical Streams

### 1. Product-Side: IPA Quantification
* **Directory**: `Product_IPA_Analysis/`
* **Script**: `ipa_quantification.py`
* **Logic**: Calculates Sentiment (Performance) and Mention Rate (Importance) to construct the Decision Matrix (Fig. 5).

### 2. User-Side: Preference Profiling
* **Directory**: `User_Preference_Clustering/`
* **Script**: `preference_profiling.py`
* **Visualization**: `persona_visualization.py`
* **Logic**: Vectorizes user concern distributions and applies **K-Means++ Clustering** to identify typical User Personas (Fig. 6).

## Pre-computed Outputs for RAG
To facilitate the immediate execution of the RAG System (`04_RAG_App`), we provide the computed results in the `outputs/` folders:
* **`car_model_scores.csv`**: The IPA metrics for all 50 models.
* **`user_vector_matrix.csv`**: The vectorized user profiles used for retrieval.
* **`persona_visualization.png`**: Visualization of the clustering results.

---

# 模块 02：产品-用户动态映射 (PUDM)

## 论文对应
本模块实现了论文 **3.2 节** 的方法，将语料库解耦为“产品端”和“用户端”两个分析流。

## 分析流

### 1. 产品端：IPA量化
* **目录**: `Product_IPA_Analysis/`
* **脚本**: `ipa_quantification.py`
* **逻辑**: 计算情感得分（表现）和提及率（重要性），构建决策矩阵（对应论文图5）。

### 2. 用户端：偏好画像
* **目录**: `User_Preference_Clustering/`
* **脚本**: `preference_profiling.py`
* **可视化**: `persona_visualization.py`
* **逻辑**: 对用户关注分布进行向量化，并应用 **K-Means++ 聚类** 识别典型用户画像（对应论文图6）。

## RAG预计算输出
为便于直接运行 RAG 系统 (`04_RAG_App`)，我们在 `outputs/` 文件夹中提供了计算结果：
* **`car_model_scores.csv`**: 所有 50 款车型的 IPA 指标数据。
* **`user_vector_matrix.csv`**: 用于检索的用户画像向量数据。
* **`persona_visualization.png`**: 聚类结果的可视化图表。