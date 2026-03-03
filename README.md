# SCSI-SLM: Consumer Voice to Engineering Insight

**Official Implementation of the Framework**

> **Paper Status:** Under proof. DOI: [https://doi.org/10.1080/09544828.2026.2639933](https://10.1080/09544828.2026.2639933) *(link will be active upon publication)*

---

## Overview

This repository contains the source code and datasets for the paper:

> **"Mapping Consumer Voice into Engineering Insight: A Structured Language Model-Driven Design Support Framework for Electric Vehicle"**

The framework (SCSI-SLM) bridges the semantic gap between unstructured consumer reviews and formal engineering design parameters through a three-stage pipeline.

### Framework Architecture

![SCSI-SLM Overall Framework Architecture](docs/images/fig_framework_overview.png)

*Figure 1: Overall architecture of the SCSI-SLM framework, illustrating the three-stage pipeline from raw consumer reviews to engineering design insights.*

---

## Gallery

> **Note:** The figures below are representative outputs from the paper. Please place the corresponding image files in the `docs/images/` directory.

### IPA Analysis — Importance-Performance Matrix

![IPA Analysis Example](docs/images/fig_ipa_analysis.png)

*Figure 2: Importance-Performance Analysis (IPA) matrix for selected EV models, identifying key engineering design priorities.*

### User Preference Clustering — Persona Visualization

![User Preference Clustering](docs/images/fig_user_clustering.png)

*Figure 3: User preference clustering results visualizing distinct consumer personas and their feature priorities.*

### RAG Application — Interactive Interface

![RAG Application Interface](docs/images/fig_rag_interface.png)

*Figure 4: Screenshot of the RAG-powered interactive design support interface.*

---

## Project Structure

The repository is organized into modules corresponding to the methodological steps described in the manuscript:

| Directory | Module Name | Manuscript Section | Description |
| :--- | :--- | :--- | :--- |
| **00_Raw_Data/** | Data Acquisition | Sec 3.1.1 | Raw consumer review datasets for 50 EV models. |
| **01_SSE_Analysis/** | Structured Semantic Encoding (SSE) | Sec 3.1 | Pipeline for cleaning, tag extraction, and dimension mapping. |
| **02_User_Modeling/** | Product-User Dynamic Mapping (PUDM) | Sec 3.2 | Algorithms for IPA (Product-side) and Persona Clustering (User-side). |
| **03_Knowledge_Graph/** | Engineering Design KG (EDKG) | Sec 3.3 | Schema definition and graph construction scripts for Neo4j. |
| **04_RAG_APP/** | Hybrid Retrieval Engine | Sec 4.3 | The online RAG application with Chain-of-Thought reasoning. |

---

## Quick Start

To run the demonstration system (RAG Engine):

1. Navigate to the `04_RAG_APP/` directory.
2. Follow the instructions in `04_RAG_APP/README.md`.
3. The system utilizes pre-computed outputs from Modules 01–03 to ensure immediate deployability.

---

## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{Jin2026Mapping,
  title   = {Mapping Consumer Voice into Engineering Insight: A Structured Language Model-Driven Design Support Framework for Electric Vehicle},
  author  = {Qingyang Jin and Luyao Wang and Wenyu Yuan and Danni Chang},
  journal = {Journal of Engineering Design},
  year    = {2026},
  doi     = {10.1080/09544828.2026.2639933},
  note    = {In Press}
}
}
```

> The paper is currently in the proof stage. The DOI [https://doi.org/10.1080/09544828.2026.2639933](https://doi.org/10.1080/09544828.2026.2639933) will become active upon official publication.

---

## 项目概述

本仓库包含论文 **"Mapping Consumer Voice into Engineering Insight: A Structured Language Model-Driven Design Support Framework for Electric Vehicle"** 的源代码与数据集。

> **论文状态：** 正在校对阶段（proof）。DOI：[https://doi.org/10.1080/09544828.2026.2639933](https://doi.org/10.1080/09544828.2026.2639933) *(正式发表后链接将生效)*

SCSI-SLM 框架通过三阶段流程，弥合了非结构化消费者评论与形式化工程设计参数之间的语义鸿沟。

### 框架总体结构

![SCSI-SLM 框架总体结构](docs/images/fig_framework_overview.png)

*图1：SCSI-SLM 框架总体结构，展示了从原始消费者评论到工程设计洞察的三阶段流程。*

---

## 示例图集

> **说明：** 以下图片为论文中的代表性输出图。请将对应图片文件放置于 `docs/images/` 目录下。

### IPA 分析 — 重要性-表现矩阵

![IPA 分析示例](docs/images/fig_ipa_analysis.png)

*图2：针对典型电动车型的 IPA 矩阵分析，用于识别关键工程设计优先级。*

### 用户偏好聚类 — 用户画像可视化

![用户偏好聚类](docs/images/fig_user_clustering.png)

*图3：用户偏好聚类结果，可视化了不同消费者画像及其功能偏好差异。*

### RAG 应用 — 交互界面

![RAG 应用交互界面](docs/images/fig_rag_interface.png)

*图4：基于 RAG 的交互式设计支持系统界面截图。*

---

## 项目结构

本仓库根据论文中描述的方法论步骤划分为以下模块：

| 目录 | 模块名称 | 论文章节 | 说明 |
| :--- | :--- | :--- | :--- |
| **00_Raw_Data/** | 数据获取 | Sec 3.1.1 | 50款电动车型的原始用户评论数据集。 |
| **01_SSE_Analysis/** | 结构化语义编码 (SSE) | Sec 3.1 | 数据清洗、标签提取及维度映射流程。 |
| **02_User_Modeling/** | 产品-用户动态映射 (PUDM) | Sec 3.2 | 用于产品端IPA分析和用户端画像聚类的算法。 |
| **03_Knowledge_Graph/** | 工程设计知识图谱 (EDKG) | Sec 3.3 | Neo4j 图谱的Schema定义与构建脚本。 |
| **04_RAG_APP/** | 混合检索引擎 | Sec 4.3 | 具备思维链推理能力的在线RAG应用。 |

---

## 快速开始

如需运行演示系统（RAG 引擎）：

1. 进入 `04_RAG_APP/` 目录。
2. 按照 `04_RAG_APP/README.md` 中的说明操作。
3. 该系统使用模块 01-03 预计算的输出结果，以确保可立即部署测试。

---

## 引用

如果本代码对您的研究有所帮助，请引用我们的论文：

```bibtex
@article{Jin2026Mapping,
  title   = {Mapping Consumer Voice into Engineering Insight: A Structured Language Model-Driven Design Support Framework for Electric Vehicle},
  author  = {Qingyang Jin and Luyao Wang and Wenyu Yuan and Danni Chang},
  journal = {Journal of Engineering Design},
  year    = {2026},
  doi     = {10.1080/09544828.2026.2639933},
  note    = {In Press}
}
```

> 论文目前处于校对阶段，DOI [https://doi.org/10.1080/09544828.2026.2639933](https://doi.org/10.1080/09544828.2026.2639933) 将在正式发表后激活。