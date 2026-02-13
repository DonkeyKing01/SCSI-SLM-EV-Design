# SCSI-SLM: Consumer Voice to Engineering Insight

**Official Implementation of the Framework**

## Overview

This repository contains the source code and datasets for the paper: **"Mapping Consumer Voice into Engineering Insight: A Structured Language Model-Driven Design Support Framework for Electric Vehicle"**.

The framework (SCSI-SLM) bridges the semantic gap between unstructured consumer reviews and formal engineering design parameters through a three-stage pipeline.

## Project Structure

The repository is organized into modules corresponding to the methodological steps described in the manuscript:

| Directory | Module Name | Manuscript Section | Description |
| :--- | :--- | :--- | :--- |
| **00_Raw_Data/** | Data Acquisition | Sec 3.1.1 | Raw consumer review datasets for 50 EV models. |
| **01_SSE_Analysis/** | Structured Semantic Encoding (SSE) | Sec 3.1 | Pipeline for cleaning, tag extraction, and dimension mapping. |
| **02_User_Modeling/** | Product-User Dynamic Mapping (PUDM) | Sec 3.2 | Algorithms for IPA (Product-side) and Persona Clustering (User-side). |
| **03_Knowledge_Graph/** | Engineering Design KG (EDKG) | Sec 3.3 | Schema definition and graph construction scripts for Neo4j. |
| **04_RAG_App/** | Hybrid Retrieval Engine | Sec 4.3 | The online RAG application with Chain-of-Thought reasoning. |

## Quick Start

To run the demonstration system (RAG Engine):

1. Navigate to the `04_RAG_App/` directory.
2. Follow the instructions in `04_RAG_App/README.md`.
3. The system utilizes pre-computed outputs from Modules 01-03 to ensure immediate deployability.

## Citation

If you find this code useful, please cite our paper:

*(Citation placeholder)*

---

## 项目概述 (Overview in Chinese)

本仓库包含论文 **"Mapping Consumer Voice into Engineering Insight: A Structured Language Model-Driven Design Support Framework for Electric Vehicle"** 的源代码与数据集。

SCSI-SLM 框架通过三阶段流程，弥合了非结构化消费者评论与形式化工程设计参数之间的语义鸿沟。

## 项目结构 (Project Structure)

本仓库根据论文中描述的方法论步骤划分为以下模块：

| 目录 | 模块名称 | 论文章节 | 说明 |
| :--- | :--- | :--- | :--- |
| **00_Raw_Data/** | 数据获取 | Sec 3.1.1 | 50款电动车型的原始用户评论数据集。 |
| **01_SSE_Analysis/** | 结构化语义编码 (SSE) | Sec 3.1 | 数据清洗、标签提取及维度映射流程。 |
| **02_User_Modeling/** | 产品-用户动态映射 (PUDM) | Sec 3.2 | 用于产品端IPA分析和用户端画像聚类的算法。 |
| **03_Knowledge_Graph/** | 工程设计知识图谱 (EDKG) | Sec 3.3 | Neo4j 图谱的Schema定义与构建脚本。 |
| **04_RAG_App/** | 混合检索引擎 | Sec 4.3 | 具备思维链推理能力的在线RAG应用。 |

## 快速开始 (Quick Start)

如需运行演示系统（RAG 引擎）：

1. 进入 `04_RAG_App/` 目录。
2. 按照 `04_RAG_App/README.md` 中的说明操作。
3. 该系统使用模块 01-03 预计算的输出结果，以确保可立即部署测试。