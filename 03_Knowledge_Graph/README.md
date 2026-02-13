# Module 03: Engineering Design Knowledge Graph (EDKG)

## Manuscript Reference
This module corresponds to **Section 3.3**, constructing the graph database that connects Users, Products, and Features.

## Graph Schema
The Neo4j graph ontology consists of the following structure:
* **Nodes**: `CarModel`, `UserPersona`, `EngineeringFeature`, `ReviewEvidence`.
* **Relationships**:
    * `(User)-[PREFERS]->(Feature)`
    * `(Car)-[PERFORMS_ON]->(Feature)`
    * `(Review)-[MENTIONS]->(Feature)`

## Contents
* `main.py`: Main script to build the knowledge graph from CSV data.
* `src/knowledge_graph_builder.py`: Core graph construction logic.
* `src/user_clustering.py`: User persona clustering integration.
* `src/data_manager.py`: Data loading and management utilities.
* `scripts/`: Cypher scripts for database initialization.
* `examples/rag_queries.py`: Sample query templates used for graph reasoning.

---

# 模块 03：工程设计知识图谱 (EDKG)

## 论文对应
本模块对应论文 **3.3 节**，构建连接用户、产品和特征的图数据库。

## 图谱模式
Neo4j 图谱本体包含以下结构：
* **节点 (Nodes)**: `CarModel` (车型), `UserPersona` (用户画像), `EngineeringFeature` (工程特征), `ReviewEvidence` (评论证据)。
* **关系 (Relationships)**:
    * `(User)-[PREFERS]->(Feature)` (用户偏好特征)
    * `(Car)-[PERFORMS_ON]->(Feature)` (车型表现特征)
    * `(Review)-[MENTIONS]->(Feature)` (评论提及特征)

## 目录说明
* `main.py`: 从 CSV 数据构建知识图谱的主脚本。
* `src/knowledge_graph_builder.py`: 核心图谱构建逻辑。
* `src/user_clustering.py`: 用户画像聚类集成。
* `src/data_manager.py`: 数据加载与管理工具。
* `scripts/`: 用于数据库初始化的 Cypher 脚本。
* `examples/rag_queries.py`: 用于图谱推理的示例查询模板。