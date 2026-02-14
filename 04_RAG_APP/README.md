# Module 04: Hybrid Retrieval-Reasoning System

## Manuscript Reference

This module implements the online application described in **Section 4.3: Hybrid Retrieval-Reasoning Engine**. It integrates the structured knowledge generated in Modules 01-03 to provide an interactive decision support interface for engineering design.

## System Architecture

The system employs a "Dual-Channel Retrieval" mechanism followed by "Chain-of-Thought (CoT)" reasoning, consisting of three main components:

1. **Intent Analysis & Routing**:
   Classifies user queries into "Specific Fact Retrieval" or "Open-ended Reasoning" (corresponding to Fig. 4 in the manuscript) to select the optimal retrieval strategy.

2. **Hybrid Retrieval**:
   * **Vector Path**: Utilizes embedding similarity to retrieve relevant "User Personas" and unstructured "Review Evidence".
   * **Graph Path**: Executes Cypher queries on the Neo4j Engineering Design Knowledge Graph (EDKG) to traverse explicit relationships (e.g., `(Car)-[PERFORMS_ON]->(Feature)`).

3. **Reasoning Generation**:
   Synthesizes the retrieved multi-modal evidence using a Large Language Model (LLM) to generate evidence-backed engineering insights.

## Key Files

* `app.py`: Main Streamlit application with web interface.
* `run.py`: Alternative entry point with environment validation.
* `load_vector_data.py`: Script to load vector data into ChromaDB.
* `core/rag_engine.py`: Core RAG engine implementation.
* `core/question_analyzer.py`: Question classification and intent analysis.
* `tools/`: Vector search, graph query, and hybrid retrieval tools.
* `database/neo4j_connection.py`: Neo4j database connection manager.
* `config/settings.py`: Configuration management (loads from root `.env`).

## Prerequisites

* **Docker & Docker Compose**: Recommended for deploying the Neo4j graph database.
* **Python 3.8+**: Required for running the application backend.
* **API Keys**: OpenAI or compatible LLM service API key.

## Configuration

**Important**: This module loads configuration from the **project root directory's `.env` file**.

1. Ensure you have created `.env` file in the project root directory (not in this folder).
2. Copy and configure from root `.env.example`:
   ```bash
   cd ..  # Navigate to project root
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

3. Required environment variables:
   * `OPENAI_API_KEY`: Your LLM provider API key
   * `NEO4J_URI`: Neo4j connection URI (default: `bolt://localhost:7688`)
   * `NEO4J_USERNAME`: Neo4j username (default: `neo4j`)
   * `NEO4J_PASSWORD`: Neo4j password
   * `NEO4J_DATABASE`: Database name (default: `neo4jfinal`)

## Quick Start (Local Execution)

### Step 1: Start Knowledge Graph (Neo4j)

If you do not have a local Neo4j instance running, use the provided Docker Compose file:

```bash
docker-compose up -d neo4j
```

Wait for Neo4j to fully start (check logs with `docker-compose logs -f neo4j`).

### Step 2: Install Dependencies

Install the required Python packages from the **project root directory**:

```bash
cd ..  # Navigate to project root
pip install -r requirements.txt
```

### Step 3: Load Vector Data (First Time Only)

If you need to initialize the vector database with embeddings:

```bash
cd 04_RAG_APP
python load_vector_data.py
```

This will create the ChromaDB vector store in `./vector_store/`.

### Step 4: Launch Application

Start the web interface using Streamlit:

```bash
streamlit run app.py
```

Or use the alternative runner with environment validation:

```bash
python run.py
```

The application will be accessible at: **http://localhost:8501**

## Usage Examples

Once the application is running, you can ask questions such as:

* **Fact Retrieval**: "小米SU7的续航表现如何?" (How is the range performance of Xiaomi SU7?)
* **Comparative Analysis**: "比较特斯拉Model 3和小鹏P7的智能配置" (Compare smart features of Tesla Model 3 vs XPeng P7)
* **User Persona Query**: "性价比导向用户更关注哪些车型特征?" (Which features do value-oriented users care about?)

## Troubleshooting

**Neo4j Connection Failed**:
- Verify Neo4j is running: `docker ps | grep neo4j`
- Check connection settings in root `.env` file
- Ensure the database name matches (default: `neo4jfinal`)

**Missing API Key Error**:
- Confirm `OPENAI_API_KEY` is set in root `.env` file
- Verify `.env` file is in the project root, not in this folder

**Vector Store Not Found**:
- Run `python load_vector_data.py` to initialize the vector database

---

# 模块 04：混合检索推理系统

## 论文对应

本模块实现了论文 **4.3 节：混合检索推理引擎** 中描述的在线应用程序。它集成了模块 01-03 生成的结构化知识，为工程设计提供交互式的决策支持界面。

## 系统架构

本系统采用"双通道检索"机制，并结合"思维链 (CoT)"推理，主要包含三个核心组件：

1. **意图分析与路由**：
   将用户查询分类为"特定事实检索"或"开放式推理"（对应论文图4），以选择最优的检索策略。

2. **混合检索**：
   * **向量通道**：利用嵌入相似度检索相关的"用户画像"和非结构化"评论证据"。
   * **图通道**：在 Neo4j 工程设计知识图谱 (EDKG) 上执行 Cypher 查询，遍历显式关系（如 `(车型)-[表现于]->(特征)`）。

3. **推理生成**：
   利用大语言模型 (LLM) 综合检索到的多模态证据，生成具有证据支撑的工程洞察。

## 核心文件

* `app.py`: 主 Streamlit 应用程序，提供 Web 界面。
* `run.py`: 备用入口点，包含环境验证功能。
* `load_vector_data.py`: 将向量数据加载到 ChromaDB 的脚本。
* `core/rag_engine.py`: 核心 RAG 引擎实现。
* `core/question_analyzer.py`: 问题分类与意图分析。
* `tools/`: 向量搜索、图查询和混合检索工具。
* `database/neo4j_connection.py`: Neo4j 数据库连接管理器。
* `config/settings.py`: 配置管理（从根目录 `.env` 加载）。

## 前置要求

* **Docker & Docker Compose**: 推荐用于部署 Neo4j 图数据库。
* **Python 3.8+**: 运行应用程序后端所需。
* **API 密钥**: OpenAI 或兼容的 LLM 服务 API 密钥。

## 配置说明

**重要提示**：本模块从**项目根目录的 `.env` 文件**加载配置。

1. 确保在项目根目录（不是本文件夹）创建了 `.env` 文件。
2. 从根目录的 `.env.example` 复制并配置：
   ```bash
   cd ..  # 进入项目根目录
   cp .env.example .env
   # 编辑 .env 填入真实的 API 密钥
   ```

3. 必需的环境变量：
   * `OPENAI_API_KEY`: LLM 提供商的 API 密钥
   * `NEO4J_URI`: Neo4j 连接 URI（默认：`bolt://localhost:7688`）
   * `NEO4J_USERNAME`: Neo4j 用户名（默认：`neo4j`）
   * `NEO4J_PASSWORD`: Neo4j 密码
   * `NEO4J_DATABASE`: 数据库名称（默认：`neo4jfinal`）

## 快速开始（本地执行）

### 步骤 1：启动知识图谱 (Neo4j)

如果您没有本地 Neo4j 实例运行，使用提供的 Docker Compose 文件：

```bash
docker-compose up -d neo4j
```

等待 Neo4j 完全启动（使用 `docker-compose logs -f neo4j` 检查日志）。

### 步骤 2：安装依赖

从**项目根目录**安装所需的 Python 包：

```bash
cd ..  # 进入项目根目录
pip install -r requirements.txt
```

### 步骤 3：加载向量数据（仅首次）

如果需要初始化向量数据库：

```bash
cd 04_RAG_APP
python load_vector_data.py
```

这将在 `./vector_store/` 中创建 ChromaDB 向量存储。

### 步骤 4：启动应用

使用 Streamlit 启动 Web 界面：

```bash
streamlit run app.py
```

或使用带环境验证的备用启动器：

```bash
python run.py
```

应用程序将在以下地址访问：**http://localhost:8501**

## 使用示例

应用程序运行后，您可以提出以下问题：

* **事实检索**: "小米SU7的续航表现如何?"
* **对比分析**: "比较特斯拉Model 3和小鹏P7的智能配置"
* **用户画像查询**: "性价比导向用户更关注哪些车型特征?"

## 故障排查

**Neo4j 连接失败**：
- 验证 Neo4j 正在运行：`docker ps | grep neo4j`
- 检查根目录 `.env` 文件中的连接设置
- 确保数据库名称匹配（默认：`neo4jfinal`）

**缺少 API 密钥错误**：
- 确认在根目录 `.env` 文件中设置了 `OPENAI_API_KEY`
- 验证 `.env` 文件在项目根目录，而不是本文件夹中

**未找到向量存储**：
- 运行 `python load_vector_data.py` 初始化向量数据库
