# Module 04: Hybrid Retrieval-Reasoning System# Module 04: Hybrid Retrieval-Reasoning System# Module 04: Hybrid Retrieval-Reasoning System



## Manuscript Reference

This module implements the online application described in **Section 4.3: Hybrid Retrieval-Reasoning Engine**. It integrates the structured knowledge generated in Modules 01-03 to provide an interactive decision support interface for engineering design.

## Manuscript Reference## Manuscript Reference

## System Architecture

The system employs a "Dual-Channel Retrieval" mechanism followed by "Chain-of-Thought (CoT)" reasoning, consisting of three main components:This module implements the online application described in **Section 4.3: Hybrid Retrieval-Reasoning Engine**. It integrates the structured knowledge generated in Modules 01-03 to provide an interactive decision support interface for engineering design.This module implements the online application described in **Section 4.3: Hybrid Retrieval-Reasoning Engine**. It integrates the structured knowledge generated in Modules 01-03 to provide an interactive decision support interface for engineering design.



1. **Intent Analysis & Routing**:

   Classifies user queries into "Specific Fact Retrieval" or "Open-ended Reasoning" (corresponding to Fig. 4 in the manuscript) to select the optimal retrieval strategy.

## System Architecture## System Architecture

2. **Hybrid Retrieval**:

   * **Vector Path**: Utilizes embedding similarity to retrieve relevant "User Personas" and unstructured "Review Evidence".The system employs a "Dual-Channel Retrieval" mechanism followed by "Chain-of-Thought (CoT)" reasoning, consisting of three main components:The system employs a "Dual-Channel Retrieval" mechanism followed by "Chain-of-Thought (CoT)" reasoning, consisting of three main components:

   * **Graph Path**: Executes Cypher queries on the Neo4j Engineering Design Knowledge Graph (EDKG) to traverse explicit relationships (e.g., `(Car)-[PERFORMS_ON]->(Feature)`).



3. **Reasoning Generation**:

   Synthesizes the retrieved multi-modal evidence using a Large Language Model (LLM) to generate evidence-backed engineering insights.1. **Intent Analysis & Routing**:1.  **Intent Analysis & Routing**:



## Key Files   Classifies user queries into "Specific Fact Retrieval" or "Open-ended Reasoning" (corresponding to Fig. 4 in the manuscript) to select the optimal retrieval strategy.    Classifies user queries into "Specific Fact Retrieval" or "Open-ended Reasoning" (corresponding to Fig. 4 in the manuscript) to select the optimal retrieval strategy.

* `app.py`: Main Streamlit application with web interface.

* `run.py`: Alternative entry point with data checking utilities.

* `load_vector_data.py`: Script to load vector data into ChromaDB.

* `core/rag_engine.py`: Core RAG engine implementation.2. **Hybrid Retrieval**:2.  **Hybrid Retrieval**:

* `core/question_analyzer.py`: Question classification and intent analysis.

* `tools/`: Vector search, graph query, and hybrid retrieval tools.   * **Vector Path**: Utilizes embedding similarity to retrieve relevant "User Personas" and unstructured "Review Evidence".    * **Vector Path**: Utilizes embedding similarity to retrieve relevant "User Personas" and unstructured "Review Evidence".

* `database/neo4j_connection.py`: Neo4j database connection manager.

   * **Graph Path**: Executes Cypher queries on the Neo4j Engineering Design Knowledge Graph (EDKG) to traverse explicit relationships (e.g., `(Car)-[PERFORMS_ON]->(Feature)`).    * **Graph Path**: Executes Cypher queries on the Neo4j Engineering Design Knowledge Graph (EDKG) to traverse explicit relationships (e.g., `(Car)-[PERFORMS_ON]->(Feature)`).

## Prerequisites

* **Docker & Docker Compose**: Recommended for deploying the Neo4j graph database.

* **Python 3.8+**: Required for running the application backend.

* **OpenAI API Key**: Required for the LLM inference engine.3. **Reasoning Generation**:3.  **Reasoning Generation**:



## Configuration   Synthesizes the retrieved multi-modal evidence using a Large Language Model (LLM) to generate evidence-backed engineering insights.    Synthesizes the retrieved multi-modal evidence using a Large Language Model (LLM) to generate evidence-backed engineering insights.

1. Navigate to the `04_RAG_APP/` directory.

2. Copy the example configuration file:

   ```bash

   cp .env.example .env## Prerequisites## Prerequisites

   ```

3. Edit the `.env` file and fill in the required configuration:* **Docker & Docker Compose**: Recommended for deploying the Neo4j graph database.* **Docker & Docker Compose**: Recommended for deploying the Neo4j graph database.

   * `OPENAI_API_KEY`: Your LLM provider API key.

   * `NEO4J_URI`: The connection URI for Neo4j (default: `bolt://localhost:7687`).* **Python 3.8+**: Required for running the application backend.* **Python 3.8+**: Required for running the application backend.

   * `NEO4J_PASSWORD`: The password set for your Neo4j instance.

* **OpenAI API Key**: Required for the LLM inference engine.* **OpenAI API Key**: Required for the LLM inference engine.

## Quick Start (Local Execution)



**Step 1: Start Knowledge Graph (Neo4j)**

## Configuration## Configuration

If you do not have a local Neo4j instance running, use the provided Docker Compose file to start one:

```bash1. Navigate to the `04_RAG_App/` directory.1.  Navigate to the `04_RAG_App/` directory.

docker-compose up -d neo4j

```2. Copy the example configuration file:2.  Copy the example configuration file:



**Step 2: Install Dependencies**   ```bash    ```bash



Install the required Python packages:   cp .env.example .env    cp .env.example .env

```bash

pip install -r requirements.txt   ```    ```

```

3. Edit the `.env` file and fill in the required configuration:3.  Edit the `.env` file and fill in the required configuration:

**Step 3: Load Vector Data (Optional)**

   * `OPENAI_API_KEY`: Your LLM provider API key.    * `OPENAI_API_KEY`: Your LLM provider API key.

If you need to reload vector data into ChromaDB:

```bash   * `NEO4J_URI`: The connection URI for Neo4j (default: `bolt://localhost:7687`).    * `NEO4J_URI`: The connection URI for Neo4j (default: `bolt://localhost:7687`).

python load_vector_data.py

```   * `NEO4J_PASSWORD`: The password set for your Neo4j instance.    * `NEO4J_PASSWORD`: The password set for your Neo4j instance.



**Step 4: Launch Application**



Start the web interface using Streamlit:## Quick Start (Local Execution)## Quick Start (Local Execution)

```bash

streamlit run app.py

```

**Step 1: Start Knowledge Graph (Neo4j)****Step 1: Start Knowledge Graph (Neo4j)**

Or use the alternative runner:

```bash

python run.py

```If you do not have a local Neo4j instance running, use the provided Docker Compose file to start one:If you do not have a local Neo4j instance running, use the provided Docker Compose file to start one:



The application will be accessible at: http://localhost:8501 (or the port specified in your console output).```bash```bash



---docker-compose up -d neo4jdocker-compose up -d neo4j



# 模块 04：混合检索推理系统``````



## 论文对应

本模块实现了论文 **4.3 节：混合检索推理引擎** 中描述的在线应用程序。它集成了模块 01-03 生成的结构化知识，为工程设计提供交互式的决策支持界面。

**Step 2: Install Dependencies****Step 2: Install Dependencies**

## 系统架构

本系统采用"双通道检索"机制，并结合"思维链 (CoT)"推理，主要包含三个核心组件：



1. **意图分析与路由**：Install the required Python packages:Install the required Python packages:

   将用户查询分类为"特定事实检索"或"开放式推理"（对应论文图4），以选择最优的检索策略。

```bash```bash

2. **混合检索**：

   * **向量通道**：利用嵌入相似度检索相关的"用户画像"和非结构化"评论证据"。pip install -r requirements.txtpip install -r requirements.txt

   * **图通道**：在 Neo4j 工程设计知识图谱 (EDKG) 上执行 Cypher 查询，遍历显式关系（如 `(车型)-[表现于]->(特征)`）。

``````

3. **推理生成**：

   利用大语言模型 (LLM) 综合检索到的多模态证据，生成具有证据支撑的工程洞察。



## 核心文件**Step 3: Initialize Environment****Step 3: Initialize Environment**

* `app.py`: 主 Streamlit 应用，包含 Web 界面。

* `run.py`: 备用启动脚本，包含数据检查工具。

* `load_vector_data.py`: 将向量数据加载到 ChromaDB 的脚本。

* `core/rag_engine.py`: 核心 RAG 引擎实现。Run the setup script to verify database connections and load necessary configurations:Run the setup script to verify database connections and load necessary configurations:

* `core/question_analyzer.py`: 问题分类与意图分析。

* `tools/`: 向量搜索、图查询和混合检索工具。```bash```bash

* `database/neo4j_connection.py`: Neo4j 数据库连接管理器。

python setup_env.pypython setup_env.py

## 先决条件

* **Docker & Docker Compose**：推荐用于部署 Neo4j 图数据库。``````

* **Python 3.8+**：用于运行应用程序后端。

* **OpenAI API Key**：用于 LLM 推理引擎。



## 配置步骤**Step 4: Launch Application****Step 4: Launch Application**

1. 进入 `04_RAG_APP/` 目录。

2. 复制示例配置文件：

   ```bash

   cp .env.example .envStart the web interface:Start the web interface:

   ```

3. 编辑 `.env` 文件并填入必要的配置信息：```bash```bash

   * `OPENAI_API_KEY`: 您的 LLM 服务 API 密钥。

   * `NEO4J_URI`: Neo4j 连接地址（默认为 `bolt://localhost:7687`）。python app.pypython app.py

   * `NEO4J_PASSWORD`: 您为 Neo4j 实例设置的密码。

``````

## 快速开始 (本地运行)



**步骤 1: 启动知识图谱 (Neo4j)**

The application will be accessible at: http://localhost:8501 (or the port specified in your console output).The application will be accessible at: http://localhost:8501 (or the port specified in your console output).

如果您没有正在运行的本地 Neo4j 实例，请使用提供的 Docker Compose 文件启动：

```bash

docker-compose up -d neo4j

```------



**步骤 2: 安装依赖**



安装所需的 Python 依赖包：# 模块 04：混合检索推理系统模块 04：混合检索推理系统

```bash

pip install -r requirements.txt论文对应

```

## 论文对应本模块实现了论文 4.3 节：混合检索推理引擎 中描述的在线应用程序。它集成了模块 01-03 生成的结构化知识，为工程设计提供交互式的决策支持界面。

**步骤 3: 加载向量数据 (可选)**

本模块实现了论文 **4.3 节：混合检索推理引擎** 中描述的在线应用程序。它集成了模块 01-03 生成的结构化知识，为工程设计提供交互式的决策支持界面。

如果需要重新加载向量数据到 ChromaDB：

```bash系统架构

python load_vector_data.py

```## 系统架构本系统采用“双通道检索”机制，并结合“思维链 (CoT)”推理，主要包含三个核心组件：



**步骤 4: 启动应用程序**本系统采用"双通道检索"机制，并结合"思维链 (CoT)"推理，主要包含三个核心组件：



使用 Streamlit 启动 Web 界面：意图分析与路由： 将用户查询分类为“特定事实检索”或“开放式推理”（对应论文图4），以选择最优的检索策略。

```bash

streamlit run app.py1. **意图分析与路由**：

```

   将用户查询分类为"特定事实检索"或"开放式推理"（对应论文图4），以选择最优的检索策略。混合检索：

或使用备用启动器：

```bash

python run.py

```2. **混合检索**：向量通道：利用嵌入相似度检索相关的“用户画像”和非结构化“评论证据”。



应用程序通常可通过以下地址访问：http://localhost:8501（或控制台输出中指定的端口）。   * **向量通道**：利用嵌入相似度检索相关的"用户画像"和非结构化"评论证据"。


   * **图通道**：在 Neo4j 工程设计知识图谱 (EDKG) 上执行 Cypher 查询，遍历显式关系（如 `(车型)-[表现于]->(特征)`）。图通道：在 Neo4j 工程设计知识图谱 (EDKG) 上执行 Cypher 查询，遍历显式关系（如 (车型)-[表现于]->(特征)）。



3. **推理生成**：推理生成： 利用大语言模型 (LLM) 综合检索到的多模态证据，生成具有证据支撑的工程洞察。

   利用大语言模型 (LLM) 综合检索到的多模态证据，生成具有证据支撑的工程洞察。

先决条件

## 先决条件Docker & Docker Compose：推荐用于部署 Neo4j 图数据库。

* **Docker & Docker Compose**：推荐用于部署 Neo4j 图数据库。

* **Python 3.8+**：用于运行应用程序后端。Python 3.8+：用于运行应用程序后端。

* **OpenAI API Key**：用于 LLM 推理引擎。

OpenAI API Key：用于 LLM 推理引擎。

## 配置步骤

1. 进入 `04_RAG_App/` 目录。配置步骤

2. 复制示例配置文件：进入 04_RAG_App/ 目录。

   ```bash

   cp .env.example .env复制示例配置文件：

   ```

3. 编辑 `.env` 文件并填入必要的配置信息：Bash

   * `OPENAI_API_KEY`: 您的 LLM 服务 API 密钥。cp .env.example .env

   * `NEO4J_URI`: Neo4j 连接地址（默认为 `bolt://localhost:7687`）。编辑 .env 文件并填入必要的配置信息：

   * `NEO4J_PASSWORD`: 您为 Neo4j 实例设置的密码。

OPENAI_API_KEY: 您的 LLM 服务 API 密钥。

## 快速开始 (本地运行)

NEO4J_URI: Neo4j 连接地址（默认为 bolt://localhost:7687）。

**步骤 1: 启动知识图谱 (Neo4j)**

NEO4J_PASSWORD: 您为 Neo4j 实例设置的密码。

如果您没有正在运行的本地 Neo4j 实例，请使用提供的 Docker Compose 文件启动：

```bash快速开始 (本地运行)

docker-compose up -d neo4j步骤 1: 启动知识图谱 (Neo4j) 如果您没有正在运行的本地 Neo4j 实例，请使用提供的 Docker Compose 文件启动：

```

Bash

**步骤 2: 安装依赖**docker-compose up -d neo4j

步骤 2: 安装依赖 安装所需的 Python 依赖包：

安装所需的 Python 依赖包：

```bashBash

pip install -r requirements.txtpip install -r requirements.txt

```步骤 3: 初始化环境 运行设置脚本以验证数据库连接并加载必要的配置：



**步骤 3: 初始化环境**Bash

python setup_env.py

运行设置脚本以验证数据库连接并加载必要的配置：步骤 4: 启动应用程序 启动 Web 界面：

```bash

python setup_env.pyBash

```python app.py

应用程序通常可通过以下地址访问：http://localhost:8501（或控制台输出中指定的端口）。
**步骤 4: 启动应用程序**

启动 Web 界面：
```bash
python app.py
```

应用程序通常可通过以下地址访问：http://localhost:8501（或控制台输出中指定的端口）。
