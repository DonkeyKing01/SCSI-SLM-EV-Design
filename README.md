# 新能源汽车智能推荐系统 RAG

基于Neo4j知识图谱 + LangChain + Streamlit的智能汽车推荐系统

## 🚗 项目概述

本系统是一个基于RAG（检索增强生成）技术的新能源汽车智能推荐系统，整合了：
- **Model_New**: 车型特征IPA分析数据
- **Persona_New**: 用户画像聚类数据  
- **Neo4jFinal**: 知识图谱（包含4个节点类型）

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   LangChain     │    │     Neo4j       │
│    前端界面      │◄──►│   RAG框架       │◄──►│   知识图谱       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └─────────────►│  DeepSeek API   │◄─────────────┘
                        │     LLM         │
                        └─────────────────┘
```

## 🔧 核心功能

### 三大搜索工具

1. **vector_tool**: 向量检索工具
   - 基于语义相似度的文档搜索
   - 支持车型特征和用户画像数据
   - 使用OpenAI Embeddings

2. **vector_graph_tool**: 向量图谱工具
   - 结合向量搜索和图谱关系
   - 支持混合搜索模式
   - 智能查询意图分析

3. **graph_cypher_tool**: 图谱查询工具
   - 原生Cypher查询执行
   - 自然语言转Cypher
   - 专业图谱分析功能

### 智能问答功能

- 🎯 **个性化推荐**: 基于用户偏好推荐车型
- 📊 **特征分析**: 多维度车型特征对比
- 👥 **用户洞察**: 用户画像和偏好分析
- 🔍 **智能搜索**: 多模态搜索策略

## 📦 安装配置

### 1. 环境要求

```bash
Python 3.8+
Neo4j 5.14.0+
Docker (可选)
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `env_template.txt` 为 `.env` 并配置：

```bash
# Neo4j 连接配置
NEO4J_URI=bolt://localhost:7688
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=neo4j123
NEO4J_DATABASE=neo4jfinal

# API配置
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_MODEL=deepseek-chat

# 其他配置
DISABLE_ANALYTICS=true
```

### 4. 启动Neo4j数据库

使用Docker Compose：
```bash
cd ../Neo4jFinal
docker-compose up -d
```

或手动启动Neo4j实例。

## 🚀 使用方法

### 1. 启动应用

```bash
streamlit run app.py
```

### 2. 数据加载

在侧边栏点击"加载车型数据"和"加载用户数据"按钮，系统会自动加载：
- `../Model_New/outputs/car_model_scores.csv`
- `../Persona_New/outputs/user_vector_matrix.csv`

### 3. 选择搜索模式

- **🔄 混合搜索**: 综合向量和图谱搜索（推荐）
- **📊 向量搜索**: 纯语义相似度搜索
- **🕸️ 图谱搜索**: 基于知识图谱关系搜索  
- **💾 Cypher查询**: 直接执行图谱查询

### 4. 开始对话

#### 示例问题

**车型推荐类：**
- "推荐一款50万以上的新能源车"
- "哪款车最适合注重操控性能的用户？"
- "小米和宝马的车型有什么区别？"

**特征分析类：**
- "续航能耗表现最好的车型排名"
- "智能配置方面哪些车型最突出？"
- "用户最关注哪些车型特征？"

**用户洞察类：**
- "极氪001的用户画像是什么样的？"
- "喜欢外观设计的用户偏好哪些车型？"
- "不同用户群体的特征偏好分析"

## 🎯 技术特性

### RAG流程

1. **查询分析**: 智能识别用户意图和查询类型
2. **多源检索**: 结合向量搜索、图谱查询和Cypher分析
3. **上下文构建**: 整合检索结果构建丰富上下文
4. **答案生成**: 使用LLM生成准确、相关的回答
5. **来源追踪**: 提供答案来源和相关度信息

### 数据特色

- **8维特征体系**: 外观设计、内饰质感、智能配置、空间实用、舒适体验、操控性能、续航能耗、价值认知
- **30个用户画像**: 基于13,682条评论聚类生成
- **4节点图谱**: CarModel、UserProfile、Review、Feature
- **IPA分析**: 重要度-绩效分析矩阵

## 📊 系统监控

系统提供实时状态监控：
- Neo4j连接状态
- RAG引擎健康度
- 向量存储状态
- 数据库统计信息

## 🔍 项目结构

```
RAG_System/
├── config/              # 配置模块
│   ├── __init__.py
│   └── settings.py
├── database/            # 数据库连接
│   ├── __init__.py
│   └── neo4j_connection.py
├── tools/               # RAG工具集
│   ├── __init__.py
│   ├── vector_tool.py
│   ├── vector_graph_tool.py
│   └── graph_cypher_tool.py
├── core/                # 核心引擎
│   ├── __init__.py
│   └── rag_engine.py
├── app.py               # Streamlit主应用
├── requirements.txt     # Python依赖
├── env_template.txt     # 环境变量模板
└── README.md           # 说明文档
```

## 🚨 注意事项

1. **数据路径**: 确保 `Model_New` 和 `Persona_New` 数据文件存在
2. **Neo4j连接**: 确认Neo4j实例运行并可访问
3. **API密钥**: 配置正确的OpenAI API密钥和Base URL
4. **内存需求**: 向量存储和图谱操作需要足够内存
5. **Analytics**: 已默认禁用用户追踪功能

## 🔄 数据更新

当有新的评论数据或车型信息时：
1. 更新CSV文件
2. 重新加载数据到向量存储
3. 更新Neo4j图谱数据
4. 重启应用刷新缓存

## 📈 性能优化

- 向量存储持久化避免重复加载
- Neo4j索引优化查询性能
- LLM请求缓存减少API调用
- 分页和限制防止数据过载

## 🤝 技术支持

本系统集成了现有项目的数据和分析结果，保持了数据的一致性和完整性。如有问题请参考原项目文档或联系开发团队。

---

**系统版本**: v1.0  
**构建时间**: 2025年1月  
**技术栈**: Neo4j + LangChain + Streamlit + SiliconFlow API