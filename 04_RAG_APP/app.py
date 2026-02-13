"""
新能源汽车智能推荐系统 - Streamlit主应用
基于Neo4j知识图谱和RAG技术的智能问答系统
"""
import streamlit as st
import sys
import os
import logging
from typing import Dict, Any
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import Settings
from core.rag_engine import rag_engine
from database.neo4j_connection import neo4j_conn
from tools.vector_tool import vector_tool

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置页面配置
st.set_page_config(
    page_title=Settings.APP_TITLE,
    page_icon=Settings.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# 禁用analytics追踪
if Settings.DISABLE_ANALYTICS:
    st.markdown("""
    <style>
    .reportview-container {
        margin-top: 0em;
    }
    .main .block-container {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def init_session_state():
    """初始化会话状态"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'search_mode' not in st.session_state:
        st.session_state.search_mode = "auto"

def check_system_status():
    """检查系统状态"""
    try:
        # 测试Neo4j连接
        neo4j_status = neo4j_conn.test_connection()
        
        # 测试RAG引擎
        engine_status = rag_engine.get_system_status()
        
        return {
            'neo4j': neo4j_status,
            'rag_engine': engine_status
        }
    except Exception as e:
        logger.error(f"系统状态检查失败: {e}")
        return {'error': str(e)}

def display_system_status():
    """显示系统状态"""
    with st.sidebar:
        st.subheader("🔧 系统状态")
        
        status = check_system_status()
        
        if 'error' in status:
            st.error(f"系统检查失败: {status['error']}")
            return False
        
        # Neo4j状态
        neo4j_ok = status['neo4j']['status'] == 'connected'
        if neo4j_ok:
            st.success("✅ Neo4j已连接")
            st.info(f"节点数: {status['neo4j'].get('node_count', 0)}")
            st.info(f"关系数: {status['neo4j'].get('relationship_count', 0)}")
        else:
            st.error("❌ Neo4j连接失败")
        
        # RAG引擎状态
        engine_status = status.get('rag_engine', {})
        if engine_status.get('rag_engine') == 'healthy':
            st.success("✅ RAG引擎正常")
        else:
            st.error("❌ RAG引擎异常")
        
        return neo4j_ok and engine_status.get('rag_engine') == 'healthy'

def display_data_overview():
    """显示数据概况"""
    with st.sidebar:
        st.subheader("📊 数据概况")
        
        # 数据统计指标
        col1, col2 = st.columns(2)
        with col1:
            st.metric("车型数量", "50+", "5")
            st.metric("用户画像", "30", "2")
        with col2:
            st.metric("评论数量", "13,682", "1,234")
            st.metric("特征维度", "8", "0")
        
        # 系统状态指示
        st.info("💡 系统运行正常")
        
        # 可选：显示简单的数据加载界面
        with st.expander("🔧 数据管理"):
            # 检查数据路径
            model_data_path = "../02_User_Modeling/Product_IPA_Analysis/outputs/car_model_scores.csv"
            persona_data_path = "../02_User_Modeling/User_Preference_Clustering/outputs/user_vector_matrix.csv"
            
            if st.button("重新加载车型数据", key="reload_model"):
                try:
                    if os.path.exists(model_data_path):
                        count = vector_tool.load_car_model_data(model_data_path)
                        st.success(f"✅ 已加载 {count} 个车型")
                    else:
                        st.error("❌ 车型数据文件不存在")
                except Exception as e:
                    st.error(f"❌ 加载失败: {e}")
            
            if st.button("重新加载用户数据", key="reload_persona"):
                try:
                    if os.path.exists(persona_data_path):
                        count = vector_tool.load_user_persona_data(persona_data_path)
                        st.success(f"✅ 已加载 {count} 个画像")
                    else:
                        st.error("❌ 用户数据文件不存在")
                except Exception as e:
                    st.error(f"❌ 加载失败: {e}")

def display_search_settings():
    """显示搜索设置"""
    st.sidebar.subheader("🔍 搜索设置")
    
    search_modes = {
        "auto": "🎯 智能模式 (推荐)",
        "graph": "🕸️ 图谱搜索",
        "vector": "📊 向量搜索",
        "cypher": "💾 Cypher查询"
    }
    
    selected_mode = st.sidebar.selectbox(
        "选择搜索模式:",
        options=list(search_modes.keys()),
        format_func=lambda x: search_modes[x],
        index=0
    )
    
    st.session_state.search_mode = selected_mode
    
    # 显示模式说明
    mode_descriptions = {
        "auto": "根据问题类型自动选择最佳搜索策略",
        "graph": "基于知识图谱关系的智能搜索",
        "vector": "基于语义相似度的向量检索",
        "cypher": "直接使用Cypher查询语言进行精确搜索"
    }
    
    st.sidebar.info(f"💡 {mode_descriptions[selected_mode]}")

def display_example_questions():
    """显示示例问题"""
    with st.sidebar:
        st.subheader("💡 示例问题")
        
        examples = {
            "车型推荐": [
                "推荐一款50万以上的新能源车",
                "哪款车最适合注重操控性能的用户？",
                "小米和宝马的车型有什么区别？"
            ],
            "特征分析": [
                "续航能耗表现最好的车型排名",
                "智能配置方面哪些车型最突出？",
                "用户最关注哪些车型特征？"
            ],
            "用户洞察": [
                "极氪001的用户画像是什么样的？",
                "喜欢外观设计的用户偏好哪些车型？",
                "不同用户群体的特征偏好分析"
            ]
        }
        
        for category, questions in examples.items():
            with st.expander(f"📝 {category}"):
                for i, question in enumerate(questions):
                    if st.button(question, key=f"example_{category}_{i}", use_container_width=True):
                        # 直接触发查询
                        st.session_state.pending_question = question
                        st.rerun()

def process_query(question: str, search_mode: str) -> Dict[str, Any]:
    """处理用户查询"""
    try:
        with st.spinner(f"正在使用{search_mode}模式搜索..."):
            result = rag_engine.query(question, search_mode)
        return result
    except Exception as e:
        logger.error(f"查询处理失败: {e}")
        return {
            "answer": f"查询处理失败: {str(e)}",
            "sources": [],
            "error": str(e)
        }

def display_chat_interface():
    """显示聊天界面"""
    st.header(f"{Settings.APP_ICON} {Settings.APP_TITLE}")
    st.markdown("---")
    
    # 聊天历史显示区域
    chat_container = st.container()
    
    with chat_container:
        for i, (question, answer, metadata) in enumerate(st.session_state.chat_history):
            # 用户问题
            with st.chat_message("user"):
                st.write(question)
            
            # AI回答
            with st.chat_message("assistant"):
                st.write(answer)
                
                # 显示元数据
                if metadata:
                    with st.expander("📊 详细信息"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("搜索模式", metadata.get('search_mode', 'unknown'))
                            st.metric("上下文数量", metadata.get('context_count', 0))
                        with col2:
                            if 'sources' in metadata and metadata['sources']:
                                st.write("**来源:**")
                                for j, source in enumerate(metadata['sources'][:3]):
                                    st.write(f"{j+1}. {source['content'][:100]}...")
    
    # 输入区域
    st.markdown("---")
    
    # 检查是否有待处理的示例问题
    if 'pending_question' in st.session_state:
        user_input = st.session_state.pending_question
        del st.session_state.pending_question
    else:
        # 用户输入
        user_input = st.chat_input("请输入您的问题...")
    
    if user_input:
        # 处理查询
        result = process_query(user_input, st.session_state.search_mode)
        
        # 添加到历史记录
        st.session_state.chat_history.append((
            user_input,
            result['answer'],
            {
                'search_mode': result.get('search_mode'),
                'context_count': result.get('context_count'),
                'sources': result.get('sources', [])
            }
        ))
        
        # 刷新页面显示最新对话
        st.rerun()

def display_analytics_dashboard():
    """显示分析仪表板"""
    st.header("📊 数据分析仪表板")
    
    try:
        # 模拟一些统计数据（实际应用中应该从数据库获取）
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("车型数量", "50+", "5")
        with col2:
            st.metric("用户画像", "30", "2")
        with col3:
            st.metric("评论数量", "13,682", "1,234")
        with col4:
            st.metric("特征维度", "8", "0")
        
        # 示例图表
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("特征关注度分布")
            features = ['外观设计', '内饰质感', '智能配置', '空间实用', '舒适体验', '操控性能', '续航能耗', '价值认知']
            values = [0.365, 0.251, 0.384, 0.310, 0.261, 0.503, 0.539, 0.257]
            
            # 创建DataFrame用于plotly
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[
                go.Bar(x=features, y=values, name='关注度')
            ])
            fig.update_layout(
                title="用户对各特征的平均关注强度",
                xaxis_title="特征类型",
                yaxis_title="关注强度",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("品牌分布")
            brands = ['小米', '智界', '享界', '极氪', '宝马', '奔驰', '其他']
            counts = [2, 8, 4, 12, 6, 5, 13]
            
            fig = go.Figure(data=[
                go.Pie(labels=brands, values=counts, name="品牌分布")
            ])
            fig.update_layout(
                title="车型品牌分布"
            )
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"分析仪表板加载失败: {e}")
        logger.error(f"Analytics dashboard error: {e}")

def main():
    """主函数"""
    init_session_state()
    
    # 侧边栏
    system_ok = display_system_status()
    
    if not system_ok:
        st.error("⚠️ 系统未正常初始化，请检查配置和连接")
        st.stop()
    
    display_data_overview()
    display_search_settings()
    display_example_questions()
    
    # 主界面选项卡
    tab1, tab2, tab3 = st.tabs(["💬 智能问答", "📊 数据分析", "ℹ️ 系统信息"])
    
    with tab1:
        display_chat_interface()
    
    with tab2:
        display_analytics_dashboard()
    
    with tab3:
        st.header("ℹ️ 系统信息")
        
        st.subheader("🏗️ 系统架构")
        st.info("""
        本系统基于以下技术栈构建：
        - **Neo4j**: 知识图谱数据库
        - **LangChain**: RAG框架
        - **Streamlit**: 前端界面
        - **OpenAI API**: 大语言模型
        - **向量搜索**: 语义检索
        """)
        
        st.subheader("📋 数据来源")
        st.info("""
        - **Product_IPA_Analysis**: 车型特征IPA分析数据
        - **User_Preference_Clustering**: 用户画像聚类数据  
        - **Neo4jFinal**: 知识图谱数据（4个节点类型）
        - **评论数据**: 13,682条真实用户评论
        """)
        
        st.subheader("🔧 功能特性")
        st.success("""
        ✅ 三种搜索工具协同工作
        ✅ 向量检索 + 图谱查询 + Cypher分析
        ✅ 个性化车型推荐
        ✅ 多维度特征分析
        ✅ 用户画像洞察
        """)
        
        # 显示详细系统状态
        with st.expander("🔍 详细系统状态"):
            status = check_system_status()
            st.json(status)

if __name__ == "__main__":
    main()