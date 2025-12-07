#!/usr/bin/env python3
"""
RAG系统启动脚本
"""
import os
import sys
import subprocess
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_requirements():
    """检查依赖包"""
    try:
        import streamlit
        import langchain
        import neo4j
        import openai
        logger.info("✅ 所有依赖包已安装")
        return True
    except ImportError as e:
        logger.error(f"❌ 缺少依赖包: {e}")
        logger.info("请运行: pip install -r requirements.txt")
        return False

def check_env_config():
    """检查环境配置"""
    required_vars = [
        'NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD',
        'OPENAI_API_KEY', 'OPENAI_BASE_URL'
    ]
    
    # 尝试加载.env文件
    env_file = '.env'
    if os.path.exists(env_file):
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("✅ 已加载.env配置文件")
    else:
        logger.warning("⚠️ 未找到.env文件，使用系统环境变量")
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"❌ 缺少环境变量: {missing_vars}")
        logger.info("请创建.env文件并配置必要的环境变量")
        return False
    
    logger.info("✅ 环境变量配置完整")
    return True

def check_neo4j_connection():
    """检查Neo4j连接"""
    try:
        from database.neo4j_connection import neo4j_conn
        status = neo4j_conn.test_connection()
        
        if status['status'] == 'connected':
            logger.info("✅ Neo4j连接正常")
            logger.info(f"节点数: {status.get('node_count', 0)}")
            logger.info(f"关系数: {status.get('relationship_count', 0)}")
            return True
        else:
            logger.error(f"❌ Neo4j连接失败: {status.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        logger.error(f"❌ Neo4j连接检查失败: {e}")
        return False

def check_data_files():
    """检查数据文件"""
    data_files = [
        "../Model_New/outputs/car_model_scores.csv",
        "../Persona_New/outputs/user_vector_matrix.csv"
    ]
    
    all_exist = True
    for file_path in data_files:
        if os.path.exists(file_path):
            logger.info(f"✅ 数据文件存在: {file_path}")
        else:
            logger.warning(f"⚠️ 数据文件不存在: {file_path}")
            all_exist = False
    
    if not all_exist:
        logger.info("💡 数据文件不完整，可以在应用中手动加载")
    
    return True  # 数据文件不是必须的，可以后续加载

def main():
    """主函数"""
    logger.info("🚀 正在启动新能源汽车智能推荐系统...")
    
    # 检查系统要求
    checks = [
        ("依赖包检查", check_requirements),
        ("环境配置检查", check_env_config),
        ("Neo4j连接检查", check_neo4j_connection),
        ("数据文件检查", check_data_files)
    ]
    
    for check_name, check_func in checks:
        logger.info(f"执行 {check_name}...")
        if not check_func():
            logger.error(f"❌ {check_name} 失败，请解决问题后重试")
            sys.exit(1)
    
    logger.info("✅ 所有检查通过，正在启动Streamlit应用...")
    
    try:
        # 启动Streamlit应用
        cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("👋 用户中断，正在关闭应用...")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Streamlit启动失败: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 启动过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()