#!/usr/bin/env python3
"""
环境变量设置脚本
"""
import os
import shutil

def setup_environment():
    """设置环境变量"""
    
    # 复制env模板为.env
    if not os.path.exists('.env'):
        if os.path.exists('env_template.txt'):
            shutil.copy('env_template.txt', '.env')
            print("✅ 已创建.env文件")
        else:
            print("❌ 未找到env_template.txt文件")
            return False
    
    # 设置环境变量（临时方案）
    env_vars = {
        'NEO4J_URI': 'bolt://localhost:7688',
        'NEO4J_USERNAME': 'neo4j',
        'NEO4J_PASSWORD': 'neo4j123',
        'NEO4J_DATABASE': 'neo4jfinal',
        'OPENAI_API_KEY': 'sk-jnjnovforhovyujpdhwkviuhxhxnrspnouzbxfrgujhslhap',
        'OPENAI_BASE_URL': 'https://api.siliconflow.cn/v1',
        'OPENAI_MODEL': 'Pro/THUDM/glm-4-9b-chat',  # 智谱GLM-4模型（免费）
        'APP_TITLE': '新能源汽车智能推荐系统',
        'APP_ICON': '🚗',
        'DISABLE_ANALYTICS': 'true',
        'VECTOR_STORE_PATH': './vector_store',
        'CHUNK_SIZE': '1000',
        'CHUNK_OVERLAP': '200',
        'EMBEDDING_MODEL': 'BAAI/bge-large-zh-v1.5'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("✅ 环境变量设置完成")
    return True

if __name__ == "__main__":
    setup_environment()