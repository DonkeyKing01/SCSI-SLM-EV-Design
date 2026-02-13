"""
新能源汽车知识图谱构建主程序
基于Persona_New聚类结果构建完整的知识图谱
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/knowledge_graph_build.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.knowledge_graph_builder import build_knowledge_graph_from_csv


def setup_argument_parser():
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='新能源汽车知识图谱构建器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  python main.py --csv ../02_User_Modeling/User_Preference_Clustering/outputs/user_dimension_vectors.csv --build
  python main.py --csv ../02_User_Modeling/User_Preference_Clustering/outputs/user_dimension_vectors.csv --clear --build
  python main.py --csv ../02_User_Modeling/User_Preference_Clustering/outputs/user_dimension_vectors.csv --neo4j-uri bolt://localhost:7688 --build
        '''
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='用户评论CSV文件路径 (必需)'
    )
    
    parser.add_argument(
        '--clustering-dir',
        type=str,
        default='../02_User_Modeling/User_Preference_Clustering/outputs',
        help='聚类结果目录路径 (默认: ../02_User_Modeling/User_Preference_Clustering/outputs)'
    )
    
    parser.add_argument(
        '--neo4j-uri',
        type=str,
        default='bolt://localhost:7688',
        help='Neo4j数据库URI (默认: bolt://localhost:7688)'
    )
    
    parser.add_argument(
        '--neo4j-user',
        type=str,
        default='neo4j',
        help='Neo4j用户名 (默认: neo4j)'
    )
    
    parser.add_argument(
        '--neo4j-password',
        type=str,
        default='neo4j123',
        help='Neo4j密码 (默认: neo4j123)'
    )
    
    parser.add_argument(
        '--neo4j-database',
        type=str,
        default='neo4j',
        help='Neo4j数据库名 (默认: neo4j)'
    )
    
    parser.add_argument(
        '--clear',
        action='store_true',
        help='清空现有数据库后重新构建'
    )
    
    parser.add_argument(
        '--build',
        action='store_true',
        help='构建知识图谱'
    )
    
    parser.add_argument(
        '--test-connection',
        action='store_true',
        help='测试Neo4j连接'
    )
    
    return parser


def test_neo4j_connection(uri: str, user: str, password: str, database: str):
    """测试Neo4j连接"""
    try:
        from src.neo4j_manager import Neo4jManager
        
        neo4j_manager = Neo4jManager(uri, user, password, database)
        
        if neo4j_manager.test_connection():
            logging.info("✓ Neo4j连接测试成功")
            
            # 获取数据库统计
            stats = neo4j_manager.get_database_stats()
            logging.info(f"数据库统计: {stats}")
            
        else:
            logging.error("✗ Neo4j连接测试失败")
            return False
            
    except Exception as e:
        logging.error(f"✗ Neo4j连接测试失败: {e}")
        return False
    finally:
        if 'neo4j_manager' in locals():
            neo4j_manager.close()
    
    return True


def build_knowledge_graph(args):
    """构建知识图谱"""
    logging.info("=" * 60)
    logging.info("开始构建新能源汽车知识图谱")
    logging.info("=" * 60)
    
    # 验证文件路径
    csv_path = Path(args.csv)
    clustering_dir = Path(args.clustering_dir)
    
    if not csv_path.exists():
        logging.error(f"CSV文件不存在: {csv_path}")
        return False
    
    if not clustering_dir.exists():
        logging.error(f"聚类目录不存在: {clustering_dir}")
        return False
    
    logging.info(f"CSV文件: {csv_path}")
    logging.info(f"聚类目录: {clustering_dir}")
    logging.info(f"Neo4j URI: {args.neo4j_uri}")
    logging.info(f"清空现有数据: {args.clear}")
    
    try:
        # 构建知识图谱
        result = build_knowledge_graph_from_csv(
            csv_file_path=str(csv_path),
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            neo4j_database=args.neo4j_database,
            clustering_dir=str(clustering_dir),
            clear_existing=args.clear
        )
        
        if result["status"] == "success":
            logging.info("✓ 知识图谱构建成功！")
            logging.info(f"构建耗时: {result['duration']:.2f}秒")
            logging.info(f"创建节点数: {result['stats']['nodes_created']}")
            logging.info(f"创建关系数: {result['stats']['relationships_created']}")
            
            # 显示数据库统计
            db_stats = result['database_stats']
            logging.info("数据库统计:")
            logging.info(f"  总节点数: {db_stats.get('total_nodes', 0)}")
            logging.info(f"  总关系数: {db_stats.get('total_relationships', 0)}")
            logging.info(f"  车型节点: {db_stats.get('car_models', 0)}")
            logging.info(f"  用户画像: {db_stats.get('user_profiles', 0)}")
            logging.info(f"  评论节点: {db_stats.get('reviews', 0)}")
            logging.info(f"  特征节点: {db_stats.get('features', 0)}")
            
            return True
        else:
            logging.error(f"✗ 知识图谱构建失败: {result.get('error', '未知错误')}")
            return False
            
    except Exception as e:
        logging.error(f"✗ 构建过程中发生错误: {e}")
        return False


def main():
    """主函数"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # 创建日志目录
    os.makedirs("logs", exist_ok=True)
    
    logging.info("新能源汽车知识图谱构建器启动")
    
    try:
        # 测试连接
        if args.test_connection:
            if test_neo4j_connection(args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.neo4j_database):
                logging.info("连接测试完成")
            else:
                logging.error("连接测试失败，请检查Neo4j配置")
                return 1
        
        # 构建知识图谱
        if args.build:
            if build_knowledge_graph(args):
                logging.info("知识图谱构建完成！")
                return 0
            else:
                logging.error("知识图谱构建失败！")
                return 1
        
        # 如果没有指定操作，显示帮助
        if not args.test_connection and not args.build:
            parser.print_help()
            return 0
            
    except KeyboardInterrupt:
        logging.info("用户中断操作")
        return 1
    except Exception as e:
        logging.error(f"程序执行失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)