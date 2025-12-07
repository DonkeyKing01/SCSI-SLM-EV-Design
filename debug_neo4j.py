#!/usr/bin/env python3
"""Neo4j调试脚本"""

from database.neo4j_connection import neo4j_conn

def check_neo4j_version():
    """检查Neo4j版本和功能"""
    try:
        graph = neo4j_conn.get_graph()
        
        # 检查版本
        result = graph.query("CALL dbms.components() YIELD name, versions RETURN name, versions")
        print("=== Neo4j 组件信息 ===")
        for record in result:
            print(f"{record['name']}: {record['versions']}")
        
        # 检查向量索引支持
        print("\n=== 检查向量索引支持 ===")
        try:
            # 尝试查询现有索引
            indexes = graph.query("SHOW INDEXES")
            print(f"现有索引数量: {len(indexes)}")
            for idx in indexes:
                print(f"索引: {idx}")
        except Exception as e:
            print(f"查询索引失败: {e}")
        
        # 检查Review节点数据
        print("\n=== Review节点统计 ===")
        review_count = graph.query("MATCH (r:Review) RETURN COUNT(r) as count")[0]['count']
        print(f"Review节点总数: {review_count}")
        
        # 检查embedding属性
        embedding_count = graph.query("MATCH (r:Review) WHERE r.embedding IS NOT NULL RETURN COUNT(r) as count")[0]['count']
        print(f"包含embedding的Review节点: {embedding_count}")
        
        # 检查数据样本
        print("\n=== Review节点样本 ===")
        samples = graph.query("MATCH (r:Review) RETURN r.content, r.embedding IS NOT NULL as hasEmbedding LIMIT 3")
        for i, sample in enumerate(samples, 1):
            print(f"样本 {i}:")
            print(f"  内容: {sample['r.content'][:100]}...")
            print(f"  有embedding: {sample['hasEmbedding']}")
        
    except Exception as e:
        print(f"检查失败: {e}")

if __name__ == "__main__":
    check_neo4j_version()