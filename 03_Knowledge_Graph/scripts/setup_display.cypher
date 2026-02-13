// Neo4j图形界面显示配置脚本
// 用于设置节点在Neo4j Browser中的显示属性

// =====================================
// 1. 设置Feature节点显示配置
// =====================================

// 确保Feature节点显示name属性
:config {
  initialNodeDisplay: 300,
  maxNeighbours: 100,
  initialRelationshipDisplay: 300,
  maxConcurrency: 4
}

// 设置节点样式 (通过Neo4j Browser执行)
// Feature节点 - 显示名称
MATCH (f:Feature) 
RETURN f.name as 特征名称, f.category as 类别, f.description as 描述
LIMIT 10;

// CarModel节点 - 显示名称和品牌
MATCH (c:CarModel) 
RETURN c.name as 车型名称, c.brand as 品牌, c.type as 类型, c.priceRange as 价格区间
LIMIT 10;

// UserProfile节点 - 显示名称和用户数量
MATCH (u:UserProfile) 
RETURN u.name as 画像名称, u.userCount as 用户数量, u.description as 描述
ORDER BY u.profileId
LIMIT 30;

// Review节点 - 显示评论ID和情感
MATCH (r:Review) 
RETURN r.reviewId as 评论ID, r.overallSentiment as 整体情感, 
       substring(r.content, 0, 30) as 内容预览
LIMIT 10;

// =====================================
// 2. 验证图形结构完整性
// =====================================

// 检查Feature节点是否正确创建且有名称
MATCH (f:Feature) 
WHERE f.name IS NOT NULL
RETURN count(f) as 有名称的特征节点数量;

// 检查各种关系是否存在
MATCH ()-[r:PUBLISHED]->() RETURN count(r) as PUBLISHED关系数量
UNION ALL
MATCH ()-[r:MENTIONS]->() RETURN count(r) as MENTIONS关系数量
UNION ALL
MATCH ()-[r:CONTAINS_ASPECT]->() RETURN count(r) as CONTAINS_ASPECT关系数量
UNION ALL
MATCH ()-[r:INTERESTED_IN]->() RETURN count(r) as INTERESTED_IN关系数量;

// =====================================
// 3. 推荐的Neo4j Browser显示设置
// =====================================

// 在Neo4j Browser中执行以下命令来优化显示:

// 设置Feature节点显示name属性:
// :style Feature { 
//   diameter: 50px; 
//   color: #8DCC93; 
//   border-color: #5CA16D; 
//   border-width: 2px; 
//   text-color-internal: #000000; 
//   caption: {name}; 
// }

// 设置CarModel节点显示name属性:
// :style CarModel { 
//   diameter: 60px; 
//   color: #569480; 
//   border-color: #447666; 
//   border-width: 2px; 
//   text-color-internal: #FFFFFF; 
//   caption: {name}; 
// }

// 设置UserProfile节点显示name属性:
// :style UserProfile { 
//   diameter: 55px; 
//   color: #F79767; 
//   border-color: #D4702A; 
//   border-width: 2px; 
//   text-color-internal: #FFFFFF; 
//   caption: {name}; 
// }

// 设置Review节点显示reviewId:
// :style Review { 
//   diameter: 40px; 
//   color: #4C8EDA; 
//   border-color: #2D5016; 
//   border-width: 1px; 
//   text-color-internal: #FFFFFF; 
//   caption: {reviewId}; 
// }

// =====================================
// 4. 测试查询 - 验证节点名称显示
// =====================================

// 测试查询1: 显示所有Feature节点及其名称
MATCH (f:Feature) 
RETURN f 
ORDER BY f.name;

// 测试查询2: 显示用户画像和相关车型
MATCH (u:UserProfile)-[i:INTERESTED_IN]->(c:CarModel)
WHERE u.profileId <= 5
RETURN u, i, c
ORDER BY i.correlationScore DESC
LIMIT 20;

// 测试查询3: 显示特征分析示例
MATCH (r:Review)-[ca:CONTAINS_ASPECT]->(f:Feature)
WHERE f.name = '智能配置' AND ca.intensity > 0.5
MATCH (r)-[m:MENTIONS]->(c:CarModel)
RETURN r, ca, f, m, c
LIMIT 10;

// =====================================
// 5. 图形可视化优化查询
// =====================================

// 小规模子图查询 - 适合可视化
MATCH path = (u:UserProfile {profileId: 1})-[:INTERESTED_IN]->(c:CarModel)
WHERE c.brand = '小米'
RETURN path
LIMIT 5;

// 特征网络查询 - 显示特征之间的关联
MATCH (f:Feature)<-[:CONTAINS_ASPECT]-(r:Review)-[:CONTAINS_ASPECT]->(f2:Feature)
WHERE f <> f2
RETURN f, r, f2
LIMIT 20;