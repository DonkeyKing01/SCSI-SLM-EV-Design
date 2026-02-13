// 新能源汽车知识图谱 - 数据库初始化脚本
// 此脚本用于创建约束、索引和基础数据结构

// =====================================
// 1. 清理现有数据 (可选 - 仅在重新初始化时使用)
// =====================================
// MATCH (n) DETACH DELETE n;

// =====================================
// 2. 创建唯一约束
// =====================================

// 车型节点唯一约束
CREATE CONSTRAINT car_model_id IF NOT EXISTS 
FOR (c:CarModel) REQUIRE c.modelId IS UNIQUE;

// 用户画像节点唯一约束
CREATE CONSTRAINT user_profile_id IF NOT EXISTS 
FOR (u:UserProfile) REQUIRE u.profileId IS UNIQUE;

// 评论节点唯一约束
CREATE CONSTRAINT review_id IF NOT EXISTS 
FOR (r:Review) REQUIRE r.reviewId IS UNIQUE;

// 特征节点唯一约束
CREATE CONSTRAINT feature_name IF NOT EXISTS 
FOR (f:Feature) REQUIRE f.name IS UNIQUE;

// =====================================
// 3. 创建性能索引
// =====================================

// 车型相关索引
CREATE INDEX car_model_name IF NOT EXISTS 
FOR (c:CarModel) ON (c.name);

CREATE INDEX car_model_brand IF NOT EXISTS 
FOR (c:CarModel) ON (c.brand);

CREATE INDEX car_model_price_range IF NOT EXISTS 
FOR (c:CarModel) ON (c.priceRange);

// 用户画像相关索引
CREATE INDEX user_profile_name IF NOT EXISTS 
FOR (u:UserProfile) ON (u.name);

CREATE INDEX user_profile_user_count IF NOT EXISTS 
FOR (u:UserProfile) ON (u.userCount);

// 评论相关索引
CREATE INDEX review_user_id IF NOT EXISTS 
FOR (r:Review) ON (r.userId);

CREATE INDEX review_sentiment IF NOT EXISTS 
FOR (r:Review) ON (r.overallSentiment);

CREATE INDEX review_car_model IF NOT EXISTS 
FOR (r:Review) ON (r.carModel);

// 特征相关索引
CREATE INDEX feature_category IF NOT EXISTS 
FOR (f:Feature) ON (f.category);

// =====================================
// 4. 创建关系索引 (用于RAG查询优化)
// =====================================

// PUBLISHED关系索引
CREATE INDEX published_user_match IF NOT EXISTS 
FOR ()-[r:PUBLISHED]-() ON (r.userMatchScore);

// MENTIONS关系索引
CREATE INDEX mentions_sentiment IF NOT EXISTS 
FOR ()-[r:MENTIONS]-() ON (r.sentimentScore);

CREATE INDEX mentions_importance IF NOT EXISTS 
FOR ()-[r:MENTIONS]-() ON (r.importance);

// CONTAINS_ASPECT关系索引
CREATE INDEX contains_aspect_intensity IF NOT EXISTS 
FOR ()-[r:CONTAINS_ASPECT]-() ON (r.intensity);

CREATE INDEX contains_aspect_sentiment IF NOT EXISTS 
FOR ()-[r:CONTAINS_ASPECT]-() ON (r.aspectSentiment);

// INTERESTED_IN关系索引
CREATE INDEX interested_correlation IF NOT EXISTS 
FOR ()-[r:INTERESTED_IN]-() ON (r.correlationScore);

CREATE INDEX interested_positive IF NOT EXISTS 
FOR ()-[r:INTERESTED_IN]-() ON (r.positiveMentions);

CREATE INDEX interested_negative IF NOT EXISTS 
FOR ()-[r:INTERESTED_IN]-() ON (r.negativeMentions);

// =====================================
// 5. 创建预定义特征节点 (8个核心维度)
// =====================================

// 外观设计
MERGE (f1:Feature {
    name: "外观设计",
    category: "视觉感知",
    description: "车型外观造型、设计美学、视觉冲击力",
    keywords: ["外观", "设计", "造型", "颜值", "美观", "好看", "漂亮", "时尚"]
});

// 内饰质感
MERGE (f2:Feature {
    name: "内饰质感",
    category: "内部体验",
    description: "车内装饰材质、做工品质、豪华感",
    keywords: ["内饰", "质感", "材质", "做工", "豪华", "精致", "档次", "品质"]
});

// 智能配置
MERGE (f3:Feature {
    name: "智能配置",
    category: "科技功能",
    description: "自动驾驶、智能车机、科技配置",
    keywords: ["智能", "自动驾驶", "车机", "科技", "配置", "功能", "系统", "辅助"]
});

// 空间实用
MERGE (f4:Feature {
    name: "空间实用",
    category: "功能性",
    description: "乘坐空间、储物能力、实用性",
    keywords: ["空间", "实用", "储物", "乘坐", "座椅", "后排", "后备箱", "装载"]
});

// 舒适体验
MERGE (f5:Feature {
    name: "舒适体验",
    category: "乘坐感受",
    description: "座椅舒适性、噪音控制、悬挂调校",
    keywords: ["舒适", "噪音", "悬挂", "减震", "座椅", "静音", "平稳", "柔软"]
});

// 操控性能
MERGE (f6:Feature {
    name: "操控性能",
    category: "驾驶体验",
    description: "动力输出、加速性能、操控感受",
    keywords: ["操控", "性能", "动力", "加速", "提速", "驾驶", "灵活", "响应"]
});

// 续航能耗
MERGE (f7:Feature {
    name: "续航能耗",
    category: "续航能力",
    description: "电池续航、充电速度、能耗表现",
    keywords: ["续航", "能耗", "电池", "充电", "里程", "电量", "省电", "耐用"]
});

// 价值认知
MERGE (f8:Feature {
    name: "价值认知",
    category: "性价比",
    description: "价格合理性、品牌价值、投资回报",
    keywords: ["价格", "性价比", "值得", "便宜", "贵", "划算", "品牌", "保值"]
});

// =====================================
// 6. 验证初始化结果
// =====================================

// 检查约束数量
SHOW CONSTRAINTS;

// 检查索引数量
SHOW INDEXES;

// 检查特征节点
MATCH (f:Feature) 
RETURN f.name as featureName, f.category as category
ORDER BY f.name;

// 显示数据库统计
MATCH (n) 
RETURN labels(n) as nodeType, count(*) as count
ORDER BY nodeType;