import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# 1. 加载环境变量
load_dotenv()
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

# 2. 将语句拆分为独立的单条执行语句
# 语句1：清空数据库
CLEAR_DB_QUERY = "MATCH (n) DETACH DELETE n"

# 语句2：刷入图谱数据（作为一个完整的单条事务）
INIT_DATA_QUERY = """
CREATE (c1:Concept {id: 'Target_Customer', name: '目标客户'})
CREATE (c2:Concept {id: 'Value_Prop', name: '价值主张'})
CREATE (c3:Concept {id: 'Channel', name: '获客渠道'})
CREATE (he1:Risk_Pattern {id: 'HE_H1_Mismatch', name: '渠道错位风险', severity: 'High'})
CREATE (p1:Project {id: 'Project_DroneFarm', name: '无人机农业配送'})
CREATE (p2:Project {id: 'Project_CampusDelivery', name: '校园外卖聚合'})
CREATE (he1)-[:INVOLVES]->(c1)
CREATE (he1)-[:INVOLVES]->(c2)
CREATE (he1)-[:INVOLVES]->(c3)
CREATE (p1)-[:TRIGGERS]->(he1)
"""


def init_database():
    print(f"正在连接到 Neo4j Aura: {URI} ...")
    try:
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        driver.verify_connectivity()
        print("✅ 数据库连接成功！")

        # 拆分执行
        with driver.session() as session:
            print("正在清空旧数据...")
            session.run(CLEAR_DB_QUERY)
            print("正在写入双创核心超图节点...")
            session.run(INIT_DATA_QUERY)

            print("✅ 双创测试数据（节点与超边）已成功刷入数据库！")

        driver.close()
    except Exception as e:
        print(f"❌ 发生错误: {e}")


if __name__ == "__main__":
    init_database()
