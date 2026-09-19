from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()

uri = os.getenv('NEO4J_URI', '')
user = os.getenv('NEO4J_USER', '')
password = os.getenv('NEO4J_PASSWORD', '')

driver = GraphDatabase.driver(uri, auth=(user, password))
with driver.session() as s:
    # 检查Case节点的连接情况
    result = s.run('''
        MATCH (c:Case)
        OPTIONAL MATCH (c)-[r]->()
        WITH c, count(r) as outgoing
        RETURN 
            count(c) as total_case,
            sum(CASE WHEN outgoing = 0 THEN 1 ELSE 0 END) as isolated_case,
            sum(CASE WHEN outgoing > 0 THEN 1 ELSE 0 END) as connected_case
    ''').single()
    print(f'Case节点总数: {result["total_case"]}')
    print(f'孤立Case: {result["isolated_case"]}')
    print(f'有连接的Case: {result["connected_case"]}')
    
    # 检查关系类型
    result = s.run('''
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN collect(relationshipType) as types
    ''').single()
    print(f'关系类型: {result["types"]}')
driver.close()
