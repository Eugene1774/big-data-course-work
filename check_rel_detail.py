from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()

uri = os.getenv('NEO4J_URI', '')
user = os.getenv('NEO4J_USER', '')
password = os.getenv('NEO4J_PASSWORD', '')

driver = GraphDatabase.driver(uri, auth=(user, password))
with driver.session() as s:
    # 检查关系两端的节点类型
    result = s.run('''
        MATCH (a)-[r:ASSOCIATED_WITH]->(b)
        RETURN labels(a)[0] as source_type, labels(b)[0] as target_type, count(*) as cnt
        ORDER BY cnt DESC
    ''').data()
    print('ASSOCIATED_WITH 关系:')
    for row in result:
        print(f'  {row["source_type"]} -> {row["target_type"]}: {row["cnt"]}')
    
    # 检查PREREQ关系
    result = s.run('''
        MATCH (a)-[r:PREREQ]->(b)
        RETURN labels(a)[0] as source_type, labels(b)[0] as target_type, count(*) as cnt
    ''').data()
    print('\nPREREQ 关系:')
    for row in result:
        print(f'  {row["source_type"]} -> {row["target_type"]}: {row["cnt"]}')
    
    # 检查FIX_STRATEGY关系
    result = s.run('''
        MATCH (a)-[r:FIX_STRATEGY]->(b)
        RETURN labels(a)[0] as source_type, labels(b)[0] as target_type, count(*) as cnt
    ''').data()
    print('\nFIX_STRATEGY 关系:')
    for row in result:
        print(f'  {row["source_type"]} -> {row["target_type"]}: {row["cnt"]}')
        
    # 测试查询能返回多少条记录
    allowed_labels = ["Project", "Case", "Concept", "KnowledgeCard", "Method", "Risk_Pattern_Edge", "Value_Loop_Edge", "Risk_Pattern", "Value_Loop", "Hyperedge"]
    cypher = """
    MATCH (n)
    WHERE any(label IN labels(n) WHERE label IN $allowed_labels OR label ENDS WITH '_Edge')
    OPTIONAL MATCH (n)-[r]->(m)
    WHERE m IS NULL OR any(label IN labels(m) WHERE label IN $allowed_labels OR label ENDS WITH '_Edge')
    RETURN n, r, m
    LIMIT 800
    """
    records = list(s.run(cypher, {"allowed_labels": allowed_labels}))
    print(f'\n查询返回记录数: {len(records)}')
    
driver.close()
