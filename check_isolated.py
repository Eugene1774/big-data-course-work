from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()

uri = os.getenv('NEO4J_URI', '')
user = os.getenv('NEO4J_USER', '')
password = os.getenv('NEO4J_PASSWORD', '')

driver = GraphDatabase.driver(uri, auth=(user, password))
with driver.session() as s:
    # 检查孤立节点（没有入边也没有出边）
    cypher = """
    MATCH (n)
    WHERE any(label IN labels(n) WHERE label IN ['Project', 'Case', 'Concept', 'KnowledgeCard', 'Method', 'Risk_Pattern', 'Value_Loop', 'Hyperedge'] OR label ENDS WITH '_Edge')
    AND NOT (n)--()
    RETURN labels(n) as labels, n.name as name, count(*) as cnt
    """
    result = list(s.run(cypher))
    print("孤立节点（没有关系的节点）:")
    for r in result:
        print(f"  {r['labels']}: {r['name']} ({r['cnt']}个)")
    
    print(f"\n孤立节点总数: {len(result)}")

driver.close()
