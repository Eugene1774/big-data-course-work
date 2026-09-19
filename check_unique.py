from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()

uri = os.getenv('NEO4J_URI', '')
user = os.getenv('NEO4J_USER', '')
password = os.getenv('NEO4J_PASSWORD', '')

driver = GraphDatabase.driver(uri, auth=(user, password))
with driver.session() as s:
    # 统计唯一节点 element_id
    cypher = """
    MATCH (n)
    WHERE any(label IN labels(n) WHERE label IN ['Project', 'Case', 'Concept', 'KnowledgeCard', 'Method', 'Risk_Pattern', 'Value_Loop', 'Hyperedge'] OR label ENDS WITH '_Edge')
    RETURN count(DISTINCT n) as distinct_nodes
    """
    result = s.run(cypher).single()
    print(f"数据库中唯一节点数: {result['distinct_nodes']}")
    
    # 统计返回的 n 和 m 节点 element_id
    cypher2 = """
    MATCH (n)
    WHERE any(label IN labels(n) WHERE label IN ['Project', 'Case', 'Concept', 'KnowledgeCard', 'Method', 'Risk_Pattern', 'Value_Loop', 'Hyperedge'] OR label ENDS WITH '_Edge')
    OPTIONAL MATCH (n)-[r]->(m)
    WHERE m IS NULL OR any(label IN labels(m) WHERE label IN ['Project', 'Case', 'Concept', 'KnowledgeCard', 'Method', 'Risk_Pattern', 'Value_Loop', 'Hyperedge'] OR label ENDS WITH '_Edge')
    RETURN n, r, m
    LIMIT 800
    """
    records = list(s.run(cypher2))
    
    n_ids = set()
    m_ids = set()
    
    for record in records:
        n = record.get("n")
        m = record.get("m")
        
        if n:
            n_ids.add(getattr(n, "element_id", None))
        if m:
            m_ids.add(getattr(m, "element_id", None))
    
    print(f"\n查询返回的唯一 n 节点数: {len(n_ids)}")
    print(f"查询返回的唯一 m 节点数: {len(m_ids)}")
    print(f"总唯一节点数 (n + m): {len(n_ids | m_ids)}")

driver.close()
