from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()

uri = os.getenv('NEO4J_URI', '')
user = os.getenv('NEO4J_USER', '')
password = os.getenv('NEO4J_PASSWORD', '')

allowed_labels = ["Project", "Case", "Concept", "KnowledgeCard", "Method", "Risk_Pattern_Edge", "Value_Loop_Edge", "Risk_Pattern", "Value_Loop", "Hyperedge"]

driver = GraphDatabase.driver(uri, auth=(user, password))
with driver.session() as s:
    cypher = """
    MATCH (n)
    WHERE any(label IN labels(n) WHERE label IN $allowed_labels OR label ENDS WITH '_Edge')
    RETURN labels(n) as labels, count(*) as cnt
    ORDER BY cnt DESC
    """
    result = s.run(cypher, {"allowed_labels": allowed_labels}).data()
    print("查询到的标签分布:")
    for r in result:
        print(f"  {r['labels']}: {r['cnt']}")
        
driver.close()
