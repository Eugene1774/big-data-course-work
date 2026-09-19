from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()

uri = os.getenv('NEO4J_URI', '')
user = os.getenv('NEO4J_USER', '')
password = os.getenv('NEO4J_PASSWORD', '')

driver = GraphDatabase.driver(uri, auth=(user, password))
with driver.session() as s:
    # 直接统计各种节点类型
    labels = ["Project", "Case", "Concept", "KnowledgeCard", "Method", "Risk_Pattern", "Value_Loop", "Hyperedge"]
    
    for label in labels:
        cnt = s.run(f"MATCH (n:`{label}`) RETURN count(n) as cnt").single()
        print(f"{label}: {cnt['cnt']}")

driver.close()
