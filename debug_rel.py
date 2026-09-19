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
    OPTIONAL MATCH (n)-[r]->(m)
    WHERE m IS NULL OR any(label IN labels(m) WHERE label IN $allowed_labels OR label ENDS WITH '_Edge')
    RETURN n, r, m
    LIMIT 3
    """
    records = list(s.run(cypher, {"allowed_labels": allowed_labels}))
    
    for idx, record in enumerate(records):
        n = record.get("n")
        m = record.get("m")
        r = record.get("r")
        
        print(f"=== 记录 {idx} ===")
        print(f"r: {r}")
        
        for raw_node, node_name in [(n, "n"), (m, "m")]:
            if raw_node is None:
                print(f"{node_name}: None")
                continue
            
            labels = list(getattr(raw_node, "labels", []))
            print(f"{node_name}: labels={labels}")
            
            # 检查关系类型
            if r:
                rel_type = r.type if hasattr(r, 'type') else str(r)
                print(f"  rel_type: {rel_type}")
        
        print()
        
driver.close()
