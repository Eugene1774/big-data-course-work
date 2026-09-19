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
        
        for raw_node, node_name in [(n, "n"), (m, "m")]:
            if raw_node is None:
                print(f"{node_name}: None")
                continue
            
            labels = list(getattr(raw_node, "labels", []))
            
            # 使用 admin_portal.py 中的实际函数
            normalized = {str(label or "").strip() for label in labels}
            
            # _is_hyperedge_label
            is_hyperedge = (
                "Risk_Pattern" in normalized
                or "Value_Loop" in normalized
                or "Hyperedge" in normalized
                or "Risk_Pattern_Edge" in normalized
                or "Value_Loop_Edge" in normalized
                or any(label.endswith("_Edge") for label in normalized)
            )
            
            # _node_kind
            if is_hyperedge:
                kind = "Hyperedge"
            elif "Project" in normalized or "Case" in normalized:
                kind = "Project"
            else:
                kind = "Concept"
            
            print(f"{node_name}: labels={labels}, normalized={normalized}, is_hyperedge={is_hyperedge}, kind={kind}")
        
        print(f"r type: {type(r)}, rel_type: {r.type() if r else None}")
        print()
        
driver.close()
