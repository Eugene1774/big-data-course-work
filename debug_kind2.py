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
    LIMIT 800
    """
    records = list(s.run(cypher, {"allowed_labels": allowed_labels}))
    
    # 逐条调试
    kind_debug = {}
    for idx, record in enumerate(records[:30]):  # 只看前30条
        n = record.get("n")
        m = record.get("m")
        
        for raw_node in (n, m):
            if raw_node is None:
                continue
            labels = list(getattr(raw_node, "labels", []))
            
            has_allowed = any(label in allowed_labels or str(label).endswith('_Edge') for label in labels)
            if not has_allowed:
                continue
            
            # _node_kind 逻辑
            normalized = {str(label or "").strip() for label in labels}
            is_hyperedge = any(
                label in normalized or label.endswith("_Edge") 
                for label in ["Risk_Pattern", "Value_Loop", "Hyperedge", "Risk_Pattern_Edge", "Value_Loop_Edge"]
            ) or any(str(label).endswith("_Edge") for label in normalized)
            
            if is_hyperedge:
                kind = "Hyperedge"
            elif "Project" in normalized or "Case" in normalized:
                kind = "Project"
            else:
                kind = "Concept"
            
            kind_debug[kind] = kind_debug.get(kind, 0) + 1
    
    print("前30条记录的 kind 分布:")
    for k, v in sorted(kind_debug.items()):
        print(f"  {k}: {v}")
        
driver.close()
