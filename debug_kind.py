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
    RETURN n
    LIMIT 20
    """
    records = list(s.run(cypher, {"allowed_labels": allowed_labels}))
    
    for i, record in enumerate(records):
        n = record.get("n")
        labels = list(getattr(n, "labels", []))
        
        # 模拟 _node_kind 逻辑
        normalized = {str(label or "").strip() for label in labels}
        
        # _is_hyperedge_label 检查
        is_hyperedge = (
            "Risk_Pattern" in normalized
            or "Value_Loop" in normalized
            or "Hyperedge" in normalized
            or "Risk_Pattern_Edge" in normalized
            or "Value_Loop_Edge" in normalized
            or any(label.endswith("_Edge") for label in normalized)
        )
        
        # _node_kind 结果
        if is_hyperedge:
            kind = "Hyperedge"
        elif "Project" in normalized or "Case" in normalized:
            kind = "Project"
        else:
            kind = "Concept"
        
        print(f"[{i}] labels={labels}")
        print(f"    normalized={normalized}")
        print(f"    is_hyperedge={is_hyperedge}")
        print(f"    -> kind={kind}")
        print()
        
driver.close()
