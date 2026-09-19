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
    LIMIT 50
    """
    records = list(s.run(cypher, {"allowed_labels": allowed_labels}))
    
    for record in records:
        n = record.get("n")
        labels = list(getattr(n, "labels", []))
        print(f"节点标签: {labels}")
        
        # 测试 _is_hyperedge_label
        normalized = {str(label or "").strip() for label in labels}
        print(f"  normalized: {normalized}")
        
        is_hyperedge = (
            "Risk_Pattern" in normalized
            or "Value_Loop" in normalized
            or "Hyperedge" in normalized
            or "Risk_Pattern_Edge" in normalized
            or "Value_Loop_Edge" in normalized
            or any(label.endswith("_Edge") for label in normalized)
        )
        print(f"  is_hyperedge: {is_hyperedge}")
        print()
        
        if len(records) > 20:
            break
        
driver.close()
