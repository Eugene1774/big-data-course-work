from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()

uri = os.getenv('NEO4J_URI', '')
user = os.getenv('NEO4J_USER', '')
password = os.getenv('NEO4J_PASSWORD', '')

ALLOWED_GRAPH_LABELS = [
    "Project",
    "Case",
    "Concept",
    "KnowledgeCard",
    "Method",
    "Risk_Pattern_Edge",
    "Value_Loop_Edge",
    "Risk_Pattern",
    "Value_Loop",
    "Hyperedge",
]

def _is_hyperedge_label(labels):
    normalized = {str(label or "").strip() for label in labels}
    if (
        "Risk_Pattern" in normalized
        or "Value_Loop" in normalized
        or "Hyperedge" in normalized
        or "Risk_Pattern_Edge" in normalized
        or "Value_Loop_Edge" in normalized
    ):
        return True
    return any(label.endswith("_Edge") for label in normalized)

def _node_kind(labels):
    if _is_hyperedge_label(labels):
        return "Hyperedge"
    normalized = {str(label or "").strip() for label in labels}
    if "Project" in normalized or "Case" in normalized:
        return "Project"
    return "Concept"

driver = GraphDatabase.driver(uri, auth=(user, password))
with driver.session() as s:
    cypher = """
    MATCH (n)
    WHERE any(label IN labels(n) WHERE label IN $allowed_labels OR label ENDS WITH '_Edge')
    OPTIONAL MATCH (n)-[r]->(m)
    WHERE m IS NULL OR any(label IN labels(m) WHERE label IN $allowed_labels OR label ENDS WITH '_Edge')
    RETURN n, r, m
    LIMIT 10
    """
    records = list(s.run(cypher, {"allowed_labels": ALLOWED_GRAPH_LABELS}))
    
    for idx, record in enumerate(records):
        n = record.get("n")
        
        if n:
            labels = [str(label) for label in list(getattr(n, "labels", []))]
            kind = _node_kind(labels)
            
            props = dict(n)
            node_name = ""
            for key in ("name", "title", "project_name"):
                node_name = str(props.get(key, "")).strip()
                if node_name:
                    break
            if not node_name:
                node_name = labels[0] if labels else "Node"
            
            element_id = str(getattr(n, "element_id", None) or getattr(n, "id", None))
            
            print(f"记录 {idx}:")
            print(f"  labels: {labels}")
            print(f"  kind: {kind}")
            print(f"  node_name: {node_name}")
            print(f"  element_id: {element_id}")
            print()
        
driver.close()
