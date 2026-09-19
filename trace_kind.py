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
    print(f"  _is_hyperedge_label input: {labels}")
    print(f"    normalized: {normalized}")
    
    result = (
        "Risk_Pattern" in normalized
        or "Value_Loop" in normalized
        or "Hyperedge" in normalized
        or "Risk_Pattern_Edge" in normalized
        or "Value_Loop_Edge" in normalized
    )
    print(f"    check _Edge: {any(label.endswith('_Edge') for label in normalized)}")
    print(f"    -> result: {result}")
    return result

def _node_kind(labels):
    print(f"_node_kind input: {labels}")
    result = _is_hyperedge_label(labels)
    if result:
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
    LIMIT 5
    """
    records = list(s.run(cypher, {"allowed_labels": ALLOWED_GRAPH_LABELS}))
    
    for idx, record in enumerate(records):
        print(f"\n=== 记录 {idx} ===")
        n = record.get("n")
        m = record.get("m")
        
        for raw_node, name in [(n, "n"), (m, "m")]:
            if raw_node is None:
                continue
            
            labels = [str(label) for label in list(getattr(raw_node, "labels", []))]
            kind = _node_kind(labels)
            print(f"  {name}: labels={labels}, kind={kind}")
        
driver.close()
