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

def _has_allowed_label(raw_labels):
    return any(label in ALLOWED_GRAPH_LABELS or label.endswith("_Edge") for label in raw_labels)

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
    records = list(s.run(cypher, {"allowed_labels": ALLOWED_GRAPH_LABELS}))
    
    node_map = {}
    for record in records:
        n = record.get("n")
        m = record.get("m")
        
        for raw_node in (n, m):
            if raw_node is None:
                continue
            
            labels = [str(label) for label in list(getattr(raw_node, "labels", []))]
            if not _has_allowed_label(labels):
                continue
            
            kind = _node_kind(labels)
            
            # 获取 node_name
            props = dict(raw_node)
            node_name = ""
            for key in ("name", "title", "project_name"):
                node_name = str(props.get(key, "")).strip()
                if node_name:
                    break
            if not node_name:
                node_name = labels[0] if labels else "Node"
            
            # 去重逻辑
            if kind == "Concept" or any(label in ["Concept", "KnowledgeCard", "Method", "Tag", "Rule"] for label in labels):
                dedup_key = f"core:{node_name}"
            elif kind == "Hyperedge":
                dedup_key = f"hyperedge:{node_name}"
            else:
                from streamlit_agraph.node import Node as AgraphNode
                element_id = str(getattr(raw_node, "element_id", None) or getattr(raw_node, "id", None) or id(raw_node))
                dedup_key = f"project:{element_id}"
            
            if dedup_key not in node_map:
                node_map[dedup_key] = kind
    
    print(f"去重后节点数: {len(node_map)}")
    
    kind_counts = {}
    for k, v in node_map.items():
        kind_counts[v] = kind_counts.get(v, 0) + 1
    
    print("\n节点类型分布:")
    for k, v in sorted(kind_counts.items()):
        print(f"  {k}: {v}")
        
driver.close()
