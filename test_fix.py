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
    
    # 模拟修复后的去重逻辑
    node_map = {}
    for record in records:
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
            
            # 获取节点名称
            element_id = str(getattr(raw_node, "element_id", None) or getattr(raw_node, "id", None) or id(raw_node))
            payload = dict(raw_node)
            node_name = ""
            for key in ("name", "title", "project_name", "id", "rule_id", "case_id"):
                node_name = str(payload.get(key, "")).strip()
                if node_name:
                    break
            if not node_name:
                node_name = labels[0] if labels else "Node"
            
            # 修复后的去重逻辑
            if kind == "Concept" or any(label in ["Concept", "KnowledgeCard", "Method", "Tag", "Rule"] for label in labels):
                dedup_key = f"core:{node_name}"
            elif kind == "Hyperedge":
                dedup_key = f"hyperedge:{node_name}"
            else:
                dedup_key = f"project:{element_id}"
            
            node_map[dedup_key] = kind
    
    print(f"修复后去重节点数: {len(node_map)}")
    print("\n节点类型分布:")
    kind_counts = {}
    for k, v in node_map.items():
        kind_counts[v] = kind_counts.get(v, 0) + 1
    for k, v in sorted(kind_counts.items()):
        print(f"  {k}: {v}")
        
    # 检查边
    edge_count = 0
    for record in records:
        r = record.get("r")
        if r is not None:
            edge_count += 1
    print(f"\n边数: {edge_count}")
        
driver.close()
