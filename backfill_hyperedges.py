from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name) or default).strip()


URI = _env("NEO4J_URI")
USER = _env("NEO4J_USER") or _env("NEO4J_USERNAME")
PASSWORD = _env("NEO4J_PASSWORD")
DB_NAME = _env("NEO4J_DATABASE", "neo4j")


STATS_CYPHER = """
MATCH (c)
WHERE c:Concept OR c:KnowledgeCard OR c:Method
WITH c,
     size([(c)--() | 1]) AS degree,
     size([(c)-[:PREREQ]-() | 1]) AS prereq_degree,
     size([(:Risk_Pattern_Edge)-[:INVOLVES]->(c) | 1]) + size([(:Value_Loop_Edge)-[:INVOLVES]->(c) | 1]) AS involves_degree
WITH
  count(c) AS concept_total,
  sum(CASE WHEN degree = 0 THEN 1 ELSE 0 END) AS isolated_total,
  sum(CASE WHEN prereq_degree = 0 AND involves_degree = 0 THEN 1 ELSE 0 END) AS missing_total
CALL {
  MATCH (p:Project)
  RETURN count(p) AS project_total
}
CALL {
  MATCH (he)
  WHERE he:Risk_Pattern_Edge OR he:Value_Loop_Edge
  RETURN count(he) AS hyperedge_total
}
CALL {
  MATCH ()-[r:TRIGGERS]->()
  RETURN count(r) AS triggers_total
}
CALL {
  MATCH ()-[r:INVOLVES]->()
  RETURN count(r) AS involves_total
}
CALL {
  MATCH ()-[r:PREREQ]->()
  RETURN count(r) AS prereq_total
}
RETURN
  concept_total,
  isolated_total,
  missing_total,
  project_total,
  hyperedge_total,
  triggers_total,
  involves_total,
  prereq_total
"""


BACKFILL_QUERIES = [
    # 1) Build demo projects by card industry.
    """
    MATCH (kc:KnowledgeCard)
    WITH DISTINCT coalesce(nullif(trim(kc.industry), ''), 'Unknown') AS industry
    MERGE (p:Project {name: '全校示例项目-' + industry})
    SET p.id = 'PRJ_' + replace(replace(industry, ' ', '_'), '/', '_'),
        p.generated_by = 'backfill_hyperedges',
        p.updated_at = datetime()
    """,
    # 2) Build one risk hyperedge per card and connect Project -> TRIGGERS -> Hyperedge -> INVOLVES -> Card.
    """
    MATCH (kc:KnowledgeCard)
    WITH kc, coalesce(kc.card_id, kc.id, kc.name, elementId(kc)) AS card_key,
         coalesce(nullif(trim(kc.industry), ''), 'Unknown') AS industry
    MERGE (he:Risk_Pattern_Edge {id: 'HE_RISK_' + card_key})
    SET he.name = '风险模式-' + coalesce(kc.name, card_key),
        he.edge_type = 'Risk_Pattern_Edge',
        he.rule_triggered = coalesce(he.rule_triggered, 'H1_BusinessModelConsistency'),
        he.generated_by = 'backfill_hyperedges',
        he.updated_at = datetime()
    MERGE (he)-[:INVOLVES]->(kc)
    MERGE (p:Project {name: '全校示例项目-' + industry})
    ON CREATE SET p.id = 'PRJ_' + replace(replace(industry, ' ', '_'), '/', '_'),
                  p.generated_by = 'backfill_hyperedges',
                  p.created_at = datetime()
    SET p.updated_at = datetime()
    MERGE (p)-[:TRIGGERS]->(he)
    """,
    # 3) Pull concept/method neighbors into the card hyperedge.
    """
    MATCH (kc:KnowledgeCard)-[:ASSOCIATED_WITH]->(n)
    WHERE n:Concept OR n:Method
    WITH kc, n, coalesce(kc.card_id, kc.id, kc.name, elementId(kc)) AS card_key
    MERGE (he:Risk_Pattern_Edge {id: 'HE_RISK_' + card_key})
    MERGE (he)-[:INVOLVES]->(n)
    """,
    # 4) Ensure residual Concept/Method nodes are also wrapped by at least one hyperedge.
    """
    MERGE (he:Value_Loop_Edge {id: 'HE_VALUE_CORE_BACKFILL'})
    SET he.name = '核心价值闭环-自动补全',
        he.edge_type = 'Value_Loop_Edge',
        he.generated_by = 'backfill_hyperedges',
        he.updated_at = datetime()
    MERGE (p:Project {name: '全校示例项目-核心链路'})
    ON CREATE SET p.id = 'PRJ_CORE_BACKFILL',
                  p.generated_by = 'backfill_hyperedges',
                  p.created_at = datetime()
    SET p.updated_at = datetime()
    MERGE (p)-[:TRIGGERS]->(he)
    WITH he
    MATCH (n)
    WHERE (n:Concept OR n:Method)
      AND size([(:Risk_Pattern_Edge)-[:INVOLVES]->(n) | 1]) + size([(:Value_Loop_Edge)-[:INVOLVES]->(n) | 1]) = 0
    MERGE (he)-[:INVOLVES]->(n)
    """,
    # 5) Backfill PREREQ from existing concept associations.
    """
    MATCH (a:Concept)-[:ASSOCIATED_WITH]->(b:Concept)
    WITH a, b LIMIT 5000
    MERGE (a)-[:PREREQ]->(b)
    """,
]


def _stats(session) -> Dict[str, int]:
    rec = session.run(STATS_CYPHER).single()
    if not rec:
        return {
            "concept_total": 0,
            "isolated_total": 0,
            "missing_total": 0,
            "project_total": 0,
            "hyperedge_total": 0,
            "triggers_total": 0,
            "involves_total": 0,
            "prereq_total": 0,
        }
    return {
        "concept_total": int(rec.get("concept_total") or 0),
        "isolated_total": int(rec.get("isolated_total") or 0),
        "missing_total": int(rec.get("missing_total") or 0),
        "project_total": int(rec.get("project_total") or 0),
        "hyperedge_total": int(rec.get("hyperedge_total") or 0),
        "triggers_total": int(rec.get("triggers_total") or 0),
        "involves_total": int(rec.get("involves_total") or 0),
        "prereq_total": int(rec.get("prereq_total") or 0),
    }


def _run_backfill(session) -> None:
    for query in BACKFILL_QUERIES:
        session.run(query).consume()


def main() -> None:
    if not URI or not USER or not PASSWORD:
        raise SystemExit("Missing Neo4j env vars: NEO4J_URI / NEO4J_USER(or NEO4J_USERNAME) / NEO4J_PASSWORD")

    print(f"Connecting: {URI} | database={DB_NAME} | user={USER}")
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    try:
        with driver.session(database=DB_NAME) as session:
            before = _stats(session)
            print("[BEFORE]", before)
            _run_backfill(session)
            after = _stats(session)
            print("[AFTER ]", after)
    finally:
        driver.close()


if __name__ == "__main__":
    main()

