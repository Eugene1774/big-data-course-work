from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Sequence, Tuple

import frontmatter
from neo4j import Driver, GraphDatabase
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent
CARDS_DIR = PROJECT_ROOT / "knowledge_base" / "cards"
DEFAULT_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_USER = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME", "neo4j")
DEFAULT_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
DEFAULT_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
DEFAULT_LLM_BASE_URL = (
    os.getenv("LLM_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or os.getenv("SILICONFLOW_BASE_URL")
    or "https://api.siliconflow.com/v1"
)
DEFAULT_LLM_MODEL = (
    os.getenv("LLM_MODEL")
    or os.getenv("OPENAI_MODEL")
    or os.getenv("SILICONFLOW_MODEL")
    or "Qwen/Qwen2.5-7B-Instruct"
)

METRIC_ALIASES = {
    "cac": "获客成本",
    "customer acquisition cost": "获客成本",
    "获客成本": "获客成本",
    "ltv": "生命周期价值",
    "life time value": "生命周期价值",
    "lifetime value": "生命周期价值",
    "生命周期价值": "生命周期价值",
    "tam": "总可服务市场",
    "总可服务市场": "总可服务市场",
    "sam": "可服务可触达市场",
    "可服务可触达市场": "可服务可触达市场",
    "som": "可获取市场",
    "可获取市场": "可获取市场",
    "mvp": "最小可行产品",
    "最小可行产品": "最小可行产品",
    "usp": "独特价值主张",
    "独特价值主张": "独特价值主张",
}

PREDICATE_ALLOWLIST = {
    "PREREQ",
    "FIX_STRATEGY",
    "ASSOCIATED_WITH",
    "SUPPORTS",
    "CONSTRAINS",
    "INDICATES",
    "TRIGGERS",
    "INVOLVES",
}

PREREQ_HINTS = ("前置", "前提", "依赖", "基础是", "需要先", "先完成", "之前先")
FIX_HINTS = ("修复", "改进", "优化", "建议", "应当", "需要", "整改", "补齐", "解决方案")
FORMULA_PATTERN = re.compile(r"\$([^$]+)\$")
RULE_ID_PATTERN = re.compile(r"\bH\d+\b", re.IGNORECASE)
CODE_TOKEN_PATTERN = re.compile(r"\b[A-Z]{2,10}\b")
LOGGER = logging.getLogger(__name__)


def build_driver(uri: str, user: str, password: str) -> Driver:
    return GraphDatabase.driver(uri, auth=(user, password))


def clear_db(driver: Driver, database: str) -> None:
    with driver.session(database=database) as session:
        session.run("MATCH (n) DETACH DELETE n")


def initialize_constraints(driver: Driver, database: str) -> None:
    statements = [
        "CREATE CONSTRAINT knowledge_card_name_unique IF NOT EXISTS FOR (n:KnowledgeCard) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (n:Concept) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT metric_name_unique IF NOT EXISTS FOR (n:Metric) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT tag_name_unique IF NOT EXISTS FOR (n:Tag) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT rule_rule_id_unique IF NOT EXISTS FOR (n:Rule) REQUIRE n.rule_id IS UNIQUE",
    ]
    with driver.session(database=database) as session:
        for statement in statements:
            session.run(statement)


def normalize_label(label: str | None) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "", str(label or "KnowledgeCard"))
    if not cleaned:
        return "KnowledgeCard"
    if cleaned[0].isdigit():
        cleaned = f"Node_{cleaned}"
    return cleaned


def normalize_lookup_text(text: str) -> str:
    lowered = str(text or "").lower().replace("\ufeff", " ")
    lowered = re.sub(r"[^\w\u4e00-\u9fff]+", " ", lowered)
    return " ".join(lowered.split())


def ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,;/|]", value) if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def iter_card_files(cards_dir: Path) -> Iterable[Path]:
    markdown_files = sorted(cards_dir.glob("*.md")) + sorted(cards_dir.glob("*.markdown"))
    if markdown_files:
        yield from markdown_files
        return

    # Current repo stores 111 cards as JSON. Keep a fallback so the pipeline is runnable here.
    yield from sorted(cards_dir.glob("*.json"))


def parse_markdown_card(path: Path) -> Dict[str, Any]:
    post = frontmatter.load(path)
    metadata = dict(post.metadata)
    content = post.content.strip()
    return {
        "card_id": metadata.get("card_id") or metadata.get("id") or path.stem,
        "name": metadata.get("name") or metadata.get("title") or path.stem,
        "node_type": metadata.get("node_type") or "KnowledgeCard",
        "description": metadata.get("description") or content[:500],
        "content": content,
        "rule_ids": ensure_list(metadata.get("rule_id") or metadata.get("rule_ids")),
        "tags": ensure_list(metadata.get("tags")),
        "related_concepts": ensure_list(metadata.get("related_concepts")),
        "type": metadata.get("type"),
        "industry": metadata.get("industry"),
        "relations": metadata.get("relations") or [],
        "hyperedges": metadata.get("hyperedges") or [],
        "projects": metadata.get("projects") or [],
        "metadata": metadata,
        "source_file": path.name,
        "source_path": str(path),
    }


def parse_json_card(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata", {}))
    content = str(metadata.get("source_excerpt") or payload.get("description") or "").strip()
    related_concepts = ensure_list(payload.get("related_concepts"))
    if not related_concepts:
        related_concepts = ensure_list(payload.get("labels"))

    return {
        "card_id": payload.get("card_id") or payload.get("id") or path.stem,
        "name": payload.get("name") or payload.get("title") or path.stem,
        "node_type": payload.get("node_type") or "KnowledgeCard",
        "description": str(payload.get("description") or "").strip(),
        "content": content,
        "rule_ids": ensure_list(payload.get("rule_id") or payload.get("rule_ids")),
        "tags": ensure_list(payload.get("tags")),
        "related_concepts": related_concepts,
        "type": payload.get("type"),
        "industry": payload.get("industry"),
        "relations": payload.get("relations") or metadata.get("relations") or [],
        "hyperedges": payload.get("hyperedges") or metadata.get("hyperedges") or [],
        "projects": payload.get("projects") or metadata.get("projects") or [],
        "metadata": payload,
        "source_file": path.name,
        "source_path": str(path),
    }


def parse_card(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in {".md", ".markdown"}:
        return parse_markdown_card(path)
    if path.suffix.lower() == ".json":
        return parse_json_card(path)
    raise ValueError(f"Unsupported card format: {path}")


def extract_formulae(text: str) -> List[str]:
    formulae: List[str] = []
    for match in FORMULA_PATTERN.finditer(text or ""):
        formula = match.group(1).strip()
        if formula and formula not in formulae:
            formulae.append(formula)
    return formulae


def try_get_llm_client() -> tuple[OpenAI | None, str | None]:
    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("SILICONFLOW_API_KEY")
    )
    if not api_key:
        return None, None
    return OpenAI(api_key=api_key, base_url=DEFAULT_LLM_BASE_URL), DEFAULT_LLM_MODEL


def parse_llm_json_payload(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def heuristic_triple_extraction(card: Dict[str, Any]) -> List[Dict[str, str]]:
    triples: List[Dict[str, str]] = []
    subject = card["name"]
    sentences = re.split(r"[。！？\n]+", card.get("content", ""))

    for concept in card.get("related_concepts", [])[:8]:
        triples.append(
            {
                "subject": subject,
                "predicate": "ASSOCIATED_WITH",
                "object": concept,
                "evidence": "metadata.related_concepts",
            }
        )

    for sentence in sentences:
        snippet = sentence.strip()
        if len(snippet) < 8:
            continue
        if any(keyword in snippet for keyword in PREREQ_HINTS):
            triples.append(
                {
                    "subject": subject,
                    "predicate": "PREREQ",
                    "object": snippet[:120],
                    "evidence": snippet[:220],
                }
            )
        if any(keyword in snippet for keyword in FIX_HINTS):
            triples.append(
                {
                    "subject": subject,
                    "predicate": "FIX_STRATEGY",
                    "object": snippet[:120],
                    "evidence": snippet[:220],
                }
            )

    deduped: List[Dict[str, str]] = []
    seen = set()
    for triple in triples:
        key = (triple["subject"], triple["predicate"], triple["object"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(triple)
    return deduped[:16]


def extract_triples_with_llm(card: Dict[str, Any], client: OpenAI | None, model: str | None) -> List[Dict[str, str]]:
    if client is None or model is None:
        return heuristic_triple_extraction(card)

    content = (card.get("content") or "")[:6000]
    prompt = f"""
你是知识图谱抽取器。请从下面卡片正文中抽取结构化三元组，严格返回 JSON。

要求:
1. 重点抽取 PREREQ 和 FIX_STRATEGY。
2. 也可以抽取 SUPPORTS、CONSTRAINS、INDICATES、ASSOCIATED_WITH。
3. 只返回如下 JSON 结构:
{{
  "triples": [
    {{
      "subject": "实体",
      "predicate": "PREREQ",
      "object": "实体或短语",
      "evidence": "原文证据"
    }}
  ]
}}
4. 谓词必须是大写英文枚举。
5. 不要输出解释，不要使用 Markdown。

卡片名称: {card["name"]}
节点类型: {card["node_type"]}
标签: {", ".join(card.get("tags", []))}
相关概念: {", ".join(card.get("related_concepts", []))}

正文:
{content}
""".strip()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        payload = parse_llm_json_payload(response.choices[0].message.content or "")
        triples = payload.get("triples", [])
        cleaned: List[Dict[str, str]] = []
        for triple in triples:
            if not isinstance(triple, dict):
                continue
            subject = str(triple.get("subject", "")).strip()
            predicate = sanitize_predicate(triple.get("predicate", ""))
            obj = str(triple.get("object", "")).strip()
            evidence = str(triple.get("evidence", "")).strip()
            if subject and predicate and obj:
                cleaned.append(
                    {
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "evidence": evidence[:500],
                    }
                )
        return cleaned or heuristic_triple_extraction(card)
    except Exception:
        return heuristic_triple_extraction(card)


def sanitize_predicate(value: str) -> str:
    predicate = re.sub(r"[^A-Z_]", "_", str(value or "").upper()).strip("_")
    if not predicate:
        return "ASSOCIATED_WITH"
    if predicate not in PREDICATE_ALLOWLIST:
        return "ASSOCIATED_WITH"
    return predicate


def canonicalize_entity_name(name: str) -> str:
    stripped = str(name or "").strip()
    if not stripped:
        return ""

    if RULE_ID_PATTERN.fullmatch(stripped.upper()):
        return stripped.upper()

    normalized = normalize_lookup_text(stripped)
    if normalized in METRIC_ALIASES:
        return METRIC_ALIASES[normalized]
    return stripped


def infer_entity_label(name: str) -> str:
    canonical = canonicalize_entity_name(name)
    if RULE_ID_PATTERN.fullmatch(canonical):
        return "Rule"
    normalized = normalize_lookup_text(canonical)
    if normalized in METRIC_ALIASES or canonical in set(METRIC_ALIASES.values()):
        return "Metric"
    return "Concept"


def fetch_existing_entity_names(driver: Driver, database: str) -> Dict[str, List[str]]:
    known: Dict[str, List[str]] = {"Rule": [], "Metric": [], "Concept": [], "KnowledgeCard": []}
    query = """
    CALL {
        MATCH (n)
        WHERE n.name IS NOT NULL
        RETURN labels(n) AS labels, n.name AS value
        UNION
        MATCH (r:Rule)
        RETURN labels(r) AS labels, r.rule_id AS value
    }
    RETURN labels, value
    """
    with driver.session(database=database) as session:
        for record in session.run(query):
            labels = record.get("labels", [])
            value = str(record.get("value", "")).strip()
            if not value:
                continue
            if "Rule" in labels:
                known["Rule"].append(value)
            elif "Metric" in labels:
                known["Metric"].append(value)
            elif "Concept" in labels:
                known["Concept"].append(value)
            elif "KnowledgeCard" in labels:
                known["KnowledgeCard"].append(value)
    return known


def align_entity_name(name: str, known_names: Sequence[str]) -> str:
    canonical = canonicalize_entity_name(name)
    if not canonical:
        return ""

    if canonical in known_names:
        return canonical

    for known_name in known_names:
        if canonical == known_name:
            return known_name
        if len(canonical) >= 2 and (canonical in known_name or known_name in canonical):
            return known_name

    matches = get_close_matches(canonical, list(known_names), n=1, cutoff=0.84)
    if matches:
        return matches[0]

    best_name = canonical
    best_score = 0.0
    for known_name in known_names:
        score = SequenceMatcher(None, canonical, known_name).ratio()
        if score > best_score:
            best_score = score
            best_name = known_name
    if best_score >= 0.88:
        return best_name

    return canonical


def seed_alignment_names(cards: Sequence[Dict[str, Any]], existing: Dict[str, List[str]]) -> Dict[str, List[str]]:
    seeded = {
        "Rule": list(existing.get("Rule", [])),
        "Metric": list(existing.get("Metric", [])),
        "Concept": list(existing.get("Concept", [])),
        "KnowledgeCard": list(existing.get("KnowledgeCard", [])),
    }

    for metric_name in set(METRIC_ALIASES.values()):
        if metric_name not in seeded["Metric"]:
            seeded["Metric"].append(metric_name)

    for card in cards:
        if card["name"] not in seeded["KnowledgeCard"]:
            seeded["KnowledgeCard"].append(card["name"])
        for value in card.get("tags", []):
            if value and value not in seeded["Concept"]:
                seeded["Concept"].append(value)
        for value in card.get("related_concepts", []):
            if value and value not in seeded["Concept"]:
                seeded["Concept"].append(value)
        for rule_id in card.get("rule_ids", []):
            normalized_rule = canonicalize_entity_name(rule_id)
            if normalized_rule and normalized_rule not in seeded["Rule"]:
                seeded["Rule"].append(normalized_rule)

    return seeded


def extract_formula_entities(formula: str) -> List[str]:
    mentioned: List[str] = []
    normalized_formula = normalize_lookup_text(formula)

    for alias, canonical in METRIC_ALIASES.items():
        if alias in normalized_formula and canonical not in mentioned:
            mentioned.append(canonical)

    for token in CODE_TOKEN_PATTERN.findall(formula or ""):
        canonical = canonicalize_entity_name(token)
        if canonical and canonical not in mentioned:
            mentioned.append(canonical)

    for rule_id in RULE_ID_PATTERN.findall(formula or ""):
        normalized_rule = canonicalize_entity_name(rule_id)
        if normalized_rule not in mentioned:
            mentioned.append(normalized_rule)

    return mentioned


def group_rows_by_label(rows: Sequence[Dict[str, Any]], label_field: str) -> DefaultDict[str, List[Dict[str, Any]]]:
    grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[label_field]].append(row)
    return grouped


def ensure_record_list(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if isinstance(item, dict)]
    return []


def normalize_relation_type(value: Any) -> str:
    raw = re.sub(r"[^A-Z_]", "_", str(value or "").upper()).strip("_")
    alias = {
        "TRIGGER": "TRIGGERS",
        "TRIGGERS": "TRIGGERS",
        "INVOLVE": "INVOLVES",
        "INVOLVES": "INVOLVES",
        "PREREQUISITE": "PREREQ",
        "PREREQUISITES": "PREREQ",
        "PREREQ": "PREREQ",
    }
    normalized = alias.get(raw, raw)
    return sanitize_predicate(normalized)


def batch_upsert_card_nodes(driver: Driver, database: str, rows: Sequence[Dict[str, Any]]) -> None:
    for label, grouped_rows in group_rows_by_label(rows, "node_label").items():
        query = f"""
        UNWIND $rows AS row
        MERGE (n:{label} {{name: row.name}})
        SET n.card_id = row.card_id,
            n.node_type = row.node_type,
            n.description = row.description,
            n.content = row.content,
            n.type = row.type,
            n.industry = row.industry,
            n.source_file = row.source_file,
            n.source_path = row.source_path,
            n.tags = row.tags,
            n.related_concepts = row.related_concepts,
            n.metadata_json = row.metadata_json,
            n.updated_at = datetime()
        """
        with driver.session(database=database) as session:
            session.run(query, {"rows": grouped_rows})


def batch_attach_rules(driver: Driver, database: str, rows: Sequence[Dict[str, Any]]) -> None:
    for label, grouped_rows in group_rows_by_label(rows, "card_label").items():
        query = f"""
        UNWIND $rows AS row
        MATCH (n:{label} {{name: row.card_name}})
        MERGE (r:Rule {{rule_id: row.rule_id}})
        SET r.name = coalesce(r.name, row.rule_id),
            r.updated_at = datetime()
        MERGE (n)-[:EVALUATED_BY]->(r)
        """
        with driver.session(database=database) as session:
            session.run(query, {"rows": grouped_rows})


def batch_attach_associations(driver: Driver, database: str, rows: Sequence[Dict[str, Any]], target_label: str, target_key: str) -> None:
    for label, grouped_rows in group_rows_by_label(rows, "card_label").items():
        query = f"""
        UNWIND $rows AS row
        MATCH (n:{label} {{name: row.card_name}})
        MERGE (t:{target_label} {{{target_key}: row.target_name}})
        SET t.name = coalesce(t.name, row.target_name),
            t.updated_at = datetime()
        MERGE (n)-[:ASSOCIATED_WITH]->(t)
        """
        with driver.session(database=database) as session:
            session.run(query, {"rows": grouped_rows})


def batch_upsert_entities(driver: Driver, database: str, rows: Sequence[Dict[str, Any]]) -> None:
    grouped = group_rows_by_label(rows, "entity_label")
    for label, grouped_rows in grouped.items():
        if label == "Rule":
            query = """
            UNWIND $rows AS row
            MERGE (n:Rule {rule_id: row.key_value})
            SET n.name = coalesce(n.name, row.display_name),
                n.updated_at = datetime()
            """
        else:
            query = f"""
            UNWIND $rows AS row
            MERGE (n:{label} {{name: row.key_value}})
            SET n.updated_at = datetime()
            """
        with driver.session(database=database) as session:
            session.run(query, {"rows": grouped_rows})


def build_node_pattern(var_name: str, label: str, row_field: str) -> str:
    if label == "Rule":
        return f"({var_name}:Rule {{rule_id: row.{row_field}}})"
    return f"({var_name}:{label} {{name: row.{row_field}}})"


def batch_write_triples(driver: Driver, database: str, rows: Sequence[Dict[str, Any]]) -> None:
    grouped: DefaultDict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["predicate"], row["subject_label"], row["object_label"])].append(row)

    for (predicate, subject_label, object_label), grouped_rows in grouped.items():
        subject_pattern = build_node_pattern("s", subject_label, "subject_key")
        object_pattern = build_node_pattern("o", object_label, "object_key")
        query = f"""
        UNWIND $rows AS row
        MERGE {subject_pattern}
        SET s.name = coalesce(s.name, row.subject_name),
            s.updated_at = datetime()
        MERGE {object_pattern}
        SET o.name = coalesce(o.name, row.object_name),
            o.updated_at = datetime()
        MERGE (s)-[r:{predicate}]->(o)
        SET r.source_card = row.source_card,
            r.evidence = row.evidence,
            r.updated_at = datetime()
        """
        with driver.session(database=database) as session:
            session.run(query, {"rows": grouped_rows})


def batch_attach_formulae(driver: Driver, database: str, metric_rows: Sequence[Dict[str, Any]], rule_rows: Sequence[Dict[str, Any]]) -> None:
    if metric_rows:
        metric_query = """
        UNWIND $rows AS row
        MERGE (n:Metric {name: row.entity_name})
        SET n.formulae =
            CASE
                WHEN n.formulae IS NULL THEN [row.formula]
                WHEN row.formula IN n.formulae THEN n.formulae
                ELSE n.formulae + row.formula
            END,
            n.updated_at = datetime()
        """
        with driver.session(database=database) as session:
            session.run(metric_query, {"rows": metric_rows})

    if rule_rows:
        rule_query = """
        UNWIND $rows AS row
        MERGE (n:Rule {rule_id: row.rule_id})
        SET n.name = coalesce(n.name, row.rule_id),
            n.formulae =
            CASE
                WHEN n.formulae IS NULL THEN [row.formula]
                WHEN row.formula IN n.formulae THEN n.formulae
                ELSE n.formulae + row.formula
            END,
            n.updated_at = datetime()
        """
        with driver.session(database=database) as session:
            session.run(rule_query, {"rows": rule_rows})


def write_pipeline_rows_single_tx(driver: Driver, database: str, rows: Dict[str, List[Dict[str, Any]]]) -> None:
    def _write(tx, payload: Dict[str, List[Dict[str, Any]]]) -> None:
        for label, grouped_rows in group_rows_by_label(payload.get("card_rows", []), "node_label").items():
            query = f"""
            UNWIND $rows AS row
            MERGE (n:{label} {{name: row.name}})
            SET n.card_id = row.card_id,
                n.node_type = row.node_type,
                n.description = row.description,
                n.content = row.content,
                n.type = row.type,
                n.industry = row.industry,
                n.source_file = row.source_file,
                n.source_path = row.source_path,
                n.tags = row.tags,
                n.related_concepts = row.related_concepts,
                n.metadata_json = row.metadata_json,
                n.updated_at = datetime()
            """
            tx.run(query, {"rows": grouped_rows}).consume()

        for label, grouped_rows in group_rows_by_label(payload.get("rule_rows", []), "card_label").items():
            query = f"""
            UNWIND $rows AS row
            MATCH (n:{label} {{name: row.card_name}})
            MERGE (r:Rule {{rule_id: row.rule_id}})
            SET r.name = coalesce(r.name, row.rule_id),
                r.updated_at = datetime()
            MERGE (n)-[:EVALUATED_BY]->(r)
            """
            tx.run(query, {"rows": grouped_rows}).consume()

        for label, grouped_rows in group_rows_by_label(payload.get("tag_rows", []), "card_label").items():
            query = f"""
            UNWIND $rows AS row
            MATCH (n:{label} {{name: row.card_name}})
            MERGE (t:Tag {{name: row.target_name}})
            SET t.updated_at = datetime()
            MERGE (n)-[:ASSOCIATED_WITH]->(t)
            """
            tx.run(query, {"rows": grouped_rows}).consume()

        for label, grouped_rows in group_rows_by_label(payload.get("concept_rows", []), "card_label").items():
            query = f"""
            UNWIND $rows AS row
            MATCH (n:{label} {{name: row.card_name}})
            MERGE (c:Concept {{name: row.target_name}})
            SET c.updated_at = datetime()
            MERGE (n)-[:ASSOCIATED_WITH]->(c)
            """
            tx.run(query, {"rows": grouped_rows}).consume()

        for label, grouped_rows in group_rows_by_label(payload.get("entity_rows", []), "entity_label").items():
            if label == "Rule":
                query = """
                UNWIND $rows AS row
                MERGE (n:Rule {rule_id: row.key_value})
                SET n.name = coalesce(n.name, row.display_name),
                    n.updated_at = datetime()
                """
            else:
                query = f"""
                UNWIND $rows AS row
                MERGE (n:{label} {{name: row.key_value}})
                SET n.updated_at = datetime()
                """
            tx.run(query, {"rows": grouped_rows}).consume()

        grouped_triples: DefaultDict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        for row in payload.get("triple_rows", []):
            grouped_triples[(row["predicate"], row["subject_label"], row["object_label"])].append(row)
        for (predicate, subject_label, object_label), grouped_rows in grouped_triples.items():
            subject_pattern = build_node_pattern("s", subject_label, "subject_key")
            object_pattern = build_node_pattern("o", object_label, "object_key")
            query = f"""
            UNWIND $rows AS row
            MERGE {subject_pattern}
            SET s.name = coalesce(s.name, row.subject_name),
                s.updated_at = datetime()
            MERGE {object_pattern}
            SET o.name = coalesce(o.name, row.object_name),
                o.updated_at = datetime()
            MERGE (s)-[r:{predicate}]->(o)
            SET r.source_card = row.source_card,
                r.evidence = row.evidence,
                r.updated_at = datetime()
            """
            tx.run(query, {"rows": grouped_rows}).consume()

        metric_rows = payload.get("formula_metric_rows", [])
        if metric_rows:
            metric_query = """
            UNWIND $rows AS row
            MERGE (n:Metric {name: row.entity_name})
            SET n.formulae =
                CASE
                    WHEN n.formulae IS NULL THEN [row.formula]
                    WHEN row.formula IN n.formulae THEN n.formulae
                    ELSE n.formulae + row.formula
                END,
                n.updated_at = datetime()
            """
            tx.run(metric_query, {"rows": metric_rows}).consume()

        formula_rule_rows = payload.get("formula_rule_rows", [])
        if formula_rule_rows:
            rule_query = """
            UNWIND $rows AS row
            MERGE (n:Rule {rule_id: row.rule_id})
            SET n.name = coalesce(n.name, row.rule_id),
                n.formulae =
                CASE
                    WHEN n.formulae IS NULL THEN [row.formula]
                    WHEN row.formula IN n.formulae THEN n.formulae
                    ELSE n.formulae + row.formula
                END,
                n.updated_at = datetime()
            """
            tx.run(rule_query, {"rows": formula_rule_rows}).consume()

        project_rows = payload.get("project_rows", [])
        if project_rows:
            project_query = """
            UNWIND $rows AS row
            MERGE (p:Project {id: row.id})
            SET p.name = row.name,
                p.description = row.description,
                p.metadata_json = row.metadata_json,
                p.updated_at = datetime()
            """
            tx.run(project_query, {"rows": project_rows}).consume()

        for edge_label, grouped_rows in group_rows_by_label(payload.get("hyperedge_rows", []), "edge_label").items():
            hyperedge_query = f"""
            UNWIND $rows AS row
            MERGE (he:{edge_label} {{id: row.id}})
            SET he.name = row.name,
                he.edge_type = row.edge_type,
                he.rule_triggered = row.rule_triggered,
                he.logical_evaluation = row.logical_evaluation,
                he.impact = row.impact,
                he.metadata_json = row.metadata_json,
                he.updated_at = datetime()
            """
            tx.run(hyperedge_query, {"rows": grouped_rows}).consume()

        trigger_rows = payload.get("project_trigger_rows", [])
        if trigger_rows:
            trigger_query = """
            UNWIND $rows AS row
            MERGE (p:Project {id: row.project_id})
            ON CREATE SET p.name = coalesce(row.project_name, row.project_id),
                          p.created_by = "kg_pipeline"
            MERGE (he:Risk_Pattern_Edge {id: row.edge_id})
            ON CREATE SET he.name = coalesce(row.edge_name, row.edge_id),
                          he.edge_type = "Risk_Pattern_Edge",
                          he.created_by = "kg_pipeline"
            MERGE (p)-[:TRIGGERS]->(he)
            """
            tx.run(trigger_query, {"rows": trigger_rows}).consume()

        involves_rows = payload.get("hyperedge_involves_rows", [])
        if involves_rows:
            involves_query = """
            UNWIND $rows AS row
            MERGE (he:Risk_Pattern_Edge {id: row.edge_id})
            ON CREATE SET he.name = coalesce(row.edge_name, row.edge_id),
                          he.edge_type = "Risk_Pattern_Edge",
                          he.created_by = "kg_pipeline"
            MERGE (c:Concept {name: row.concept_name})
            SET c.updated_at = datetime()
            MERGE (he)-[:INVOLVES]->(c)
            """
            tx.run(involves_query, {"rows": involves_rows}).consume()

        prereq_rows = payload.get("prereq_rows", [])
        if prereq_rows:
            prereq_query = """
            UNWIND $rows AS row
            MERGE (a:Concept {name: row.a_name})
            MERGE (b:Concept {name: row.b_name})
            SET a.updated_at = datetime(),
                b.updated_at = datetime()
            MERGE (a)-[:PREREQ]->(b)
            """
            tx.run(prereq_query, {"rows": prereq_rows}).consume()

    with driver.session(database=database) as session:
        session.execute_write(_write, rows)


def prepare_pipeline_rows(cards: Sequence[Dict[str, Any]], known_names: Dict[str, List[str]], client: OpenAI | None, model: str | None) -> Dict[str, List[Dict[str, Any]]]:
    card_rows: List[Dict[str, Any]] = []
    rule_rows: List[Dict[str, Any]] = []
    tag_rows: List[Dict[str, Any]] = []
    concept_rows: List[Dict[str, Any]] = []
    entity_rows: List[Dict[str, Any]] = []
    triple_rows: List[Dict[str, Any]] = []
    formula_metric_rows: List[Dict[str, Any]] = []
    formula_rule_rows: List[Dict[str, Any]] = []
    project_rows: List[Dict[str, Any]] = []
    hyperedge_rows: List[Dict[str, Any]] = []
    project_trigger_rows: List[Dict[str, Any]] = []
    hyperedge_involves_rows: List[Dict[str, Any]] = []
    prereq_rows: List[Dict[str, Any]] = []
    seen_projects: set[Tuple[str, str]] = set()
    seen_hyperedges: set[Tuple[str, str]] = set()
    seen_triggers: set[Tuple[str, str]] = set()
    seen_involves: set[Tuple[str, str]] = set()
    seen_prereq: set[Tuple[str, str]] = set()

    seen_entities = {
        "Rule": set(known_names["Rule"]),
        "Metric": set(known_names["Metric"]),
        "Concept": set(known_names["Concept"]),
        "KnowledgeCard": set(known_names["KnowledgeCard"]),
    }

    for card in cards:
        node_label = normalize_label(card["node_type"])
        formulas = extract_formulae(card.get("content", ""))
        triples = extract_triples_with_llm(card, client, model)

        card_rows.append(
            {
                "node_label": node_label,
                "name": card["name"],
                "card_id": card["card_id"],
                "node_type": card["node_type"],
                "description": card["description"],
                "content": card["content"],
                "type": card["type"],
                "industry": card["industry"],
                "source_file": card["source_file"],
                "source_path": card["source_path"],
                "tags": card["tags"],
                "related_concepts": card["related_concepts"],
                "metadata_json": json.dumps(card["metadata"], ensure_ascii=False),
            }
        )

        for rule_id in card.get("rule_ids", []):
            normalized_rule = align_entity_name(rule_id, known_names["Rule"])
            rule_rows.append(
                {
                    "card_label": node_label,
                    "card_name": card["name"],
                    "rule_id": normalized_rule,
                }
            )
            seen_entities["Rule"].add(normalized_rule)

        for tag in card.get("tags", []):
            aligned_tag = align_entity_name(tag, known_names["Concept"])
            tag_rows.append(
                {
                    "card_label": node_label,
                    "card_name": card["name"],
                    "target_name": aligned_tag,
                }
            )
            seen_entities["Concept"].add(aligned_tag)

        for concept in card.get("related_concepts", []):
            aligned_concept = align_entity_name(concept, known_names["Concept"])
            concept_rows.append(
                {
                    "card_label": node_label,
                    "card_name": card["name"],
                    "target_name": aligned_concept,
                }
            )
            seen_entities["Concept"].add(aligned_concept)

        for triple in triples:
            subject_text = str(triple["subject"] or "").strip()
            # Keep card-centered edges attached to the real card node, avoid isolated card stars.
            if normalize_lookup_text(subject_text) == normalize_lookup_text(card["name"]):
                subject_label = node_label
                subject_name = card["name"]
            else:
                subject_label = infer_entity_label(subject_text)
                subject_name = align_entity_name(
                    subject_text,
                    list(seen_entities[subject_label]) + known_names.get(subject_label, []),
                )

            object_label = infer_entity_label(triple["object"])
            object_name = align_entity_name(triple["object"], list(seen_entities[object_label]) + known_names.get(object_label, []))

            seen_entities.setdefault(subject_label, set()).add(subject_name)
            seen_entities.setdefault(object_label, set()).add(object_name)

            entity_rows.append(
                {
                    "entity_label": subject_label,
                    "key_value": subject_name if subject_label != "Rule" else subject_name,
                    "display_name": subject_name,
                }
            )
            entity_rows.append(
                {
                    "entity_label": object_label,
                    "key_value": object_name if object_label != "Rule" else object_name,
                    "display_name": object_name,
                }
            )

            triple_rows.append(
                {
                    "predicate": sanitize_predicate(triple["predicate"]),
                    "subject_label": subject_label,
                    "subject_key": subject_name,
                    "subject_name": subject_name,
                    "object_label": object_label,
                    "object_key": object_name,
                    "object_name": object_name,
                    "source_card": card["name"],
                    "evidence": triple.get("evidence", "")[:500],
                }
            )

        for project in ensure_record_list(card.get("projects")):
            project_id = str(project.get("id") or project.get("project_id") or project.get("name") or "").strip()
            project_name = str(project.get("name") or project_id).strip()
            if not project_id:
                continue
            project_key = ("Project", project_id)
            if project_key not in seen_projects:
                seen_projects.add(project_key)
                project_rows.append(
                    {
                        "id": project_id,
                        "name": project_name,
                        "description": str(project.get("description") or ""),
                        "metadata_json": json.dumps(project, ensure_ascii=False),
                    }
                )

        for hyperedge in ensure_record_list(card.get("hyperedges")):
            edge_id = str(hyperedge.get("id") or hyperedge.get("hyperedge_id") or "").strip()
            if not edge_id:
                continue
            edge_type = str(hyperedge.get("edge_type") or "Risk_Pattern_Edge").strip() or "Risk_Pattern_Edge"
            edge_label = normalize_label(edge_type)
            edge_key = (edge_label, edge_id)
            if edge_key not in seen_hyperedges:
                seen_hyperedges.add(edge_key)
                hyperedge_rows.append(
                    {
                        "edge_label": edge_label,
                        "id": edge_id,
                        "name": str(hyperedge.get("name") or edge_id),
                        "edge_type": edge_type,
                        "rule_triggered": str(hyperedge.get("rule_triggered") or ""),
                        "logical_evaluation": str(hyperedge.get("logical_evaluation") or ""),
                        "impact": str(hyperedge.get("impact") or ""),
                        "metadata_json": json.dumps(hyperedge, ensure_ascii=False),
                    }
                )

            linked_projects = ensure_list(hyperedge.get("project_ids") or hyperedge.get("projects"))
            for project_id in linked_projects:
                project_id = str(project_id).strip()
                if not project_id:
                    continue
                trigger_key = (project_id, edge_id)
                if trigger_key in seen_triggers:
                    continue
                seen_triggers.add(trigger_key)
                project_trigger_rows.append(
                    {
                        "project_id": project_id,
                        "project_name": project_id,
                        "edge_id": edge_id,
                        "edge_name": str(hyperedge.get("name") or edge_id),
                    }
                )

            involved_concepts: List[str] = []
            nodes_involved = hyperedge.get("nodes_involved")
            if isinstance(nodes_involved, dict):
                involved_concepts.extend(str(v).strip() for v in nodes_involved.values() if str(v).strip())
            else:
                involved_concepts.extend(ensure_list(nodes_involved))
            involved_concepts.extend(ensure_list(hyperedge.get("participants")))
            for concept_name in involved_concepts:
                concept_name = str(concept_name).strip()
                if not concept_name:
                    continue
                involves_key = (edge_id, concept_name)
                if involves_key in seen_involves:
                    continue
                seen_involves.add(involves_key)
                hyperedge_involves_rows.append(
                    {
                        "edge_id": edge_id,
                        "edge_name": str(hyperedge.get("name") or edge_id),
                        "concept_name": concept_name,
                    }
                )

        for relation in ensure_record_list(card.get("relations")):
            relation_type = normalize_relation_type(
                relation.get("type") or relation.get("relation_type") or relation.get("predicate")
            )
            source = str(
                relation.get("source")
                or relation.get("from")
                or relation.get("subject")
                or relation.get("source_id")
                or ""
            ).strip()
            target = str(
                relation.get("target")
                or relation.get("to")
                or relation.get("object")
                or relation.get("target_id")
                or ""
            ).strip()
            if not source or not target:
                continue

            if relation_type == "PREREQ":
                prereq_key = (source, target)
                if prereq_key not in seen_prereq:
                    seen_prereq.add(prereq_key)
                    prereq_rows.append({"a_name": source, "b_name": target})
                continue

            if relation_type == "TRIGGERS":
                trigger_key = (source, target)
                if trigger_key not in seen_triggers:
                    seen_triggers.add(trigger_key)
                    project_trigger_rows.append(
                        {
                            "project_id": source,
                            "project_name": source,
                            "edge_id": target,
                            "edge_name": target,
                        }
                    )
                continue

            if relation_type == "INVOLVES":
                involves_key = (source, target)
                if involves_key not in seen_involves:
                    seen_involves.add(involves_key)
                    hyperedge_involves_rows.append(
                        {
                            "edge_id": source,
                            "edge_name": source,
                            "concept_name": target,
                        }
                    )
                continue

        for formula in formulas:
            for entity_name in extract_formula_entities(formula):
                label = infer_entity_label(entity_name)
                aligned_name = align_entity_name(entity_name, list(seen_entities[label]) + known_names.get(label, []))
                if label == "Metric":
                    formula_metric_rows.append({"entity_name": aligned_name, "formula": formula})
                elif label == "Rule":
                    formula_rule_rows.append({"rule_id": aligned_name, "formula": formula})

        rules_display = ", ".join(card.get("rule_ids", [])) if card.get("rule_ids") else "-"
        print(
            f"[PARSED] {card['source_file']} | node_type={card['node_type']} | "
            f"rules={rules_display} | triples={len(triples)} | formulas={len(formulas)}"
        )

    return {
        "card_rows": card_rows,
        "rule_rows": rule_rows,
        "tag_rows": tag_rows,
        "concept_rows": concept_rows,
        "entity_rows": entity_rows,
        "triple_rows": triple_rows,
        "formula_metric_rows": formula_metric_rows,
        "formula_rule_rows": formula_rule_rows,
        "project_rows": project_rows,
        "hyperedge_rows": hyperedge_rows,
        "project_trigger_rows": project_trigger_rows,
        "hyperedge_involves_rows": hyperedge_involves_rows,
        "prereq_rows": prereq_rows,
    }


def load_cards(cards_dir: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for index, path in enumerate(iter_card_files(cards_dir), start=1):
        if limit is not None and index > limit:
            break
        cards.append(parse_card(path))
    return cards


def run_pipeline(driver: Driver, database: str, cards_dir: Path, limit: int | None = None) -> None:
    client, model = try_get_llm_client()
    cards = load_cards(cards_dir, limit=limit)
    existing = fetch_existing_entity_names(driver, database)
    known_names = seed_alignment_names(cards, existing)
    rows = prepare_pipeline_rows(cards, known_names, client, model)
    relationship_count = (
        len(rows["rule_rows"])
        + len(rows["tag_rows"])
        + len(rows["concept_rows"])
        + len(rows["triple_rows"])
        + len(rows["project_trigger_rows"])
        + len(rows["hyperedge_involves_rows"])
        + len(rows["prereq_rows"])
    )
    LOGGER.info("About to write relationships into Neo4j: %d", relationship_count)
    if relationship_count == 0:
        LOGGER.warning(
            "No relationships extracted from payload. Upstream extraction may have returned only isolated nodes."
        )

    write_pipeline_rows_single_tx(driver, database, rows)

    print(
        f"\n[SUMMARY] cards={len(rows['card_rows'])} "
        f"rules={len(rows['rule_rows'])} triples={len(rows['triple_rows'])} "
        f"projects={len(rows['project_rows'])} hyperedges={len(rows['hyperedge_rows'])} "
        f"triggers={len(rows['project_trigger_rows'])} involves={len(rows['hyperedge_involves_rows'])} "
        f"prereq={len(rows['prereq_rows'])} "
        f"metric_formula_links={len(rows['formula_metric_rows'])} "
        f"rule_formula_links={len(rows['formula_rule_rows'])}"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Enhanced KG pipeline with LLM extraction and entity alignment.")
    parser.add_argument("--uri", default=DEFAULT_URI, help="Neo4j bolt URI")
    parser.add_argument("--user", default=DEFAULT_USER, help="Neo4j username")
    parser.add_argument("--password", default=DEFAULT_PASSWORD, help="Neo4j password")
    parser.add_argument("--database", default=DEFAULT_DATABASE, help="Neo4j database name")
    parser.add_argument("--cards-dir", default=str(CARDS_DIR), help="Directory containing card files")
    parser.add_argument("--clear", action="store_true", help="Clear the database before importing")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N files")
    args = parser.parse_args()

    cards_dir = Path(args.cards_dir)
    if not cards_dir.exists():
        raise SystemExit(f"Cards directory does not exist: {cards_dir}")

    driver = build_driver(args.uri, args.user, args.password)
    try:
        if args.clear:
            print("Clearing database...")
            clear_db(driver, args.database)

        print("Initializing constraints...")
        initialize_constraints(driver, args.database)

        print(f"Running pipeline for cards in: {cards_dir}")
        run_pipeline(driver, args.database, cards_dir, limit=args.limit)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
