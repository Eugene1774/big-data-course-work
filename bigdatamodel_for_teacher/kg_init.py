from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import frontmatter
from neo4j import Driver

from neo4j_resilience import close_managed_driver, get_managed_driver, validate_schema_or_raise
from schema.case import KnowledgeGraphCase


PROJECT_ROOT = Path(__file__).resolve().parent
CARDS_DIR = PROJECT_ROOT / "knowledge_base" / "cards"
CASES_DIR = PROJECT_ROOT / "processed_json"
DEFAULT_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_USER = os.getenv("NEO4J_USER", "neo4j")
DEFAULT_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
DEFAULT_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
LOGGER = logging.getLogger(__name__)


def clear_db(driver: Driver, database: str) -> None:
    with driver.session(database=database) as session:
        session.run("MATCH (n) DETACH DELETE n")


def initialize_constraints(driver: Driver, database: str) -> None:
    statements = [
        "CREATE CONSTRAINT knowledge_card_name_unique IF NOT EXISTS FOR (n:KnowledgeCard) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT card_name_unique IF NOT EXISTS FOR (n:Card) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT case_case_id_unique IF NOT EXISTS FOR (n:Case) REQUIRE n.case_id IS UNIQUE",
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


def resolve_node_labels(node_type: str | None) -> List[str]:
    primary_label = normalize_label(node_type)
    if primary_label in {"KnowledgeCard", "Card"}:
        return ["KnowledgeCard", "Card"]
    return [primary_label]


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
    yield from sorted(cards_dir.glob("*.json"))


def iter_case_files(cases_dir: Path) -> Iterable[Path]:
    for path in sorted(cases_dir.glob("*.json")):
        if path.name.startswith("._"):
            continue
        if path.name.startswith("_failed_runs_"):
            continue
        yield path


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
        "metadata": metadata,
        "source_file": path.name,
        "source_path": str(path),
    }


def parse_json_card(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    description = str(payload.get("description") or "").strip()
    metadata = dict(payload.get("metadata", {}))
    related_concepts = ensure_list(payload.get("related_concepts"))
    if not related_concepts:
        related_concepts = ensure_list(payload.get("labels"))

    return {
        "card_id": payload.get("card_id") or payload.get("id") or path.stem,
        "name": payload.get("name") or payload.get("title") or path.stem,
        "node_type": payload.get("node_type") or "KnowledgeCard",
        "description": description,
        "content": metadata.get("source_excerpt", "") or description,
        "rule_ids": ensure_list(payload.get("rule_id") or payload.get("rule_ids")),
        "tags": ensure_list(payload.get("tags")),
        "related_concepts": related_concepts,
        "type": payload.get("type"),
        "industry": payload.get("industry"),
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


def parse_case(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    case = KnowledgeGraphCase.from_dict(payload)
    source_file = str(payload.get("source_file") or path.name).strip()
    case_id = str(case.case_id).strip()
    title = str(case.title).strip()
    description = str(case.description).strip()
    if not case_id or not title:
        raise ValueError(f"Invalid case file: {path}")
    return {
        "case_id": case_id,
        "title": title,
        "description": description,
        "source_file": source_file or path.name,
        "source_path": str(path),
    }


def upsert_card_node(driver: Driver, database: str, card: Dict[str, Any]) -> None:
    labels = ":".join(resolve_node_labels(card["node_type"]))
    query = f"""
    MERGE (n:{labels} {{name: $name}})
    SET n.card_id = $card_id,
        n.node_type = $node_type,
        n.description = $description,
        n.content = $content,
        n.type = $type,
        n.industry = $industry,
        n.source_file = $source_file,
        n.source_path = $source_path,
        n.tags = $tags,
        n.related_concepts = $related_concepts,
        n.metadata_json = $metadata_json,
        n.updated_at = datetime()
    """
    params = {
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
    with driver.session(database=database) as session:
        session.run(query, params)


def upsert_case_node(driver: Driver, database: str, case: Dict[str, Any]) -> None:
    query = """
    MERGE (c:Case {case_id: $case_id})
    SET c.title = $title,
        c.description = $description,
        c.source_file = $source_file,
        c.source_path = $source_path,
        c.updated_at = datetime()
    """
    with driver.session(database=database) as session:
        session.run(query, case)


def attach_rules(driver: Driver, database: str, node_name: str, node_type: str, rule_ids: List[str]) -> None:
    if not rule_ids:
        return

    label = normalize_label(node_type)
    query = f"""
    MATCH (n:{label} {{name: $node_name}})
    MERGE (r:Rule {{rule_id: $rule_id}})
    ON CREATE SET r.created_at = datetime()
    MERGE (n)-[:EVALUATED_BY]->(r)
    """
    with driver.session(database=database) as session:
        for rule_id in rule_ids:
            session.run(query, {"node_name": node_name, "rule_id": rule_id})


def attach_associations(
    driver: Driver,
    database: str,
    node_name: str,
    node_type: str,
    tags: List[str],
    related_concepts: List[str],
) -> None:
    label = normalize_label(node_type)
    tag_query = f"""
    MATCH (n:{label} {{name: $node_name}})
    MERGE (t:Tag {{name: $tag}})
    MERGE (n)-[:ASSOCIATED_WITH]->(t)
    """
    concept_query = f"""
    MATCH (n:{label} {{name: $node_name}})
    MERGE (c:Concept {{name: $concept}})
    MERGE (n)-[:ASSOCIATED_WITH]->(c)
    """
    with driver.session(database=database) as session:
        for tag in tags:
            session.run(tag_query, {"node_name": node_name, "tag": tag})
        for concept in related_concepts:
            session.run(concept_query, {"node_name": node_name, "concept": concept})


def import_cards(driver: Driver, database: str, cards_dir: Path) -> None:
    imported = 0
    for path in iter_card_files(cards_dir):
        try:
            card = parse_card(path)
            upsert_card_node(driver, database, card)
            attach_rules(driver, database, card["name"], card["node_type"], card["rule_ids"])
            attach_associations(
                driver,
                database,
                card["name"],
                card["node_type"],
                card["tags"],
                card["related_concepts"],
            )
            imported += 1
            rules_display = ", ".join(card["rule_ids"]) if card["rule_ids"] else "-"
            print(
                f"[OK] {path.name} | node_type={card['node_type']} | rules={rules_display} | "
                f"tags={len(card['tags'])} | related={len(card['related_concepts'])}"
            )
        except Exception as exc:
            print(f"[FAIL] {path.name} | error={exc}")
            LOGGER.exception("Card import failed for %s", path)

    print(f"\nImported {imported} card files from {cards_dir}")


def import_cases(driver: Driver, database: str, cases_dir: Path) -> None:
    imported = 0
    for path in iter_case_files(cases_dir):
        try:
            case = parse_case(path)
            upsert_case_node(driver, database, case)
            imported += 1
            print(f"[OK] {path.name} | case_id={case['case_id']} | title={case['title']}")
        except Exception as exc:
            print(f"[FAIL] {path.name} | error={exc}")
            LOGGER.exception("Case import failed for %s", path)

    print(f"\nImported {imported} case files from {cases_dir}")


def build_driver(
    uri: str,
    user: str,
    password: str,
    database: str,
    *,
    validate_schema: bool = False,
) -> Driver:
    return get_managed_driver(
        uri,
        user,
        password,
        database,
        validate_schema=validate_schema,
    )


def main() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )

    parser = argparse.ArgumentParser(description="Initialize Neo4j knowledge graph from knowledge cards.")
    parser.add_argument("--uri", default=DEFAULT_URI, help="Neo4j bolt URI")
    parser.add_argument("--user", default=DEFAULT_USER, help="Neo4j username")
    parser.add_argument("--password", default=DEFAULT_PASSWORD, help="Neo4j password")
    parser.add_argument("--database", default=DEFAULT_DATABASE, help="Neo4j database name")
    parser.add_argument("--cards-dir", default=str(CARDS_DIR), help="Directory containing card files")
    parser.add_argument(
        "--cases-dir",
        default=str(CASES_DIR),
        help="Directory containing case JSON files (legacy D*.json or processed extraction JSON).",
    )
    parser.add_argument("--clear", action="store_true", help="Clear the database before importing")
    args = parser.parse_args()

    cards_dir = Path(args.cards_dir)
    if not cards_dir.exists():
        raise SystemExit(f"Cards directory does not exist: {cards_dir}")

    cases_dir = Path(args.cases_dir)
    if not cases_dir.exists():
        raise SystemExit(f"Cases directory does not exist: {cases_dir}")

    driver = build_driver(args.uri, args.user, args.password, args.database, validate_schema=False)
    try:
        if args.clear:
            print("Clearing database...")
            clear_db(driver, args.database)

        print("Initializing constraints...")
        initialize_constraints(driver, args.database)

        print(f"Importing cards from: {cards_dir}")
        import_cards(driver, args.database, cards_dir)

        print(f"Importing cases from: {cases_dir}")
        import_cases(driver, args.database, cases_dir)

        print("Validating schema...")
        validate_schema_or_raise(
            driver,
            args.database,
            uri=args.uri,
            user=args.user,
            password=args.password,
        )
    finally:
        close_managed_driver(args.uri, args.user, args.password)


if __name__ == "__main__":
    main()
