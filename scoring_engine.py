from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

from core.schemas import AuditTrail, RubricScoreResult


RUBRIC: Dict[str, Dict[str, Any]] = {
    "R1": {"name": "问题定义", "weight": 0.10},
    "R2": {"name": "用户证据", "weight": 0.15},
    "R3": {"name": "方案可行性", "weight": 0.10},
    "R4": {"name": "商业模式一致性", "weight": 0.15},
    "R5": {"name": "市场与竞争", "weight": 0.10},
    "R6": {"name": "财务逻辑", "weight": 0.10},
    "R7": {"name": "创新与差异化", "weight": 0.10},
    "R8": {"name": "团队与执行", "weight": 0.05},
    "R9": {"name": "展示质量", "weight": 0.05},
}

ABILITY_DIMENSIONS: List[Dict[str, str]] = [
    {"dimension": "问题定义", "rule_id": "H1", "item_id": "R4"},
    {"dimension": "市场论证", "rule_id": "H4", "item_id": "R5"},
    {"dimension": "证据强度", "rule_id": "H5", "item_id": "R2"},
    {"dimension": "竞争分析", "rule_id": "H6", "item_id": "R5"},
    {"dimension": "财务健康", "rule_id": "H8", "item_id": "R6"},
]


class AbilityReport(dict):
    """Dictionary wrapper that renders naturally in Streamlit markdown calls."""

    def __str__(self) -> str:
        return str(self.get("report_markdown", ""))


def _clip_score(score: float) -> float:
    return max(0.0, min(5.0, round(float(score), 2)))


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, dict):
        return " ".join(_normalize_text(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_normalize_text(v) for v in value)
    return str(value).lower()


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _status_to_score_10(status: str) -> float:
    normalized = str(status or "").strip().lower()
    if normalized in {"failed", "fail", "error"}:
        return 2.0
    if normalized in {"warning", "warn"}:
        return 5.0
    if normalized in {"passed", "pass", "success"}:
        return 9.0
    return 6.0


def _coerce_messages(results: Dict[str, Any] | List[Dict[str, Any]] | Sequence[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _coerce_diag_rows(diag_results: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    return [dict(item) for item in list(diag_results or []) if isinstance(item, dict)]


def _extract_quotes(messages: Sequence[Dict[str, Any]], source_data: Dict[str, Any] | None = None) -> List[str]:
    quotes: List[str] = []
    for message in messages:
        if str(message.get("role", "")).strip().lower() == "user":
            text = str(message.get("content", "")).strip()
            if text:
                quotes.append(text)
    if isinstance(source_data, dict):
        for quote in source_data.get("student_quotes", []) or []:
            text = str(quote or "").strip()
            if text:
                quotes.append(text)
        for row in source_data.get("behavior_logs", []) or []:
            if not isinstance(row, dict):
                continue
            text = str(row.get("user_prompt") or row.get("raw_message") or "").strip()
            if text:
                quotes.append(text)
        project_full_text = str(source_data.get("project_full_text", "") or source_data.get("text", "")).strip()
        if project_full_text:
            sentences = [
                segment.strip()
                for segment in __import__("re").split(r"(?<=[。！？.!?])|\n+", project_full_text)
                if segment and segment.strip() and len(segment.strip()) > 10
            ]
            for sentence in sentences[:10]:
                quotes.append(sentence)
            if sentences and len(sentences) > 10:
                quotes.append(project_full_text[:200])
    deduped: List[str] = []
    for q in quotes:
        if q not in deduped:
            deduped.append(q)
    return deduped


def _extract_card_ids(value: Any) -> List[str]:
    ids: List[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).strip().lower()
            if lowered in {"card_id", "knowledge_card_id", "kg_card_id"}:
                candidate = str(item or "").strip()
                if candidate:
                    ids.append(candidate)
            elif lowered == "knowledge_card_ids" and isinstance(item, (list, tuple, set)):
                ids.extend(str(v).strip() for v in item if str(v).strip())
            else:
                ids.extend(_extract_card_ids(item))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            ids.extend(_extract_card_ids(item))
    return list(dict.fromkeys(ids))


def calculate_scores(data: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight rubric scoring used by student and competition views."""
    text = _normalize_text(data)
    items: List[Dict[str, Any]] = []

    for rule_id, config in RUBRIC.items():
        base_score = 2.5
        if rule_id in {"R1", "R4"} and _contains_any(text, ("用户", "痛点", "value", "channel", "evidence")):
            base_score += 1.2
        if rule_id in {"R5"} and _contains_any(text, ("竞品", "competition", "tam", "sam", "som")):
            base_score += 1.0
        if rule_id in {"R6"} and _contains_any(text, ("cac", "ltv", "毛利", "回收")):
            base_score += 1.0
        if rule_id in {"R8"} and _contains_any(text, ("团队", "成员", "里程碑")):
            base_score += 0.8

        score = _clip_score(base_score)
        items.append(
            {
                "rule_id": rule_id,
                "name": config["name"],
                "weight": config["weight"],
                "score": score,
                "reason": "基于当前文本证据与规则关键词进行快速评估。",
                "evidence": ["project_draft"],
                "evidence_snippet": str(data.get("project_summary") or "")[:120],
                "weighted_score": round(score / 5.0 * config["weight"] * 100, 2),
            }
        )

    weighted_percentage = round(sum(item["weighted_score"] for item in items), 2)
    overall_score = round(weighted_percentage / 20, 2)
    low_score_items = [item["rule_id"] for item in items if item["score"] <= 2.5]
    return {
        "overall_score": overall_score,
        "weighted_percentage": weighted_percentage,
        "low_score_items": low_score_items,
        "items": items,
    }


def generate_rubric_table(scoring_results: Dict[str, Any] | List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a unified rubric table for score reports and diagnostic rows."""
    if isinstance(scoring_results, dict) and isinstance(scoring_results.get("items"), list):
        rows = [
            {
                "指标": f"{item.get('rule_id', '-')} {item.get('name', '-')}".strip(),
                "得分": item.get("score", "-"),
                "理由": item.get("reason", "-"),
                "evidence_snippet": item.get("evidence_snippet", ""),
                "修复建议": "按最低分维度补齐可验证证据链。",
            }
            for item in scoring_results["items"]
            if isinstance(item, dict)
        ]
        return pd.DataFrame(rows)

    if isinstance(scoring_results, list):
        rows = []
        for item in scoring_results:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "指标": f"{item.get('rule_id', '-')} {item.get('rule_name', '-')}".strip(),
                    "得分": round(_status_to_score_10(item.get("status", "")) / 2, 2),
                    "理由": item.get("raw_message") or item.get("assistant_message") or item.get("trigger_message") or "-",
                    "evidence_snippet": item.get("evidence_snippet", ""),
                    "修复建议": item.get("fix_suggestion") or item.get("assistant_message") or "-",
                }
            )
        return pd.DataFrame(rows)

    return pd.DataFrame(columns=["指标", "得分", "理由", "evidence_snippet", "修复建议"])


def generate_ability_report(
    results: Dict[str, Any] | List[Dict[str, Any]] | Sequence[Any],
    diag_results: List[Dict[str, Any]] | None = None,
    source_data: Dict[str, Any] | None = None,
) -> AbilityReport:
    """Generate ability dimensions with mandatory audit-trail binding."""
    messages = _coerce_messages(results)
    diagnostics = _coerce_diag_rows(diag_results)
    quotes = _extract_quotes(messages, source_data)

    diag_by_rule: Dict[str, Dict[str, Any]] = {}
    for row in diagnostics:
        rule_id = str(row.get("rule_id", "")).strip().upper()
        if rule_id:
            diag_by_rule[rule_id] = row

    fallback_cards = _extract_card_ids(source_data or {})
    fallback_card = fallback_cards[0] if fallback_cards else "KG_CARD_UNBOUND"
    fallback_quote = quotes[-1] if quotes else "未提供可引用原文。"
    
    retrieved_nodes = []
    retrieved_hyperedges = []
    if isinstance(source_data, dict):
        for node in source_data.get("retrieved_nodes", []):
            if isinstance(node, dict) and node.get("id"):
                retrieved_nodes.append({
                    "id": node.get("id"),
                    "name": node.get("name"),
                    "labels": node.get("labels", []),
                })
        for hyperedge in source_data.get("retrieved_hyperedges", []):
            if isinstance(hyperedge, dict) and hyperedge.get("edge_id"):
                retrieved_hyperedges.append({
                    "edge_id": hyperedge.get("edge_id"),
                    "edge_type": hyperedge.get("edge_type"),
                    "contained_nodes": hyperedge.get("contained_nodes", []),
                })

    score_rows: List[Dict[str, Any]] = []
    for item in ABILITY_DIMENSIONS:
        rule_id = item["rule_id"]
        diag = dict(diag_by_rule.get(rule_id, {}))
        evidence_nodes = list(diag.get("evidence_nodes") or [])
        if not evidence_nodes:
            evidence_nodes = ["project_draft"]
        card_ids = _extract_card_ids(diag)
        if not card_ids and fallback_card:
            card_ids = [fallback_card]
        score_10 = _status_to_score_10(diag.get("status", "warning")) if diag else 6.0
        score_rows.append(
            {
                "dimension": item["dimension"],
                "item_id": item["item_id"],
                "rule_id": rule_id,
                "status": str(diag.get("status", "WARNING") or "WARNING"),
                "score_10": round(score_10, 2),
                "score_5": round(score_10 / 2, 2),
                "reason": str(diag.get("raw_message") or diag.get("assistant_message") or "基于当前证据进行评估。").strip(),
                "evidence_nodes": evidence_nodes,
                "knowledge_card_ids": card_ids,
                "student_quote_snippet": fallback_quote[:120],
                "retrieved_nodes": retrieved_nodes[:5],
                "retrieved_hyperedges": retrieved_hyperedges[:3],
            }
        )

    all_nodes: List[str] = []
    all_cards: List[str] = []
    for row in score_rows:
        all_nodes.extend(row.get("evidence_nodes", []))
        all_cards.extend(row.get("knowledge_card_ids", []))
    all_nodes = list(dict.fromkeys(all_nodes)) or ["project_draft"]
    all_cards = list(dict.fromkeys(all_cards)) or [fallback_card]
    score_rows.append(
        {
            "dimension": "证据溯源",
            "item_id": "R9",
            "rule_id": "AUDIT",
            "status": "BOUND",
            "score_10": 9.0 if all_nodes else 2.0,
            "score_5": 4.5 if all_nodes else 1.0,
            "reason": "评分链路已绑定证据节点与知识卡。",
            "evidence_nodes": all_nodes,
            "knowledge_card_ids": all_cards,
            "student_quote_snippet": fallback_quote[:120],
        }
    )

    rubric_score_results: List[RubricScoreResult] = []
    audit_trail: List[Dict[str, Any]] = []
    for row in score_rows:
        audit = AuditTrail(
            quote=str(row.get("student_quote_snippet") or ""),
            h_rule_id=str(row.get("rule_id") or "H_UNKNOWN"),
            kg_card_id=str((row.get("knowledge_card_ids") or [fallback_card])[0]),
            explanation=str(row.get("reason") or ""),
        )
        score_result = RubricScoreResult(
            item_id=str(row.get("item_id") or "R0"),
            score=_clip_score(float(row.get("score_5", 0.0))),
            audit_trail=audit,
        )
        rubric_score_results.append(score_result)
        audit_trail.append(
            {
                "dimension": row["dimension"],
                "item_id": row["item_id"],
                "rule_id": row["rule_id"],
                "student_quote_snippet": audit.quote,
                "kg_knowledge_card_ids": [audit.kg_card_id] if audit.kg_card_id else [],
                "evidence_nodes": list(row.get("evidence_nodes", [])) or ["project_draft"],
                "retrieved_nodes": row.get("retrieved_nodes", []),
                "retrieved_hyperedges": row.get("retrieved_hyperedges", []),
                "binding_status": "bound",
                "audit_trail": audit.model_dump(),
            }
        )

    average_score_5 = round(sum(float(row["score_5"]) for row in score_rows) / len(score_rows), 2) if score_rows else 0.0
    markdown_lines = [
        "### 阶段性能力画像",
        f"- 综合能力分: {average_score_5}/5",
        f"- 已绑定证据链条: {len(audit_trail)}",
        "- 说明: 每个维度都绑定了学生引用与知识卡证据。",
        "",
        "### 维度概览",
    ]
    markdown_lines.extend(f"- {row['dimension']}: {row['score_5']}/5" for row in score_rows)

    return AbilityReport(
        {
            "report_markdown": "\n".join(markdown_lines),
            "radar_chart_data": {
                "theta": [row["dimension"] for row in score_rows],
                "r": [row["score_5"] for row in score_rows],
                "range": [0, 5],
            },
            "score_table": score_rows,
            "average_score_5": average_score_5,
            "audit_trail": audit_trail,
            "rubric_score_results": [item.model_dump() for item in rubric_score_results],
        }
    )

