from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scoring_engine import generate_ability_report


REAL_CASE_PATH = PROJECT_ROOT / "data" / "D006_Inconsistent.json"


class AbilityReportTests(unittest.TestCase):
    def test_ability_report_binds_every_dimension_to_evidence_nodes(self) -> None:
        payload = json.loads(REAL_CASE_PATH.read_text(encoding="utf-8"))
        messages = [
            {"role": "user", "content": "我们的竞品替代方案是什么？"},
            {"role": "user", "content": "如果 CAC 太高，我该先改渠道还是改转化？"},
        ]

        report = generate_ability_report(messages, [], payload)
        score_table = report.get("score_table", [])
        audit_trail = report.get("audit_trail", [])

        self.assertTrue(score_table, "score_table should not be empty.")
        self.assertEqual(len(score_table), 6, "five dimensions plus one audit dimension are expected.")
        self.assertEqual(len(audit_trail), len(score_table), "audit_trail should align with score_table rows.")
        self.assertTrue(
            all(row.get("evidence_nodes") for row in score_table),
            "Every ability dimension must be bound to non-empty evidence_nodes.",
        )

    def test_audit_trail_contains_quote_and_kg_card_binding(self) -> None:
        payload = json.loads(REAL_CASE_PATH.read_text(encoding="utf-8"))
        messages = [
            {"role": "user", "content": "我们的竞品替代方案是什么？"},
            {"role": "user", "content": "如果 CAC 太高，我该先改渠道还是改转化？"},
        ]

        report = generate_ability_report(messages, [], payload)
        audit_trail = report["audit_trail"]

        self.assertTrue(
            any(entry.get("student_quote_snippet") for entry in audit_trail),
            "At least one audit entry should carry a student quote snippet.",
        )
        self.assertTrue(
            any(entry.get("kg_knowledge_card_ids") for entry in audit_trail),
            "At least one audit entry should carry a KG knowledge card binding.",
        )

    def test_report_markdown_hides_backend_debug_tokens(self) -> None:
        payload = json.loads(REAL_CASE_PATH.read_text(encoding="utf-8"))
        messages = [
            {"role": "user", "content": "请解释我们的用户价值闭环。"},
        ]
        report = generate_ability_report(messages, [], payload)
        markdown = str(report.get("report_markdown", ""))

        self.assertNotIn("rule=", markdown)
        self.assertNotIn("evidence_nodes=", markdown)


if __name__ == "__main__":
    unittest.main()
