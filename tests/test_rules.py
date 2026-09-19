from __future__ import annotations

import sys
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rules.checker import GraphRuleChecker, RuleStatus
from schema.case import KnowledgeGraphCase


def build_inconsistent_market_case() -> KnowledgeGraphCase:
    """Build a minimal mock project where SOM > TAM to trigger H4."""
    raw_case = {
        "case_id": "MOCK_H4_FAIL",
        "title": "逻辑混乱 Mock 项目",
        "description": "用于测试 H4 市场规模口径错误。",
        "nodes": [
            {
                "id": "project_mock_h4",
                "node_type": "Project",
                "name": "逻辑混乱 Mock 项目",
                "description": "一个故意设置为 SOM 大于 TAM 的测试项目。",
                "stage": "idea",
                "team_name": "Test Team",
                "tags": ["test"],
                "metadata": {},
            },
            {
                "id": "metric_mock_tam",
                "node_type": "Metric",
                "name": "TAM",
                "description": "总体可服务市场规模",
                "metric_key": "TAM",
                "value": 100.0,
                "unit": "CNY",
                "tags": ["market"],
                "metadata": {},
            },
            {
                "id": "metric_mock_sam",
                "node_type": "Metric",
                "name": "SAM",
                "description": "可触达细分市场规模",
                "metric_key": "SAM",
                "value": 80.0,
                "unit": "CNY",
                "tags": ["market"],
                "metadata": {},
            },
            {
                "id": "metric_mock_som",
                "node_type": "Metric",
                "name": "SOM",
                "description": "短期可获得市场规模",
                "metric_key": "SOM",
                "value": 120.0,
                "unit": "CNY",
                "tags": ["market"],
                "metadata": {},
            },
        ],
        "relations": [],
        "hyperedges": [],
    }
    return KnowledgeGraphCase.from_dict(raw_case)


class GraphRuleCheckerTests(unittest.TestCase):
    def test_verify_all_detects_h4_market_size_error(self) -> None:
        checker = GraphRuleChecker()
        case = build_inconsistent_market_case()

        report = checker.verify_all(case)
        h4_result = next((item for item in report.results if item.rule_id == "H4"), None)

        self.assertIsNotNone(h4_result, "H4 should be present in the verification report.")
        self.assertEqual(h4_result.status, RuleStatus.FAILED)
        self.assertTrue(h4_result.is_triggered)
        self.assertIn("TAM >= SAM >= SOM", h4_result.message)
        self.assertEqual(
            h4_result.fix_suggestion,
            "重新核算 TAM、SAM、SOM，确保三者是严格子集关系。",
        )


if __name__ == "__main__":
    unittest.main()
