from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_AGENT_NAMES = [
    "student_learning_tutor_node",
    "project_coach_node",
    "competition_advisor_node",
    "assessment_assistant_node",
    "instructor_assistant_node",
]
EXPECTED_RULE_IDS = [f"H{i}" for i in range(1, 16)]


@dataclass
class Finding:
    category: str
    target: str
    status: str
    detail: str


def _load_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - surfaced in CLI
        return exc


def _field_names(model_class: Any) -> set[str]:
    if hasattr(model_class, "model_fields"):
        return set(model_class.model_fields.keys())
    if hasattr(model_class, "__fields__"):
        return set(model_class.__fields__.keys())
    return set()


def _ascii_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render_row(values: Iterable[str]) -> str:
        padded = [value.ljust(widths[index]) for index, value in enumerate(values)]
        return "| " + " | ".join(padded) + " |"

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    parts = [separator, render_row(headers), separator]
    parts.extend(render_row(row) for row in rows)
    parts.append(separator)
    return "\n".join(parts)


def scan_structure() -> List[Finding]:
    findings: List[Finding] = []

    required_layout = {
        "core": ["graph.py", "nodes.py", "router.py", "state.py"],
        "rules": ["checker.py"],
        "pages": ["student_portal.py", "teacher_portal.py", "admin_portal.py"],
        "kg": ["kg_init.py", "kg_pipeline.py", "kg_hypergraph.py"],
    }

    for folder_name, expected_files in required_layout.items():
        folder_path = PROJECT_ROOT / folder_name
        if folder_name == "kg":
            if folder_path.exists():
                findings.append(Finding("structure", str(folder_path.relative_to(PROJECT_ROOT)), "OK", "directory present"))
                for file_name in expected_files:
                    file_path = folder_path / file_name
                    findings.append(
                        Finding(
                            "structure",
                            str(file_path.relative_to(PROJECT_ROOT)),
                            "OK" if file_path.exists() else "MISSING",
                            "file present" if file_path.exists() else "expected file missing",
                        )
                    )
            else:
                findings.append(
                    Finding(
                        "structure",
                        "kg/",
                        "MISSING",
                        "expected kg/ directory is absent; root-level kg_*.py scripts detected instead"
                        if any((PROJECT_ROOT / name).exists() for name in expected_files)
                        else "expected kg/ directory is absent",
                    )
                )
                for file_name in expected_files:
                    root_level_path = PROJECT_ROOT / file_name
                    findings.append(
                        Finding(
                            "structure",
                            file_name,
                            "OK" if root_level_path.exists() else "MISSING",
                            "root-level fallback script present" if root_level_path.exists() else "script missing",
                        )
                    )
            continue

        findings.append(
            Finding(
                "structure",
                f"{folder_name}/",
                "OK" if folder_path.exists() else "MISSING",
                "directory present" if folder_path.exists() else "required directory missing",
            )
        )
        for file_name in expected_files:
            file_path = folder_path / file_name
            findings.append(
                Finding(
                    "structure",
                    str(file_path.relative_to(PROJECT_ROOT)),
                    "OK" if file_path.exists() else "MISSING",
                    "file present" if file_path.exists() else "required file missing",
                )
            )

    return findings


def validate_agents() -> tuple[List[Finding], int]:
    findings: List[Finding] = []
    module = _load_module("core.nodes")
    if isinstance(module, Exception):
        return [Finding("agent", "core.nodes", "ERROR", f"import failed: {module}")], 0

    implemented = 0
    for agent_name in EXPECTED_AGENT_NAMES:
        func = getattr(module, agent_name, None)
        if func is None:
            findings.append(Finding("agent", agent_name, "MISSING", "expected async agent node is absent"))
            continue
        if not inspect.iscoroutinefunction(func):
            findings.append(Finding("agent", agent_name, "WEAK", "agent exists but is not async"))
            continue
        implemented += 1
        findings.append(Finding("agent", agent_name, "OK", "async agent node present"))

    return findings, implemented


def _real_case_path() -> Path | None:
    candidate_dirs = [PROJECT_ROOT / "processed_json", PROJECT_ROOT / "data"]
    for data_dir in candidate_dirs:
        if not data_dir.exists():
            continue
        for path in sorted(data_dir.glob("*.json")):
            if path.name.startswith("._"):
                continue
            if path.name.startswith("_failed_runs_"):
                continue
            return path
    return None


def validate_rules() -> tuple[List[Finding], int]:
    findings: List[Finding] = []

    checker_module = _load_module("rules.checker")
    case_module = _load_module("schema.case")
    if isinstance(checker_module, Exception):
        return [Finding("rule", "rules.checker", "ERROR", f"import failed: {checker_module}")], 0
    if isinstance(case_module, Exception):
        return [Finding("rule", "schema.case", "ERROR", f"import failed: {case_module}")], 0

    GraphRuleChecker = getattr(checker_module, "GraphRuleChecker")
    RuleResult = getattr(checker_module, "RuleResult")
    KnowledgeGraphCase = getattr(case_module, "KnowledgeGraphCase")

    rule_result_fields = _field_names(RuleResult)
    required_fields = {"severity", "evidence_nodes"}
    missing_fields = sorted(required_fields - rule_result_fields)
    findings.append(
        Finding(
            "rule",
            "RuleResult",
            "OK" if not missing_fields else "MISSING",
            "required fields present" if not missing_fields else f"missing fields: {', '.join(missing_fields)}",
        )
    )

    case_path = _real_case_path()
    if case_path is None:
        findings.append(
            Finding(
                "rule",
                "processed_json/*.json | data/*.json",
                "ERROR",
                "no real case data found for validation",
            )
        )
        return findings, 0

    try:
        case = KnowledgeGraphCase.from_json_file(case_path)
        checker = GraphRuleChecker()
        report = checker.verify_all(case)
    except Exception as exc:
        findings.append(Finding("rule", str(case_path.relative_to(PROJECT_ROOT)), "ERROR", f"verification failed: {exc}"))
        return findings, 0

    results_by_rule = {getattr(item, "rule_id", ""): item for item in getattr(report, "results", [])}
    passed_rules = 0
    for rule_id in EXPECTED_RULE_IDS:
        result = results_by_rule.get(rule_id)
        if result is None:
            findings.append(Finding("rule", rule_id, "MISSING", f"rule missing from verify_all() report built from {case_path.name}"))
            continue

        result_fields = set(getattr(result, "model_dump", lambda: getattr(result, "dict")())().keys())
        missing_result_fields = sorted(required_fields - result_fields)
        if isinstance(result, RuleResult) and not missing_result_fields:
            passed_rules += 1
            findings.append(Finding("rule", rule_id, "OK", f"RuleResult verified against real case {case_path.name}"))
        else:
            detail_bits = []
            if not isinstance(result, RuleResult):
                detail_bits.append("result is not RuleResult")
            if missing_result_fields:
                detail_bits.append(f"missing fields: {', '.join(missing_result_fields)}")
            findings.append(Finding("rule", rule_id, "WEAK", "; ".join(detail_bits) or "rule result needs strengthening"))

    return findings, passed_rules


def build_summary(agent_count: int, rule_count: int) -> List[Finding]:
    agent_ratio = agent_count / len(EXPECTED_AGENT_NAMES) if EXPECTED_AGENT_NAMES else 0.0
    rule_ratio = rule_count / len(EXPECTED_RULE_IDS) if EXPECTED_RULE_IDS else 0.0
    completion_pct = round(((agent_ratio + rule_ratio) / 2.0) * 100, 2)

    return [
        Finding("summary", "implemented_agents", "INFO", f"{agent_count}/{len(EXPECTED_AGENT_NAMES)}"),
        Finding("summary", "verified_rules", "INFO", f"{rule_count}/{len(EXPECTED_RULE_IDS)}"),
        Finding("summary", "project_completion", "INFO", f"{completion_pct}%"),
    ]


def print_report(findings: List[Finding], agent_count: int, rule_count: int) -> int:
    structure_rows = [[f.target, f.status, f.detail] for f in findings if f.category == "structure"]
    agent_rows = [[f.target, f.status, f.detail] for f in findings if f.category == "agent"]
    rule_rows = [[f.target, f.status, f.detail] for f in findings if f.category == "rule"]
    summary_rows = [[f.target, f.status, f.detail] for f in build_summary(agent_count, rule_count)]

    print("\n[Validator] Project Structure")
    print(_ascii_table(["Target", "Status", "Detail"], structure_rows))
    print("\n[Validator] Agent Coverage")
    print(_ascii_table(["Target", "Status", "Detail"], agent_rows))
    print("\n[Validator] Rule Self-Check")
    print(_ascii_table(["Target", "Status", "Detail"], rule_rows))
    print("\n[Validator] Summary")
    print(_ascii_table(["Metric", "Status", "Value"], summary_rows))

    gaps = [
        finding
        for finding in findings
        if finding.status in {"MISSING", "WEAK", "ERROR"}
    ]
    if gaps:
        print("\n[Validator] Missing / Needs Strengthening")
        print(_ascii_table(["Category", "Target", "Status", "Detail"], [[g.category, g.target, g.status, g.detail] for g in gaps]))
        return 1

    print("\n[Validator] No structural gaps detected.")
    return 0


def main() -> int:
    sys.path.insert(0, str(PROJECT_ROOT))
    structure_findings = scan_structure()
    agent_findings, agent_count = validate_agents()
    rule_findings, rule_count = validate_rules()
    findings = [*structure_findings, *agent_findings, *rule_findings]
    return print_report(findings, agent_count, rule_count)


if __name__ == "__main__":
    raise SystemExit(main())
