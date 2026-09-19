from __future__ import annotations

from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


class TeacherDataService:
    """Service layer for teacher portal data loading and resilient parsing."""

    def __init__(self, rule_to_dimension: Dict[str, str] | None = None) -> None:
        self.rule_to_dimension = dict(rule_to_dimension or {})

    @staticmethod
    def _safe_dict(value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _safe_list(value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        return []

    @staticmethod
    def _safe_str(value: Any, default: str = "") -> str:
        if value is None:
            return default
        text = str(value).strip()
        return text if text else default

    @staticmethod
    def _dedupe_texts(items: Iterable[Any]) -> List[str]:
        seen = set()
        result: List[str] = []
        for item in items:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    def _resolve_class_name(self, details: Dict[str, Any]) -> str:
        class_name = self._safe_str(
            details.get("class_name")
            or details.get("class_id")
            or details.get("class")
        )
        return class_name or "默认班级"

    def _coerce_diagnostics(self, value: Any) -> List[Dict[str, Any]]:
        diagnostics: List[Dict[str, Any]] = []
        for item in self._safe_list(value):
            if isinstance(item, dict):
                rule_id = self._safe_str(item.get("rule_id")).upper()
                if rule_id:
                    normalized = dict(item)
                    normalized["rule_id"] = rule_id
                    diagnostics.append(normalized)
        return diagnostics

    def _extract_rubric_rows(self, details: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        rubric_block = details.get("rubric") or details.get("rubric_scores")
        score_snapshot = self._safe_dict(details.get("score_snapshot"))
        if not rubric_block:
            rubric_block = score_snapshot.get("rubric")

        if isinstance(rubric_block, dict):
            for criterion, payload in rubric_block.items():
                payload_dict = self._safe_dict(payload)
                rows.append(
                    {
                        "criterion": self._safe_str(criterion, "未命名维度"),
                        "score": payload_dict.get("score", payload_dict.get("value", 0)),
                        "max_score": payload_dict.get("max_score", 10),
                        "comment": self._safe_str(payload_dict.get("comment", "")),
                        "evidence_trace": self._safe_list(payload_dict.get("evidence_trace")),
                    }
                )
        elif isinstance(rubric_block, list):
            for item in rubric_block:
                item_dict = self._safe_dict(item)
                if not item_dict:
                    continue
                rows.append(
                    {
                        "criterion": self._safe_str(item_dict.get("criterion"), "未命名维度"),
                        "score": item_dict.get("score", item_dict.get("value", 0)),
                        "max_score": item_dict.get("max_score", 10),
                        "comment": self._safe_str(item_dict.get("comment", "")),
                        "evidence_trace": self._safe_list(item_dict.get("evidence_trace")),
                    }
                )
        return rows

    def _extract_evidence_trace(self, details: Dict[str, Any], rubric_rows: List[Dict[str, Any]]) -> List[str]:
        traces: List[str] = []
        for row in rubric_rows:
            traces.extend(row.get("evidence_trace", []))

        for key in (
            "evidence_trace",
            "student_quotes",
            "student_quote",
            "quoted_text",
            "source_excerpt",
            "evidence_snippet",
        ):
            value = details.get(key)
            if isinstance(value, list):
                traces.extend(value)
            elif isinstance(value, str):
                traces.append(value)
        return self._dedupe_texts(traces)

    def _extract_common_errors(
        self,
        details: Dict[str, Any],
        diagnostics: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        errors: List[Dict[str, Any]] = []
        common_block = details.get("common_errors")
        for item in self._safe_list(common_block):
            item_dict = self._safe_dict(item)
            if not item_dict:
                continue
            error_label = self._safe_str(item_dict.get("error") or item_dict.get("label"))
            if not error_label:
                continue
            errors.append(
                {
                    "error": error_label,
                    "count": int(item_dict.get("count", 1) or 1),
                    "impact": self._safe_str(item_dict.get("impact", "")),
                }
            )

        if errors:
            return errors

        fallback: List[Dict[str, Any]] = []
        for item in diagnostics:
            fallback.append(
                {
                    "error": self._safe_str(item.get("trigger_message"), self._safe_str(item.get("rule_id"), "未知错误")),
                    "count": 1,
                    "impact": self._safe_str(item.get("fix_suggestion", "")),
                }
            )
        return fallback

    def _parse_triggered_rules_json(self, raw_rules: Any) -> List[Dict[str, Any]]:
        if raw_rules is None:
            return []
        if isinstance(raw_rules, str):
            text = raw_rules.strip()
            if not text:
                return []
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                return []
        else:
            decoded = raw_rules

        if isinstance(decoded, dict):
            decoded = [decoded]
        if not isinstance(decoded, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for item in decoded:
            if isinstance(item, dict):
                rule_id = self._safe_str(item.get("rule_id")).upper()
                if not rule_id:
                    continue
                payload = dict(item)
                payload["rule_id"] = rule_id
                normalized.append(payload)
            elif isinstance(item, str):
                rule_id = item.strip().upper()
                if rule_id:
                    normalized.append({"rule_id": rule_id})
        return normalized

    def _load_records_from_sqlite(self, db_path: Path) -> List[Dict[str, Any]]:
        if not db_path.exists():
            return []

        try:
            with sqlite3.connect(db_path, timeout=10.0) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT
                        id,
                        timestamp,
                        student_id,
                        project_id,
                        agent_role,
                        user_input,
                        agent_response,
                        triggered_rules,
                        next_step,
                        session_id
                    FROM interaction_logs
                    ORDER BY id ASC
                    """
                ).fetchall()
        except sqlite3.Error:
            return []

        records: List[Dict[str, Any]] = []
        for row in rows:
            diagnostics = self._parse_triggered_rules_json(row["triggered_rules"])
            student_id = self._safe_str(row["student_id"])
            project_id = self._safe_str(row["project_id"])
            details = {
                "timestamp": self._safe_str(row["timestamp"]),
                "class_name": "默认班级",
                "role": "student",
                "user_id": student_id,
                "project_id": project_id,
                "project_name": project_id or "未命名项目",
                "stage": "",
                "user_prompt": self._safe_str(row["user_input"]),
                "assistant_reply": self._safe_str(row["agent_response"]),
                "diagnostic_results": diagnostics,
                "next_step": self._safe_str(row["next_step"]),
                "session_id": self._safe_str(row["session_id"]),
                "agent_role": self._safe_str(row["agent_role"]),
            }
            records.append(
                {
                    "level": "INFO",
                    "source_path": str(db_path),
                    "message": self._safe_str(row["agent_role"], "chat_turn"),
                    "details": details,
                }
            )
        return records

    def load_records(self, log_path: str, use_mock: bool = False) -> List[Dict[str, Any]]:
        if use_mock:
            return self.generate_mock_records()

        path = Path(log_path)
        if path.suffix.lower() == ".db":
            return self._load_records_from_sqlite(path)
        if not path.exists():
            return []

        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    records.append(payload)
        return records

    def get_class_options(self, records: List[Dict[str, Any]]) -> List[str]:
        class_names = set()
        for record in records:
            details = self._safe_dict(record.get("details"))
            if not details:
                continue
            class_names.add(self._resolve_class_name(details))
        result = ["全部班级"] + sorted(name for name in class_names if name)
        return result if len(result) > 1 else ["全部班级", "默认班级"]

    def build_rule_dataframe(self, records: List[Dict[str, Any]], selected_class: str = "全部班级") -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for record in records:
            details = self._safe_dict(record.get("details"))
            if not details:
                continue
            class_name = self._resolve_class_name(details)
            if selected_class != "全部班级" and class_name != selected_class:
                continue

            diagnostics = self._coerce_diagnostics(details.get("diagnostic_results"))
            if not diagnostics:
                continue
            base_row = {
                "timestamp": details.get("timestamp"),
                "stage": details.get("stage"),
                "project_id": details.get("project_id"),
                "user_id": details.get("user_id"),
                "class_name": class_name,
                "project_name": details.get("project_name") or details.get("project_upload_name"),
                "message": record.get("message"),
            }
            for item in diagnostics:
                row = dict(base_row)
                row.update(item)
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["rule_id"] = df["rule_id"].astype(str).str.upper()
        df = df[df["rule_id"].str.startswith("H", na=False)].copy()
        if df.empty:
            return df
        df["dimension"] = df["rule_id"].map(self.rule_to_dimension).fillna("Other")
        return df

    def build_student_profiles(
        self,
        records: List[Dict[str, Any]],
        selected_class: str = "全部班级",
    ) -> List[Dict[str, Any]]:
        profiles: Dict[str, Dict[str, Any]] = {}

        for index, record in enumerate(records):
            if self._safe_str(record.get("level")).upper() != "INFO":
                continue

            details = self._safe_dict(record.get("details"))
            if not details:
                continue

            class_name = self._resolve_class_name(details)
            if selected_class != "全部班级" and class_name != selected_class:
                continue

            user_id = self._safe_str(details.get("user_id"))
            project_id = self._safe_str(details.get("project_id"))
            project_name = self._safe_str(details.get("project_name") or details.get("project_upload_name"))

            student_key = user_id or project_id or project_name or f"log_student_{index + 1}"
            student_name = user_id or project_name or project_id or f"学生 {index + 1}"

            profile = profiles.setdefault(
                student_key,
                {
                    "student_key": student_key,
                    "student_name": student_name,
                    "student_label": student_name,
                    "class_name": class_name,
                    "user_id": user_id,
                    "project_id": project_id,
                    "project_name": project_name,
                    "latest_stage": "",
                    "latest_timestamp": "",
                    "behavior_logs": [],
                    "diagnostic_results": [],
                    "triggered_rules_history": [],
                    "student_quotes": [],
                    "evidence_trace": [],
                    "common_errors": [],
                    "rubric_rows": [],
                    "project_summary": "",
                    "project_full_text": "",
                    "score_snapshot": {},
                },
            )

            profile["latest_stage"] = self._safe_str(details.get("stage"), profile["latest_stage"])
            profile["latest_timestamp"] = self._safe_str(details.get("timestamp"), profile["latest_timestamp"])
            if project_name:
                profile["project_name"] = project_name

            profile["behavior_logs"].append(
                {
                    "timestamp": details.get("timestamp"),
                    "event_type": record.get("message"),
                    "stage": details.get("stage"),
                    "user_prompt": details.get("user_prompt", ""),
                    "assistant_reply": details.get("assistant_reply", ""),
                    "project_excerpt": self._safe_str(details.get("project_full_text"))[:220],
                    "raw_message": details.get("message", ""),
                }
            )

            user_prompt = self._safe_str(details.get("user_prompt"))
            if user_prompt:
                profile["student_quotes"].append(user_prompt)

            project_text = self._safe_str(details.get("project_full_text"))
            if project_text:
                profile["project_full_text"] = project_text
                profile["project_summary"] = project_text[:300]

            diagnostics = self._coerce_diagnostics(details.get("diagnostic_results"))
            if diagnostics:
                profile["diagnostic_results"] = diagnostics
                profile["triggered_rules_history"].extend(diagnostics)

            rubric_rows = self._extract_rubric_rows(details)
            if rubric_rows:
                profile["rubric_rows"] = rubric_rows

            evidence_trace = self._extract_evidence_trace(details, rubric_rows)
            if evidence_trace:
                profile["evidence_trace"].extend(evidence_trace)

            common_errors = self._extract_common_errors(details, diagnostics)
            if common_errors:
                profile["common_errors"] = common_errors

            score_snapshot = self._safe_dict(details.get("score_snapshot"))
            if score_snapshot:
                profile["score_snapshot"] = score_snapshot

        result = list(profiles.values())
        for profile in result:
            profile["student_quotes"] = self._dedupe_texts(profile.get("student_quotes", []))
            profile["evidence_trace"] = self._dedupe_texts(profile.get("evidence_trace", []))
            profile["rule_count"] = len(profile.get("triggered_rules_history", []))
            profile["rule_ids"] = sorted(
                {
                    self._safe_str(item.get("rule_id")).upper()
                    for item in profile.get("triggered_rules_history", [])
                    if isinstance(item, dict) and self._safe_str(item.get("rule_id"))
                }
            )
            profile["common_errors"] = sorted(
                profile.get("common_errors", []),
                key=lambda item: int(self._safe_dict(item).get("count", 0)),
                reverse=True,
            )[:5]

        result.sort(key=lambda item: (item.get("latest_timestamp", ""), item.get("student_label", "")), reverse=True)
        return result

    def generate_mock_records(self) -> List[Dict[str, Any]]:
        now = datetime.now().replace(second=0, microsecond=0)
        students = [
            {
                "class_name": "创新创业一班",
                "user_id": "学生 12",
                "project_id": "proj_eco_12",
                "project_name": "校园低碳餐饮订阅",
                "stage": "验证期",
                "diagnostics": [
                    {
                        "rule_id": "H8",
                        "severity": "high",
                        "trigger_message": "CAC/LTV 逻辑不自洽，获客成本高于预计终身价值。",
                        "fix_suggestion": "补充分渠道 CAC 与 3 个月留存，重新计算 LTV。",
                    },
                    {
                        "rule_id": "H4",
                        "severity": "medium",
                        "trigger_message": "TAM/SAM/SOM 估算过于乐观，缺少样本依据。",
                        "fix_suggestion": "把市场规模按城市、客群和客单价拆解。",
                    },
                ],
                "evidence": [
                    "学生原文：预计 1 万用户首月转化 40%，但未提供渠道成本。",
                    "学生原文：LTV=120 元，CAC 仅写“约 30 元”，无计算口径。",
                ],
                "common_errors": [
                    {"error": "CAC/LTV 逻辑不自洽", "count": 5, "impact": "商业模式难以持续"},
                    {"error": "TAM/SAM/SOM 估算过于乐观", "count": 4, "impact": "市场判断偏差"},
                ],
                "overall_score": 61,
            },
            {
                "class_name": "创新创业一班",
                "user_id": "学生 30",
                "project_id": "proj_ai_30",
                "project_name": "AI 校园心理预警",
                "stage": "原型期",
                "diagnostics": [
                    {
                        "rule_id": "H1",
                        "severity": "high",
                        "trigger_message": "目标用户与价值主张错位，学生与辅导员痛点混用。",
                        "fix_suggestion": "单独定义首要付费方与使用方的价值闭环。",
                    },
                    {
                        "rule_id": "H6",
                        "severity": "medium",
                        "trigger_message": "竞品对比维度缺失，无法证明方案差异化。",
                        "fix_suggestion": "补齐准确率、成本、部署门槛三项横向对比。",
                    },
                ],
                "evidence": [
                    "学生原文：主要服务对象是学校管理层，同时希望 C 端学生自付费。",
                    "学生原文：核心价值描述在“监测”与“干预服务”之间反复切换。",
                ],
                "common_errors": [
                    {"error": "用户画像与价值主张错位", "count": 6, "impact": "无法形成稳定 PMF"},
                    {"error": "竞品比较缺少关键指标", "count": 3, "impact": "答辩说服力下降"},
                ],
                "overall_score": 58,
            },
            {
                "class_name": "创新创业二班",
                "user_id": "学生 08",
                "project_id": "proj_robot_08",
                "project_name": "仓储巡检机器人",
                "stage": "验证期",
                "diagnostics": [
                    {
                        "rule_id": "H10",
                        "severity": "medium",
                        "trigger_message": "执行计划缺少里程碑，关键节点与资源未绑定。",
                        "fix_suggestion": "以季度为单位给出里程碑、负责人与预算。",
                    },
                    {
                        "rule_id": "H11",
                        "severity": "medium",
                        "trigger_message": "风险预案停留在口号，缺少触发阈值。",
                        "fix_suggestion": "为每个核心风险补充触发条件和备选动作。",
                    },
                ],
                "evidence": [
                    "学生原文：预计 2 个月完成量产测试，但没有供应链备份方案。",
                    "学生原文：风险管理写“持续优化”，未定义具体阈值。",
                ],
                "common_errors": [
                    {"error": "里程碑缺失与资源未绑定", "count": 4, "impact": "执行进度不可控"},
                    {"error": "风险阈值定义缺失", "count": 3, "impact": "应对延迟"},
                ],
                "overall_score": 66,
            },
        ]

        records: List[Dict[str, Any]] = []
        for idx, student in enumerate(students):
            timestamp = (now - timedelta(minutes=idx * 11)).isoformat(timespec="seconds")
            score = int(student["overall_score"])
            diagnostics = list(student["diagnostics"])
            evidence = list(student["evidence"])
            rubric = {
                "problem_definition": {
                    "score": max(score - 10, 0),
                    "max_score": 20,
                    "comment": "问题陈述有价值，但边界与用户群仍需收敛。",
                    "evidence_trace": evidence[:1],
                },
                "business_model": {
                    "score": max(score - 16, 0),
                    "max_score": 20,
                    "comment": "商业路径需要补齐收入口径与成本口径。",
                    "evidence_trace": evidence[:2],
                },
                "execution_feasibility": {
                    "score": max(score - 12, 0),
                    "max_score": 20,
                    "comment": "计划可执行性中等，需补足关键里程碑。",
                    "evidence_trace": evidence[1:],
                },
            }
            records.append(
                {
                    "level": "INFO",
                    "source_path": "mock_agent",
                    "message": "chat_turn",
                    "details": {
                        "timestamp": timestamp,
                        "class_name": student["class_name"],
                        "role": "student",
                        "user_id": student["user_id"],
                        "project_id": student["project_id"],
                        "project_name": student["project_name"],
                        "stage": student["stage"],
                        "user_prompt": "请帮我看下商业计划书逻辑。",
                        "assistant_reply": "已完成多维度诊断，建议优先修复高风险规则。",
                        "project_full_text": (
                            f"{student['project_name']} 当前方案包含目标用户、价值主张、"
                            "渠道策略和财务测算，但关键证据仍不足。"
                        ),
                        "diagnostic_results": diagnostics,
                        "common_errors": student["common_errors"],
                        "evidence_trace": evidence,
                        "score_snapshot": {
                            "overall_score": score,
                            "adjusted_weighted_percentage": round(score * 1.2, 1),
                            "common_error_top5": [
                                "CAC/LTV 逻辑不自洽",
                                "TAM/SAM/SOM 估算过于乐观",
                                "用户画像与价值主张错位",
                                "关键指标缺少证据口径",
                                "竞品比较维度不完整",
                            ],
                            "rubric": rubric,
                        },
                    },
                }
            )
        return records
