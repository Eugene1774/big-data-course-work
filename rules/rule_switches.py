from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
RULE_SWITCH_FILE = CONFIG_DIR / "rule_switches.json"
RULE_IDS = [f"H{i}" for i in range(1, 16)]
DEFAULT_RULE_SWITCHES: Dict[str, bool] = {rule_id: True for rule_id in RULE_IDS}


def ensure_rule_switch_file() -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not RULE_SWITCH_FILE.exists():
        RULE_SWITCH_FILE.write_text(
            json.dumps(DEFAULT_RULE_SWITCHES, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return RULE_SWITCH_FILE


def load_rule_switches() -> Dict[str, bool]:
    path = ensure_rule_switch_file()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_RULE_SWITCHES)

    if not isinstance(raw, dict):
        return dict(DEFAULT_RULE_SWITCHES)

    merged = dict(DEFAULT_RULE_SWITCHES)
    for rule_id in RULE_IDS:
        if rule_id in raw:
            merged[rule_id] = bool(raw[rule_id])
    return merged


def save_rule_switches(rule_switches: Dict[str, bool]) -> Dict[str, bool]:
    normalized = dict(DEFAULT_RULE_SWITCHES)
    for rule_id in RULE_IDS:
        if rule_id in rule_switches:
            normalized[rule_id] = bool(rule_switches[rule_id])

    path = ensure_rule_switch_file()
    path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return normalized


def is_rule_enabled(rule_id: str) -> bool:
    switches = load_rule_switches()
    return bool(switches.get(str(rule_id).strip().upper(), True))

