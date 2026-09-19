from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROJECTS_DIR = DATA_DIR / "projects"

PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


def get_project_path(project_id: str) -> Path:
    """获取项目数据文件路径"""
    return PROJECTS_DIR / f"{project_id}.json"


def save_project_data(project_id: str, data: Dict[str, Any]) -> None:
    """保存项目数据"""
    path = get_project_path(project_id)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_project_data(project_id: str) -> Optional[Dict[str, Any]]:
    """加载项目数据"""
    path = get_project_path(project_id)
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def create_project(project_id: str, project_name: str) -> Dict[str, Any]:
    """创建新项目"""
    project_data = {
        "project_id": project_id,
        "project_name": project_name,
        "scores": [0, 0, 0, 0, 0],
        "diagnostic_history": [],
        "chat_history": [],
        "created_at": "2026-04-20",
        "members": []
    }
    save_project_data(project_id, project_data)
    return project_data


def update_project_score(project_id: str, scores: List[float]) -> bool:
    """更新项目分数"""
    project_data = load_project_data(project_id)
    if not project_data:
        return False
    project_data["scores"] = scores
    save_project_data(project_id, project_data)
    return True


def add_project_member(project_id: str, user_id: str, user_name: str, team_role: str) -> bool:
    """添加项目成员"""
    project_data = load_project_data(project_id)
    if not project_data:
        return False
    
    members = project_data.get("members", [])
    if not any(m.get("user_id") == user_id for m in members):
        members.append({
            "user_id": user_id,
            "user_name": user_name,
            "team_role": team_role
        })
        project_data["members"] = members
        save_project_data(project_id, project_data)
    return True


def list_projects() -> List[Dict[str, Any]]:
    """列出所有项目"""
    projects = []
    for path in PROJECTS_DIR.glob("*.json"):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                project_data = json.load(f)
                projects.append(project_data)
        except Exception:
            pass
    return projects


def get_project_members(project_id: str) -> List[Dict[str, Any]]:
    """获取项目成员"""
    project_data = load_project_data(project_id)
    if not project_data:
        return []
    return project_data.get("members", [])
