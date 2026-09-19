from __future__ import annotations

import streamlit as st

from core.db_init import init_interaction_log_db, log_interaction_to_db


def _append_rbac_log(required_role: str, current_role: str, page_name: str) -> None:
    try:
        init_interaction_log_db()
        log_interaction_to_db(
            student_id=str(current_role or "anonymous").strip(),
            project_id=str(page_name or "").strip(),
            agent_role="RBAC",
            user_input=f"required_role={required_role}",
            agent_response="403_forbidden",
            triggered_rules=[],
            next_step="",
            session_id="rbac_guard",
        )
    except Exception:
        return


def enforce_rbac(required_role: str, page_name: str = "") -> None:
    """Hard RBAC gate for Streamlit pages. Unauthorized access is blocked with 403-style stop."""
    current_role = str(st.session_state.get("role", "")).strip()
    if current_role == required_role:
        return

    _append_rbac_log(required_role, current_role, page_name or required_role)
    st.error("403 Forbidden: You do not have permission to access this page.")
    st.stop()
