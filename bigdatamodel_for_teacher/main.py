from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import streamlit as st


def ensure_langchain_debug_compat() -> None:
    """Provide `langchain.debug` for mixed langchain/langchain_core installs."""
    try:
        langchain_module = importlib.import_module("langchain")
    except Exception:
        langchain_module = types.ModuleType("langchain")
        sys.modules.setdefault("langchain", langchain_module)

    if not hasattr(langchain_module, "debug"):
        langchain_module.debug = False


ensure_langchain_debug_compat()

st.set_page_config(page_title="多角色导航中枢", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
PAGES_DIR = PROJECT_ROOT / "pages"

ROLE_OPTIONS = ["学生端", "教师端", "教务端"]
ROLE_KEY_MAP = {
    "学生端": "student",
    "教师端": "teacher",
    "教务端": "admin",
}


def init_session_state() -> None:
    defaults = {
        "role": "student",
        "user_id": "",
        "project_id": None,
        "chat_history": [],
        "agent_next_step": "",
        "agent_active_agent": "",
        "agent_route_target": "",
        "agent_path_evidence": [],
        "logged_in": False,
        "user_name": "",
        "team_role": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_role_label(role_key: str) -> str:
    for label, key in ROLE_KEY_MAP.items():
        if key == role_key:
            return label
    return "学生端"


def build_navigation(role_key: str):
    student_pages = [
        st.Page(
            str(PAGES_DIR / "student_portal.py"),
            title="项目陪跑教练",
            icon=":material/school:",
            default=True,
        )
    ]
    teacher_pages = [
        st.Page(
            str(PAGES_DIR / "teacher_portal.py"),
            title="班级看板",
            icon=":material/dashboard:",
            default=True,
        )
    ]
    admin_pages = [
        st.Page(
            str(PAGES_DIR / "admin_portal.py"),
            title="全局预警",
            icon=":material/admin_panel_settings:",
            default=True,
        )
    ]

    page_groups = {
        "student": {"学生组": student_pages},
        "teacher": {"教师组": teacher_pages},
        "admin": {"教务组": admin_pages},
    }

    return st.navigation(page_groups[role_key], position="sidebar")


def login_page():
    """拦截式登录页：在用户未登录时强制显示"""
    st.title("🚀 大创智能体 - 欢迎登录")
    st.markdown("请填写您的基本信息以进入系统。")

    with st.form("login_form"):
        name = st.text_input("姓名", placeholder="例如：张三")

        role = st.selectbox(
            "选择系统角色",
            options=["student", "teacher", "admin"],
            format_func=lambda x: {"student": "🎓 学生", "teacher": "👨‍🏫 导师", "admin": "🏢 教务"}[x],
        )

        team_role = st.selectbox("团队职务 (仅学生需要)", ["项目负责人", "核心队员", "普通成员", "无"])

        submitted = st.form_submit_button("进入系统", type="primary")

        if submitted:
            if not name.strip():
                st.error("⚠️ 请输入您的姓名！")
            else:
                st.session_state.user_name = name
                st.session_state.role = role
                st.session_state.team_role = team_role if role == "student" else ""
                st.session_state.logged_in = True
                st.session_state.user_id = f"uid_{name}"
                st.rerun()


def render_sidebar_header():
    """在侧边栏顶端显示用户身份信息及快速切换"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("👤 当前用户")

    user_name = st.session_state.get("user_name", "未登录")
    user_role = st.session_state.get("role", "student")
    team_role = st.session_state.get("team_role", "")

    role_labels = {"student": "学生", "teacher": "导师", "admin": "教务"}
    role_display = role_labels.get(user_role, "未知")

    display_title = f"**{user_name}** - {role_display}"
    if team_role and user_role == "student":
        display_title += f" ({team_role})"

    st.sidebar.markdown(display_title)

    st.sidebar.markdown("---")

    st.sidebar.markdown("### 🔄 视角切换 (Demo)")
    new_role = st.sidebar.radio(
        "快速跳转至：",
        options=["student", "teacher", "admin"],
        index=["student", "teacher", "admin"].index(user_role),
        format_func=lambda x: {"student": "🎓 学生工作台", "teacher": "👨‍🏫 导师看板", "admin": "🏢 教务大盘"}[x],
        key="role_switcher",
    )

    if new_role != user_role:
        st.session_state.role = new_role
        st.rerun()


def main() -> None:
    init_session_state()

    if not st.session_state.get("logged_in", False):
        login_page()
        st.stop()

    render_sidebar_header()

    nav = build_navigation(st.session_state.role)
    nav.run()

    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 退出登录 / 重置"):
        st.session_state.clear()
        st.rerun()


if __name__ == "__main__":
    main()
