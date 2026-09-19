# LLM API 配置
# 主接入点：GPUSTACK (deepseek-v4-flash)，凭据优先级：环境变量 > st.secrets > .streamlit/secrets.toml
# 备用接入点：SiliconFlow（当 GPUSTACK 凭据缺失时自动回退）

import os
import tomllib
from pathlib import Path

# SiliconFlow API 配置（备用回退）：环境变量 > .streamlit/secrets.toml

# SiliconFlow API 基础 URL
BASE_URL = "https://api.siliconflow.com/v1"

# 使用的模型
DEFAULT_MODEL = "Qwen/Qwen3-8B"

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SECRETS_PATH = _PROJECT_ROOT / ".streamlit" / "secrets.toml"


def _read_secrets_toml() -> dict:
    try:
        with open(_SECRETS_PATH, "rb") as handle:
            return tomllib.load(handle)
    except Exception:
        return {}


API_KEY = (
    os.getenv("SILICONFLOW_API_KEY", "").strip()
    or str(_read_secrets_toml().get("SILICONFLOW_API_KEY") or "").strip()
)


def _resolve_gpustack() -> dict | None:
    api_key = str(os.getenv("GPUSTACK_API_KEY") or "").strip()
    base_url = str(os.getenv("GPUSTACK_BASE_URL") or "").strip()
    model = str(os.getenv("GPUSTACK_MODEL") or "").strip()

    if not (api_key and base_url and model):
        secrets: dict = {}
        try:
            import streamlit as st

            secrets = dict(st.secrets)
        except Exception:
            secrets = _read_secrets_toml()

        api_key = api_key or str(secrets.get("GPUSTACK_API_KEY") or "").strip()
        base_url = base_url or str(secrets.get("GPUSTACK_BASE_URL") or "").strip()
        model = model or str(secrets.get("GPUSTACK_MODEL") or "").strip()

    if api_key and base_url and model:
        return {"api_key": api_key, "base_url": base_url, "model": model}
    return None


def get_llm_config() -> dict:
    """返回当前生效的 LLM 接入配置 {api_key, base_url, model}。

    GPUSTACK 优先；凭据不完整时回退 SiliconFlow。
    """
    gpustack = _resolve_gpustack()
    if gpustack:
        return gpustack
    return {"api_key": API_KEY, "base_url": BASE_URL, "model": DEFAULT_MODEL}
