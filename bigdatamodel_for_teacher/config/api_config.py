# SiliconFlow API 配置
# API Key 从环境变量 SILICONFLOW_API_KEY 读取（可写入项目根目录 .env 文件）
# 获取 API key: https://cloud.siliconflow.com/

import os

API_KEY = os.getenv("SILICONFLOW_API_KEY", "")

# SiliconFlow API 基础 URL
BASE_URL = "https://api.siliconflow.com/v1"

# 使用的模型
DEFAULT_MODEL = "Qwen/Qwen3-8B"
