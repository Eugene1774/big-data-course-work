from openai import OpenAI
from config.api_config import API_KEY, BASE_URL, DEFAULT_MODEL

# 初始化 OpenAI 客户端（使用 SiliconFlow 配置）
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

try:
    # 测试 API 调用
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": "你是一位友好的助手。"},
            {"role": "user", "content": "你好，测试一下 SiliconFlow API 是否正常工作。"}
        ],
        temperature=0.7,
        max_tokens=100
    )
    
    print("SiliconFlow API 调用成功！")
    print(f"使用的模型: {DEFAULT_MODEL}")
    print("响应内容:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"SiliconFlow API 调用失败: {e}")
    print("请检查 config/api_config.py 中的 API key 是否正确。")
    print("获取 API key: https://cloud.siliconflow.com/")
