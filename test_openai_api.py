from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI()

try:
    # 测试 API 调用
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是一位友好的助手。"},
            {"role": "user", "content": "你好，测试一下 API 是否正常工作。"}
        ],
        temperature=0.7,
        max_tokens=100
    )
    
    print("API 调用成功！")
    print("响应内容:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"API 调用失败: {e}")
    print("请检查是否设置了有效的 OpenAI API key。")
    print("你可以通过以下方式设置 API key:")
    print("1. 设置环境变量: set OPENAI_API_KEY=your_api_key")
    print('2. 在代码中直接设置: client = OpenAI(api_key="your_api_key")')
