from openai import OpenAI
import os

api_key = os.environ.get("GPUSTACK_API_KEY", "")
if not api_key:
    raise SystemExit("GPUSTACK_API_KEY not set")

print(f"Key length: {len(api_key)}")

client = OpenAI(api_key=api_key, base_url="https://maas.bit.edu.cn/v1")
resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "reply OK"}],
    max_tokens=16,
)
print("RESULT:", resp.choices[0].message.content)
