"""
测试Qwen模型是否支持OpenAI Function Calling
"""
import requests
import json

API_KEY = "sk-zskagakurneevlklkhhzbaxunehikfyeinnjvyizyfstvtci"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"

# 测试1: 简单对话（不使用工具）
print("=" * 60)
print("测试1: 简单对话（不使用工具）")
print("=" * 60)

payload1 = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "你好，请简单介绍一下你自己"}
    ]
}

response1 = requests.post(
    API_URL,
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json=payload1,
    timeout=30
)

print(f"状态码: {response1.status_code}")
if response1.status_code == 200:
    result1 = response1.json()
    print(f"响应: {json.dumps(result1, indent=2, ensure_ascii=False)}")
else:
    print(f"错误: {response1.text}")

# 测试2: 使用Function Calling
print("\n" + "=" * 60)
print("测试2: 使用Function Calling")
print("=" * 60)

payload2 = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "帮我读取test.pdf文件的第1页"}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "read_pdf",
                "description": "读取PDF文件的指定页面",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "PDF文件路径"
                        },
                        "page_number": {
                            "type": "integer",
                            "description": "页码（从1开始）"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        }
    ],
    "tool_choice": "auto"
}

response2 = requests.post(
    API_URL,
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json=payload2,
    timeout=30
)

print(f"状态码: {response2.status_code}")
if response2.status_code == 200:
    result2 = response2.json()
    print(f"响应: {json.dumps(result2, indent=2, ensure_ascii=False)}")

    # 检查响应格式
    message = result2.get("choices", [{}])[0].get("message", {})
    has_content = message.get("content") is not None and message.get("content") != ""
    has_tool_calls = message.get("tool_calls") is not None and len(message.get("tool_calls", [])) > 0

    print("\n" + "=" * 60)
    print("响应分析:")
    print(f"  - 包含文本内容 (content): {has_content}")
    print(f"  - 包含工具调用 (tool_calls): {has_tool_calls}")

    if not has_content and not has_tool_calls:
        print("  ⚠️  警告: 响应既没有文本也没有工具调用！")
        print("  这就是导致 'Please return text or call a tool' 错误的原因")
    elif has_tool_calls:
        print("  ✅ 模型支持Function Calling")
        print(f"  调用的工具: {message['tool_calls'][0]['function']['name']}")
        print(f"  工具参数: {message['tool_calls'][0]['function']['arguments']}")
    else:
        print("  ℹ️  模型返回了文本但没有调用工具")
else:
    print(f"错误: {response2.text}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
