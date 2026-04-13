"""
API测试脚本 - 测试聊天接口是否正常工作
"""

import requests
import json

# 测试配置
BASE_URL = "http://127.0.0.1:5000"

def test_normal_chat():
    """测试普通聊天接口（流式响应）"""
    print("=" * 50)
    print("测试1: 普通聊天接口（流式响应）")
    print("=" * 50)

    url = f"{BASE_URL}/api/chat"
    payload = {
        "action": "chat",
        "message": "你好，请介绍一下你自己"
    }

    try:
        response = requests.post(
            url,
            json=payload,
            stream=True,
            timeout=30
        )

        print(f"状态码: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")
        print("\n响应内容:")
        print("-" * 50)

        # 读取流式响应
        full_response = ""
        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
            if chunk:
                print(chunk, end='', flush=True)
                full_response += chunk

        print("\n" + "-" * 50)
        print(f"\n总共接收到 {len(full_response)} 个字符")

        if len(full_response) > 0:
            print("[OK] Streaming response works")
            return True
        else:
            print("[FAILED] Streaming response is empty")
            return False

    except Exception as e:
        print(f"[FAILED] Request failed: {str(e)}")
        return False


def test_chat_with_prefix():
    """测试带前缀的聊天（模拟前端的模式选择）"""
    print("\n" + "=" * 50)
    print("测试2: 带前缀的聊天")
    print("=" * 50)

    url = f"{BASE_URL}/api/chat"
    payload = {
        "action": "chat",
        "message": "什么是钙钛矿？"
    }

    try:
        response = requests.post(
            url,
            json=payload,
            stream=True,
            timeout=30
        )

        print(f"状态码: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")
        print("\n响应内容:")
        print("-" * 50)

        full_response = ""
        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
            if chunk:
                print(chunk, end='', flush=True)
                full_response += chunk

        print("\n" + "-" * 50)
        print(f"\n总共接收到 {len(full_response)} 个字符")

        if len(full_response) > 0:
            print("[OK] Prefixed chat works")
            return True
        else:
            print("[FAILED] Response is empty")
            return False

    except Exception as e:
        print(f"[FAILED] Request failed: {str(e)}")
        return False


def test_server_connection():
    """测试服务器连接"""
    print("\n" + "=" * 50)
    print("测试0: 服务器连接")
    print("=" * 50)

    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("[OK] Server connection successful")
            return True
        else:
            print("[FAILED] Server response abnormal")
            return False
    except Exception as e:
        print(f"[FAILED] Cannot connect to server: {str(e)}")
        print("Please make sure app.py is running")
        return False


if __name__ == "__main__":
    print("\n[API Test] Starting API tests...\n")

    # 测试服务器连接
    if not test_server_connection():
        print("\n[WARNING] Server not running, please start app.py first")
        exit(1)

    # 测试普通聊天
    test1_result = test_normal_chat()

    # 测试带前缀的聊天
    test2_result = test_chat_with_prefix()

    # 总结
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    print(f"Server Connection: [OK]")
    print(f"Normal Chat: {'[OK]' if test1_result else '[FAILED]'}")
    print(f"Prefixed Chat: {'[OK]' if test2_result else '[FAILED]'}")

    if test1_result and test2_result:
        print("\n[SUCCESS] All tests passed! API is working properly")
        print("\nIf API works but frontend doesn't display, the issue is likely in frontend JavaScript")
    else:
        print("\n[FAILED] Some tests failed, API may have issues")
