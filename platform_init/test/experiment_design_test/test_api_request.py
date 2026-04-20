"""
测试修复后的 /api/experiment_chat 接口

模拟前端请求，测试接口是否正确调用 experiment_agent
"""

import sys
import os
import json
import requests

# 设置标准输出编码为 UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


def test_experiment_chat_api():
    """测试 /api/experiment_chat 接口"""
    print("\n" + "="*80)
    print("测试 /api/experiment_chat 接口")
    print("="*80)

    url = "http://127.0.0.1:5000/api/experiment_chat"

    payload = {
        "session_id": "test_session_api",
        "message": "设计一个旋涂实验，转速3000rpm，加速度1000rpm/s，持续时间30秒，使用PbI2试剂，体积50µl"
    }

    print(f"\n[测试] 请求 URL: {url}")
    print(f"[测试] Session ID: {payload['session_id']}")
    print(f"[测试] 用户消息: {payload['message']}")
    print(f"\n[测试] 发送请求...")

    try:
        response = requests.post(url, json=payload, timeout=60)

        print(f"\n[测试] 响应状态码: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"[测试] 响应类型: {data.get('type')}")

            if data.get('type') == 'experiment_design':
                print(f"[测试] [OK] 实验设计生成成功")

                experiment_json = data.get('experiment_json', {})
                visual_data = data.get('visual_data', {})
                reply = data.get('reply', '')

                print(f"\n[测试] 实验名称: {experiment_json.get('experiment_name', '未命名')}")
                print(f"[测试] 步骤数: {len(experiment_json.get('steps', []))}")
                print(f"[测试] 节点数: {len(visual_data.get('nodes', []))}")
                print(f"[测试] 边数: {len(visual_data.get('edges', []))}")
                print(f"\n[测试] AI 回复:\n{reply}")

                print(f"\n[测试] 完整 JSON:")
                print(json.dumps(experiment_json, ensure_ascii=False, indent=2))

                return True
            elif data.get('type') == 'error':
                print(f"[测试] [FAIL] 返回错误")
                print(f"[测试] 错误信息: {data.get('reply')}")
                return False
            else:
                print(f"[测试] [FAIL] 未知响应类型: {data.get('type')}")
                return False
        else:
            print(f"[测试] [FAIL] HTTP 错误: {response.status_code}")
            print(f"[测试] 响应内容: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"\n[测试] [FAIL] 无法连接到服务器")
        print(f"[测试] 请确保 Flask 应用正在运行: python app.py")
        return False
    except Exception as e:
        print(f"\n[测试] [FAIL] 请求失败")
        print(f"[测试] 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行测试"""
    print("\n" + "="*80)
    print("实验设计对话接口 API 测试")
    print("="*80)
    print("\n[提示] 请确保 Flask 应用正在运行: python app.py")
    print("[提示] 按 Ctrl+C 取消测试\n")

    input("按 Enter 键开始测试...")

    success = test_experiment_chat_api()

    print("\n" + "="*80)
    if success:
        print("[OK] 测试通过")
    else:
        print("[FAIL] 测试失败")
    print("="*80 + "\n")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
