"""
测试Flask路由层的experiment_chat接口

模拟前端请求，检查响应是否正常
"""

import sys
import os
import time
import json
import requests

# 测试配置
FLASK_URL = "http://127.0.0.1:5000"
TEST_TIMEOUT = 60  # 前端请求超时时间


def test_experiment_chat_api():
    """测试/api/experiment_chat接口"""
    print("\n" + "="*60)
    print("测试Flask /api/experiment_chat接口")
    print("="*60)

    url = f"{FLASK_URL}/api/experiment_chat"

    test_cases = [
        {
            "name": "简单实验设计",
            "data": {
                "session_id": "test_session_1",
                "message": "设计一个简单的旋涂实验"
            }
        },
        {
            "name": "复杂实验设计",
            "data": {
                "session_id": "test_session_2",
                "message": "设计一个钙钛矿薄膜制备实验：先设置温度到80度，然后旋涂3000rpm持续30秒"
            }
        }
    ]

    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"测试用例: {test_case['name']}")
        print(f"{'='*60}")
        print(f"请求数据: {json.dumps(test_case['data'], ensure_ascii=False)}")

        try:
            start_time = time.time()

            response = requests.post(
                url,
                json=test_case['data'],
                timeout=TEST_TIMEOUT
            )

            elapsed = time.time() - start_time

            print(f"\n响应状态码: {response.status_code}")
            print(f"响应耗时: {elapsed:.2f}秒")

            if response.status_code == 200:
                data = response.json()
                print(f"响应类型: {data.get('type', 'unknown')}")

                if data.get('type') == 'experiment_design':
                    print(f"✅ 成功生成实验设计")
                    exp_json = data.get('experiment_json', {})
                    print(f"  实验名称: {exp_json.get('experiment_name', '未命名')}")
                    print(f"  步骤数量: {len(exp_json.get('steps', []))}")
                    print(f"  是否包含visual_data: {'visual_data' in data}")
                    print(f"\n响应JSON (前500字符):")
                    print(json.dumps(data, ensure_ascii=False, indent=2)[:500])
                elif data.get('type') == 'error':
                    print(f"❌ 返回错误: {data.get('reply', '未知错误')}")
                else:
                    print(f"⚠️ 未知响应类型: {data.get('type')}")
                    print(f"响应内容: {json.dumps(data, ensure_ascii=False, indent=2)[:500]}")
            else:
                print(f"❌ HTTP错误: {response.status_code}")
                print(f"响应内容: {response.text[:500]}")

        except requests.Timeout:
            print(f"❌ 请求超时 (>{TEST_TIMEOUT}秒)")
            print("可能原因:")
            print("  1. Flask路由内部阻塞")
            print("  2. parse_experiment_design耗时过长")
            print("  3. 响应未正确返回")
        except requests.ConnectionError:
            print(f"❌ 连接失败 - Flask服务器未运行")
            print(f"请先启动: python app.py")
            return False
        except Exception as e:
            print(f"❌ 请求失败: {e}")
            import traceback
            traceback.print_exc()

    return True


def check_flask_running():
    """检查Flask服务器是否运行"""
    print("\n" + "="*60)
    print("检查Flask服务器状态")
    print("="*60)

    try:
        response = requests.get(f"{FLASK_URL}/", timeout=5)
        print(f"✅ Flask服务器运行中 (状态码: {response.status_code})")
        return True
    except requests.ConnectionError:
        print(f"❌ Flask服务器未运行")
        print(f"请在另一个终端运行: python app.py")
        return False
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False


def main():
    """主测试流程"""
    print("\n" + "="*80)
    print("Flask路由层测试 - experiment_chat接口")
    print("="*80)

    # 检查Flask是否运行
    if not check_flask_running():
        print("\n请先启动Flask服务器，然后重新运行此测试")
        return

    # 测试API接口
    test_experiment_chat_api()

    print("\n" + "="*80)
    print("测试完成")
    print("="*80)


if __name__ == "__main__":
    main()
