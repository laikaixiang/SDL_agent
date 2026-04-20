"""
快速验证脚本 - 测试修复后的实验设计流程

运行此脚本前，请确保Flask服务器正在运行（python app.py）
"""

import sys
import os
import json
import requests
import time
import io

# 设置stdout为UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

FLASK_URL = "http://127.0.0.1:5000"


def test_experiment_chat():
    """测试/api/experiment_chat接口"""
    print("\n" + "="*80)
    print("测试实验设计对话接口")
    print("="*80)

    url = f"{FLASK_URL}/api/experiment_chat"

    test_message = "设计一个简单的旋涂实验"

    print(f"\n发送请求: {test_message}")
    print(f"URL: {url}")
    print(f"超时设置: 30秒")

    try:
        start_time = time.time()

        response = requests.post(
            url,
            json={
                "session_id": "test_session",
                "message": test_message
            },
            timeout=30
        )

        elapsed = time.time() - start_time

        print(f"\n✅ 请求成功")
        print(f"耗时: {elapsed:.2f}秒")
        print(f"状态码: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"\n响应类型: {data.get('type')}")

            if data.get('type') == 'experiment_design':
                exp_json = data.get('experiment_json', {})
                print(f"✅ 实验设计生成成功")
                print(f"  实验名称: {exp_json.get('experiment_name')}")
                print(f"  步骤数量: {len(exp_json.get('steps', []))}")
                print(f"  是否包含visual_data: {'visual_data' in data}")

                print(f"\n响应摘要:")
                print(f"  - experiment_json字段: {list(exp_json.keys())}")
                print(f"  - visual_data节点数: {len(data.get('visual_data', {}).get('nodes', []))}")
                print(f"  - visual_data边数: {len(data.get('visual_data', {}).get('edges', []))}")

                return True
            elif data.get('type') == 'error':
                print(f"❌ 返回错误: {data.get('reply')}")
                return False
            else:
                print(f"⚠️ 未知响应类型: {data.get('type')}")
                return False
        else:
            print(f"❌ HTTP错误: {response.status_code}")
            print(f"响应: {response.text[:500]}")
            return False

    except requests.Timeout:
        print(f"\n❌ 请求超时（>30秒）")
        print("可能原因:")
        print("  1. LLM响应过慢")
        print("  2. 网络问题")
        print("  3. Flask服务器负载过高")
        return False
    except requests.ConnectionError:
        print(f"\n❌ 连接失败")
        print("请确保Flask服务器正在运行:")
        print("  python app.py")
        return False
    except Exception as e:
        print(f"\n❌ 请求失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_flask_running():
    """检查Flask是否运行"""
    print("\n" + "="*80)
    print("检查Flask服务器状态")
    print("="*80)

    try:
        response = requests.get(f"{FLASK_URL}/", timeout=5)
        print(f"✅ Flask服务器运行中")
        return True
    except:
        print(f"❌ Flask服务器未运行")
        print(f"\n请在另一个终端运行:")
        print(f"  cd D:/PycharmProjects/SDL_agent")
        print(f"  python app.py")
        return False


def main():
    """主函数"""
    print("\n" + "="*80)
    print("实验设计流程修复验证")
    print("="*80)

    # 检查Flask
    if not check_flask_running():
        return

    # 测试接口
    success = test_experiment_chat()

    print("\n" + "="*80)
    print("测试结果")
    print("="*80)

    if success:
        print("✅ 实验设计流程正常工作")
        print("\n修复内容:")
        print("  1. 前端增加30秒超时设置")
        print("  2. 添加超时错误提示")
        print("\n下一步:")
        print("  1. 在浏览器中测试完整流程")
        print("  2. 如仍有问题，检查浏览器控制台错误")
    else:
        print("❌ 测试失败")
        print("\n请检查:")
        print("  1. Flask服务器日志")
        print("  2. API配置（config.txt）")
        print("  3. 网络连接")


if __name__ == "__main__":
    main()
