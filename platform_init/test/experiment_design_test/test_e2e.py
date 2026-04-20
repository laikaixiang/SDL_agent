"""
端到端测试：完整测试实验设计对话流程

测试流程：
1. 发送 "实验设计：xxx" 到 /api/chat
2. 检查返回的 experiment_design_mode 响应
3. 使用返回的 command 调用 /api/experiment_chat
4. 检查最终的实验设计 JSON
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


def test_step1_chat_api():
    """步骤1：测试 /api/chat 识别实验设计请求"""
    print("\n" + "="*80)
    print("步骤1：测试 /api/chat 识别实验设计请求")
    print("="*80)

    url = "http://127.0.0.1:5000/api/chat"
    payload = {
        "message": "实验设计：设计一个旋涂实验，转速3000rpm，加速度1000rpm/s，持续时间30秒，使用PbI2试剂，体积50µl"
    }

    print(f"\n[测试] 请求 URL: {url}")
    print(f"[测试] 用户消息: {payload['message']}")
    print(f"\n[测试] 发送请求...")

    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"\n[测试] 响应状态码: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"[测试] 响应类型: {data.get('type')}")
            print(f"[测试] 响应内容:")
            print(json.dumps(data, ensure_ascii=False, indent=2))

            if data.get('type') == 'experiment_design_mode':
                print(f"\n[测试] [OK] 正确识别为实验设计模式")
                print(f"[测试] 提取的命令: {data.get('command')}")
                return True, data.get('command')
            else:
                print(f"\n[测试] [FAIL] 未识别为实验设计模式")
                print(f"[测试] 实际类型: {data.get('type')}")
                return False, None
        else:
            print(f"[测试] [FAIL] HTTP 错误: {response.status_code}")
            print(f"[测试] 响应内容: {response.text}")
            return False, None

    except Exception as e:
        print(f"\n[测试] [FAIL] 请求失败")
        print(f"[测试] 错误: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_step2_experiment_chat_api(command):
    """步骤2：测试 /api/experiment_chat 生成实验设计"""
    print("\n" + "="*80)
    print("步骤2：测试 /api/experiment_chat 生成实验设计")
    print("="*80)

    url = "http://127.0.0.1:5000/api/experiment_chat"
    session_id = 'test_e2e_' + str(int(os.times().elapsed * 1000))
    payload = {
        "session_id": session_id,
        "message": command
    }

    print(f"\n[测试] 请求 URL: {url}")
    print(f"[测试] Session ID: {session_id}")
    print(f"[测试] 命令: {command}")
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

                print(f"\n[测试] 完整 experiment_json:")
                print(json.dumps(experiment_json, ensure_ascii=False, indent=2))

                print(f"\n[测试] 完整 visual_data:")
                print(json.dumps(visual_data, ensure_ascii=False, indent=2))

                return True
            elif data.get('type') == 'error':
                print(f"[测试] [FAIL] 返回错误")
                print(f"[测试] 错误信息: {data.get('reply')}")
                return False
            else:
                print(f"[测试] [FAIL] 未知响应类型: {data.get('type')}")
                print(f"[测试] 完整响应:")
                print(json.dumps(data, ensure_ascii=False, indent=2))
                return False
        else:
            print(f"[测试] [FAIL] HTTP 错误: {response.status_code}")
            print(f"[测试] 响应内容: {response.text}")
            return False

    except Exception as e:
        print(f"\n[测试] [FAIL] 请求失败")
        print(f"[测试] 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行端到端测试"""
    print("\n" + "="*80)
    print("实验设计对话 - 端到端测试")
    print("="*80)
    print("\n[提示] 请确保 Flask 应用正在运行: python app.py")
    print("[提示] 按 Ctrl+C 取消测试\n")

    input("按 Enter 键开始测试...")

    # 步骤1：测试 /api/chat
    step1_success, command = test_step1_chat_api()

    if not step1_success:
        print("\n" + "="*80)
        print("[FAIL] 步骤1失败，无法继续")
        print("="*80 + "\n")
        return False

    # 步骤2：测试 /api/experiment_chat
    step2_success = test_step2_experiment_chat_api(command)

    # 总结
    print("\n" + "="*80)
    print("测试结果总结")
    print("="*80)
    print(f"步骤1 (/api/chat): {'[PASS]' if step1_success else '[FAIL]'}")
    print(f"步骤2 (/api/experiment_chat): {'[PASS]' if step2_success else '[FAIL]'}")

    all_success = step1_success and step2_success

    print("\n" + "="*80)
    if all_success:
        print("[OK] 所有测试通过 - 端到端流程正常")
    else:
        print("[FAIL] 部分测试失败 - 请检查失败的步骤")
    print("="*80 + "\n")

    return all_success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[测试] 用户取消测试")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("\n\n[测试] [FAIL] 无法连接到服务器")
        print("[测试] 请确保 Flask 应用正在运行: python app.py")
        sys.exit(1)
