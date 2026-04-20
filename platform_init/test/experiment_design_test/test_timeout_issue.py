"""
测试实验设计对话流程超时问题

可能原因：
1. LLM API调用超时
2. parse_experiment_design内部阻塞
3. JSON解析失败导致长时间等待
4. 网络连接问题
"""

import sys
import os
import time
import json
import io

# 设置stdout为UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from core.field_inference import ExperimentDesignAgent
from core.config import Config


def test_api_connection():
    """测试1: API连接是否正常"""
    print("\n" + "="*60)
    print("测试1: API连接测试")
    print("="*60)

    try:
        config = Config()
        print(f"API URL: {config.API_URL}")
        print(f"Model: {config.MODEL_NAME_TALK}")
        print(f"API Key: {config.API_KEY[:10]}..." if config.API_KEY else "未设置")

        # 简单的API调用测试
        import requests
        start_time = time.time()

        response = requests.post(
            config.API_URL,
            headers={
                "Authorization": f"Bearer {config.API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": config.MODEL_NAME_TALK,
                "messages": [{"role": "user", "content": "测试连接，请回复'OK'"}],
                "max_tokens": 10
            },
            timeout=30
        )

        elapsed = time.time() - start_time
        print(f"✅ API响应成功 (耗时: {elapsed:.2f}秒)")
        print(f"状态码: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"响应内容: {json.dumps(data, ensure_ascii=False, indent=2)[:200]}...")
            return True
        else:
            print(f"❌ API返回错误: {response.text}")
            return False

    except requests.Timeout:
        print(f"❌ API连接超时 (>30秒)")
        return False
    except Exception as e:
        print(f"❌ API连接失败: {e}")
        return False


def test_agent_creation():
    """测试2: Agent创建是否正常"""
    print("\n" + "="*60)
    print("测试2: Agent创建测试")
    print("="*60)

    try:
        start_time = time.time()
        agent = ExperimentDesignAgent()
        elapsed = time.time() - start_time

        print(f"✅ Agent创建成功 (耗时: {elapsed:.2f}秒)")
        print(f"硬件工具数量: {len(agent.hardware_registry)}")
        print(f"软件算法数量: {len(agent.software_registry)}")
        print(f"辅助操作数量: {len(agent.helper_registry)}")

        return True, agent
    except Exception as e:
        print(f"❌ Agent创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_parse_with_timeout(agent, user_message, timeout=60):
    """测试3: 带超时的parse_experiment_design调用"""
    print("\n" + "="*60)
    print("测试3: parse_experiment_design调用测试")
    print("="*60)
    print(f"用户消息: {user_message}")
    print(f"超时设置: {timeout}秒")

    import threading

    result_container = {"success": None, "result": None, "error": None}

    def run_parse():
        try:
            start_time = time.time()
            success, result = agent.parse_experiment_design(user_message)
            elapsed = time.time() - start_time

            result_container["success"] = success
            result_container["result"] = result
            result_container["elapsed"] = elapsed
        except Exception as e:
            result_container["error"] = str(e)
            import traceback
            result_container["traceback"] = traceback.format_exc()

    thread = threading.Thread(target=run_parse)
    thread.daemon = True
    thread.start()

    # 等待结果或超时
    thread.join(timeout=timeout)

    if thread.is_alive():
        print(f"❌ 调用超时 (>{timeout}秒)")
        print("线程仍在运行，可能卡在:")
        print("  - LLM API调用")
        print("  - JSON解析")
        print("  - 内部循环")
        return False

    if result_container.get("error"):
        print(f"❌ 调用出错: {result_container['error']}")
        print(f"堆栈跟踪:\n{result_container.get('traceback', '')}")
        return False

    success = result_container.get("success")
    result = result_container.get("result")
    elapsed = result_container.get("elapsed", 0)

    if success:
        print(f"✅ 调用成功 (耗时: {elapsed:.2f}秒)")
        print(f"实验名称: {result.get('experiment_name', '未命名')}")
        print(f"步骤数量: {len(result.get('steps', []))}")
        print(f"\n生成的JSON (前500字符):")
        print(json.dumps(result, ensure_ascii=False, indent=2)[:500])
        return True
    else:
        print(f"❌ 调用失败 (耗时: {elapsed:.2f}秒)")
        print(f"错误信息: {result}")
        return False


def test_simple_message():
    """测试4: 简单消息测试"""
    print("\n" + "="*60)
    print("测试4: 简单消息测试")
    print("="*60)

    success, agent = test_agent_creation()
    if not success:
        return False

    simple_message = "设计一个简单的旋涂实验"
    return test_parse_with_timeout(agent, simple_message, timeout=60)


def test_complex_message():
    """测试5: 复杂消息测试"""
    print("\n" + "="*60)
    print("测试5: 复杂消息测试")
    print("="*60)

    success, agent = test_agent_creation()
    if not success:
        return False

    complex_message = """
    设计一个钙钛矿薄膜制备实验：
    1. 先设置温度到80度
    2. 进行旋涂，转速3000rpm，时间30秒
    3. 等待5秒
    4. 收集光谱数据
    """
    return test_parse_with_timeout(agent, complex_message, timeout=90)


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("实验设计对话流程超时问题诊断")
    print("="*80)

    results = {}

    # 测试1: API连接
    results["API连接"] = test_api_connection()

    # 测试2: Agent创建
    success, agent = test_agent_creation()
    results["Agent创建"] = success

    if not success:
        print("\n❌ Agent创建失败，无法继续后续测试")
        return

    # 测试3: 简单消息
    results["简单消息"] = test_simple_message()

    # 测试4: 复杂消息
    results["复杂消息"] = test_complex_message()

    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")

    # 诊断建议
    print("\n" + "="*80)
    print("诊断建议")
    print("="*80)

    if not results["API连接"]:
        print("❌ API连接失败 - 检查网络、API_KEY、API_URL配置")
    elif not results["Agent创建"]:
        print("❌ Agent创建失败 - 检查依赖、注册表文件")
    elif not results["简单消息"]:
        print("❌ 简单消息超时 - 可能是LLM调用或JSON解析问题")
        print("   建议: 在parse_experiment_design中添加日志，定位卡住的位置")
    elif not results["复杂消息"]:
        print("⚠️ 复杂消息超时 - 可能是提示词过长或LLM处理时间过长")
        print("   建议: 优化提示词长度，或增加超时时间")
    else:
        print("✅ 所有测试通过 - 问题可能在Flask路由层或前端")
        print("   建议: 检查Flask的超时设置、前端的请求超时配置")


if __name__ == "__main__":
    main()
