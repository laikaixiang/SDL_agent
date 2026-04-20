"""
实验设计对话接口测试

测试 /api/experiment_chat 接口是否正确调用 ExperimentDesignAgent
"""

import sys
import os
import json
import asyncio

# 设置标准输出编码为 UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from core.experiment_agent import ExperimentDesignAgent


async def test_experiment_agent_direct():
    """测试1：直接调用 ExperimentDesignAgent"""
    print("\n" + "="*60)
    print("测试1：直接调用 ExperimentDesignAgent")
    print("="*60)

    agent = ExperimentDesignAgent()
    session_id = "test_session_001"
    user_message = "设计一个旋涂实验，转速3000rpm，加速度1000rpm/s，持续时间30秒，使用PbI2试剂，体积50µl"

    # 模拟 send_event 回调
    events = []
    async def mock_send_event(event):
        events.append(event)
        print(f"[Event] {event.get('type')}: {event}")

    print(f"\n[测试] Session ID: {session_id}")
    print(f"[测试] 用户消息: {user_message}")
    print(f"[测试] 开始调用 agent.run()...\n")

    try:
        result = await agent.run(session_id, user_message, mock_send_event)
        print(f"\n[测试] [OK] 调用成功")
        print(f"[测试] 返回结果: {result}")
        print(f"[测试] 捕获事件数: {len(events)}")
        for i, event in enumerate(events):
            print(f"[测试] 事件 {i+1}: {event.get('type')}")
        return True
    except Exception as e:
        print(f"\n[测试] [FAIL] 调用失败")
        print(f"[测试] 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_field_inference_agent():
    """测试2：测试 field_inference.ExperimentDesignAgent（当前 app.py 使用的版本）"""
    print("\n" + "="*60)
    print("测试2：测试 field_inference.ExperimentDesignAgent")
    print("="*60)

    from core.field_inference import ExperimentDesignAgent as FieldAgent

    agent = FieldAgent()
    user_message = "设计一个旋涂实验，转速3000rpm，加速度1000rpm/s，持续时间30秒，使用PbI2试剂，体积50µl"

    print(f"\n[测试] 用户消息: {user_message}")
    print(f"[测试] 开始调用 agent.parse_experiment_design()...\n")

    try:
        success, result = agent.parse_experiment_design(user_message)
        print(f"\n[测试] 调用完成")
        print(f"[测试] 成功: {success}")
        if success:
            print(f"[测试] [OK] 生成成功")
            print(f"[测试] 实验名称: {result.get('experiment_name', '未命名')}")
            print(f"[测试] 步骤数: {len(result.get('steps', []))}")
            print(f"\n[测试] 完整JSON:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"[测试] [FAIL] 生成失败")
            print(f"[测试] 错误: {result}")
        return success
    except Exception as e:
        print(f"\n[测试] [FAIL] 调用失败")
        print(f"[测试] 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_methods():
    """测试3：检查 ExperimentDesignAgent 的方法"""
    print("\n" + "="*60)
    print("测试3：检查 ExperimentDesignAgent 的方法")
    print("="*60)

    from core.experiment_agent import ExperimentDesignAgent

    agent = ExperimentDesignAgent()

    methods = [
        'run',
        'set_pdf_path',
        'submit_response',
        'wait_for_response',
        'clear_session',
        'get_active_sessions',
        'create_response_queue'
    ]

    print(f"\n[测试] 检查方法是否存在:")
    all_exist = True
    for method in methods:
        exists = hasattr(agent, method)
        status = "[OK]" if exists else "[FAIL]"
        print(f"  {status} {method}: {exists}")
        if not exists:
            all_exist = False

    return all_exist


def test_app_import():
    """测试4：检查 app.py 中的 experiment_agent"""
    print("\n" + "="*60)
    print("测试4：检查 app.py 中的 experiment_agent")
    print("="*60)

    try:
        import app

        print(f"\n[测试] app.py 导入成功")
        print(f"[测试] experiment_agent 类型: {type(app.experiment_agent).__name__}")
        print(f"[测试] experiment_agent 模块: {type(app.experiment_agent).__module__}")

        # 检查是否是交互式版本
        has_run = hasattr(app.experiment_agent, 'run')
        has_set_pdf = hasattr(app.experiment_agent, 'set_pdf_path')
        has_submit = hasattr(app.experiment_agent, 'submit_response')

        print(f"\n[测试] 方法检查:")
        print(f"  [OK] run: {has_run}" if has_run else f"  [FAIL] run: {has_run}")
        print(f"  [OK] set_pdf_path: {has_set_pdf}" if has_set_pdf else f"  [FAIL] set_pdf_path: {has_set_pdf}")
        print(f"  [OK] submit_response: {has_submit}" if has_submit else f"  [FAIL] submit_response: {has_submit}")

        is_interactive = has_run and has_set_pdf and has_submit

        if is_interactive:
            print(f"\n[测试] [OK] app.experiment_agent 是交互式版本")
        else:
            print(f"\n[测试] [FAIL] app.experiment_agent 不是交互式版本")

        return is_interactive
    except Exception as e:
        print(f"\n[测试] ❌ 导入失败")
        print(f"[测试] 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("实验设计对话接口测试套件")
    print("="*80)

    results = {}

    # 测试3：方法检查（同步）
    results['methods'] = test_agent_methods()

    # 测试4：app.py 导入检查（同步）
    results['app_import'] = test_app_import()

    # 测试2：field_inference 版本（同步）
    results['field_inference'] = test_field_inference_agent()

    # 测试1：交互式版本（异步）
    results['interactive'] = asyncio.run(test_experiment_agent_direct())

    # 总结
    print("\n" + "="*80)
    print("测试结果总结")
    print("="*80)
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} - {test_name}")

    all_passed = all(results.values())
    print("\n" + "="*80)
    if all_passed:
        print("[OK] 所有测试通过")
    else:
        print("[FAIL] 部分测试失败")
    print("="*80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
