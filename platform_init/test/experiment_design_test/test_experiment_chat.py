"""
测试实验设计对话流程
"""
import asyncio
from core.experiment_agent import ExperimentDesignAgent
from core.experiment_manager import ExperimentManager
from core.software_manager import SoftwareManager

def test_experiment_chat():
    print("="*60)
    print("开始测试实验设计对话流程")
    print("="*60)

    # 初始化组件
    print("\n[1] 初始化 SoftwareManager...")
    software_manager = SoftwareManager()
    print("✓ SoftwareManager 初始化成功")

    print("\n[2] 初始化 ExperimentDesignAgent...")
    experiment_agent = ExperimentDesignAgent()
    print("✓ ExperimentDesignAgent 初始化成功")

    print("\n[3] 初始化 ExperimentManager...")
    experiment_manager = ExperimentManager(software_manager=software_manager)
    print("✓ ExperimentManager 初始化成功")

    # 模拟请求
    session_id = "test_session"
    user_message = "制备钙钛矿薄膜，包括旋涂、退火、光谱采集"

    print(f"\n[4] 准备调用 experiment_agent.run()...")
    print(f"    Session: {session_id}")
    print(f"    Message: {user_message}")

    events = []

    async def collect_event(event):
        print(f"    收到事件: {event.get('type')}")
        events.append(event)

    # 运行异步 agent
    print("\n[5] 运行异步 agent...")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result_text = loop.run_until_complete(
            experiment_agent.run(session_id, user_message, collect_event)
        )
        loop.close()
        print(f"✓ Agent 执行完成")
        print(f"    返回文本: {result_text[:100]}...")
    except Exception as e:
        print(f"✗ Agent 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 检查事件
    print(f"\n[6] 检查收集的事件 (共 {len(events)} 个)...")
    experiment_event = None
    for event in events:
        if event.get('type') == 'experiment_design_generated':
            experiment_event = event
            print(f"✓ 找到实验设计生成事件")
            break

    if not experiment_event:
        print("✗ 未找到实验设计生成事件")
        print(f"   所有事件: {events}")
        return

    # 验证实验方案
    print("\n[7] 验证实验方案...")
    experiment_json = experiment_event.get('experiment_json', {})
    is_valid, error_msg = experiment_manager.validate_plan(experiment_json)
    if not is_valid:
        print(f"⚠ 验证警告: {error_msg}")
    else:
        print(f"✓ 验证通过")

    print(f"\n[8] 实验名称: {experiment_json.get('experiment_name', '未命名')}")
    print(f"    步骤数量: {len(experiment_json.get('steps', []))}")

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)

if __name__ == "__main__":
    test_experiment_chat()
