"""
硬件控制测试模块
测试硬件函数调用是否成功（无需实际MQTT连接）
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core import HardwareController


def test_parse_command():
    """测试命令解析"""
    print("\n" + "="*60)
    print("测试1: 命令解析功能")
    print("="*60)

    controller = HardwareController()

    test_cases = [
        "设置温度为150度",
        "旋涂实验，转速3000rpm，时长30秒",
        "移动机械臂到位置 x=10, y=20, z=30",
    ]

    for i, cmd in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {cmd}")
        success, tool_calls = controller.agent.parse_complex_command(cmd)

        if success:
            print(f"  ✓ 解析成功")
            for tool_call in tool_calls:
                print(f"    - 工具: {tool_call.get('name')}")
                print(f"      参数: {tool_call.get('params')}")
        else:
            print(f"  ✗ 解析失败")

    return True


def test_tool_validation():
    """测试工具参数验证"""
    print("\n" + "="*60)
    print("测试2: 参数验证功能")
    print("="*60)

    controller = HardwareController()

    test_cases = [
        {
            "name": "set_temperature",
            "params": {"target": 150.0},
            "expected": True,
            "desc": "正常温度设置"
        },
        {
            "name": "do_experiment",
            "params": {"reagent": "PbI2", "spin_speed": 3000, "spin_dur": 30000},
            "expected": True,
            "desc": "正常旋涂参数"
        },
        {
            "name": "do_experiment",
            "params": {"reagent": "PbI2", "spin_speed": 7000, "spin_dur": 30000},
            "expected": False,
            "desc": "转速超限（应失败）"
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {test['desc']}")
        valid, error = controller.agent.validate_tool_params(
            test["name"],
            test["params"]
        )

        if valid == test["expected"]:
            print(f"  ✓ 验证结果符合预期: {'通过' if valid else '拒绝'}")
            if error:
                print(f"    错误信息: {error}")
        else:
            print(f"  ✗ 验证结果不符合预期")
            print(f"    期望: {test['expected']}, 实际: {valid}")

    return True


def test_tool_call_detection():
    """测试工具调用检测（不实际执行MQTT）"""
    print("\n" + "="*60)
    print("测试3: 工具调用检测（模拟）")
    print("="*60)

    controller = HardwareController()

    # 模拟工具调用
    tool_calls = [
        {
            "name": "set_temperature",
            "params": {"target": 150.0}
        },
        {
            "name": "do_experiment",
            "params": {
                "reagent": "PbI2",
                "spin_speed": 3000,
                "spin_acc": 1000,
                "spin_dur": 30000,
                "volume": 10
            }
        }
    ]

    print("\n模拟工具调用序列:")
    for i, tool_call in enumerate(tool_calls, 1):
        print(f"\n  步骤 {i}: {tool_call['name']}")

        # 验证参数
        valid, error = controller.agent.validate_tool_params(
            tool_call["name"],
            tool_call["params"]
        )

        if valid:
            print(f"    ✓ 参数验证通过")
            print(f"    ✓ 函数调用检测成功: {tool_call['name']}")
        else:
            print(f"    ✗ 参数验证失败: {error}")

    return True


def test_confirmation_message():
    """测试确认消息生成"""
    print("\n" + "="*60)
    print("测试4: 确认消息生成")
    print("="*60)

    controller = HardwareController()

    tool_calls = [
        {
            "name": "do_experiment",
            "params": {
                "reagent": "PbI2",
                "spin_speed": 3000,
                "spin_dur": 30000,
                "volume": 10
            }
        }
    ]

    confirmation = controller.ask_for_experiment_confirmation(tool_calls)
    print("\n生成的确认消息:")
    print(confirmation)

    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("硬件控制模块测试套件")
    print("="*60)
    print("\n注意: 本测试不需要实际的MQTT连接")
    print("仅测试函数调用检测和参数验证功能\n")

    try:
        test_parse_command()
        test_tool_validation()
        test_tool_call_detection()
        test_confirmation_message()

        print("\n" + "="*60)
        print("✅ 所有测试完成")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
