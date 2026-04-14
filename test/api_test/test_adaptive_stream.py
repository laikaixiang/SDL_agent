"""
自适应流式处理器测试脚本
测试新的自适应流式响应功能

使用方法:
    conda activate SDL_agent
    python test/api_test/test_adaptive_stream.py
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.config import Config
from core.llm_client import LLMClient
from core.adaptive_stream import AdaptiveStreamHandler


def test_adaptive_stream():
    """测试自适应流式处理器"""

    print("=" * 60)
    print("  自适应流式处理器测试")
    print("=" * 60)

    # 初始化
    config = Config()
    llm_client = LLMClient()
    handler = AdaptiveStreamHandler(config, llm_client)

    print("\n[测试1] 检测流式支持")
    print("-" * 60)
    supports = handler.supports_streaming()
    print(f"结果: {'支持流式响应' if supports else '不支持流式响应'}")

    print("\n[测试2] 获取状态信息")
    print("-" * 60)
    status = handler.get_status()
    print(f"流式支持: {status['streaming_support']}")
    print(f"上次检测时间: {status['last_check_time']}")
    print(f"检测间隔: {status['check_interval']}秒")
    print(f"距离下次检测: {status['time_until_recheck']:.0f}秒")

    print("\n[测试3] 测试非流式响应")
    print("-" * 60)
    print("发送消息: 用一句话介绍你自己")
    response = handler.generate_non_streaming_response("用一句话介绍你自己")
    print(f"响应: {response[:100]}...")

    if supports:
        print("\n[测试4] 测试流式响应")
        print("-" * 60)
        print("发送消息: 数到3")
        print("响应: ", end="", flush=True)
        for chunk in handler.generate_streaming_response("数到3"):
            print(chunk, end="", flush=True)
        print()
    else:
        print("\n[测试4] 跳过流式响应测试（API不支持）")

    print("\n" + "=" * 60)
    print("  测试完成")
    print("=" * 60)

    return supports


if __name__ == "__main__":
    try:
        result = test_adaptive_stream()
        print(f"\n最终结果: API {'支持' if result else '不支持'}流式响应")
        print("\n提示: 如果不支持流式响应，系统会自动使用非流式模式并模拟流式输出。")
    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
