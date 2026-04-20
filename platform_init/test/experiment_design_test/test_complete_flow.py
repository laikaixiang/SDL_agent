"""
完整流程测试 - 模拟Flask路由的完整执行流程

测试从用户消息到JSON响应的完整链路
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
from experiment.format import ExperimentFormatConverter


def test_complete_flow():
    """测试完整流程：模拟Flask路由的experiment_chat函数"""
    print("\n" + "="*80)
    print("完整流程测试 - 模拟Flask /api/experiment_chat")
    print("="*80)

    # 模拟请求数据
    user_message = "设计一个简单的旋涂实验"
    session_id = "test_session"

    print(f"\n[请求] session_id: {session_id}")
    print(f"[请求] message: {user_message}")

    try:
        # 步骤1: 创建Agent和Converter
        print("\n" + "-"*60)
        print("步骤1: 创建ExperimentDesignAgent和ExperimentFormatConverter")
        print("-"*60)

        start_time = time.time()
        agent = ExperimentDesignAgent()
        converter = ExperimentFormatConverter()
        elapsed = time.time() - start_time

        print(f"✅ 创建成功 (耗时: {elapsed:.2f}秒)")

        # 步骤2: 调用parse_experiment_design
        print("\n" + "-"*60)
        print("步骤2: 调用agent.parse_experiment_design()")
        print("-"*60)

        start_time = time.time()
        success, result = agent.parse_experiment_design(user_message)
        elapsed = time.time() - start_time

        print(f"调用完成 (耗时: {elapsed:.2f}秒)")
        print(f"success: {success}")

        if not success:
            print(f"❌ 生成失败: {result}")
            return False

        print(f"✅ 生成成功")
        print(f"实验名称: {result.get('experiment_name', '未命名')}")
        print(f"步骤数量: {len(result.get('steps', []))}")

        # 步骤3: 添加时间戳
        print("\n" + "-"*60)
        print("步骤3: 添加时间戳")
        print("-"*60)

        import datetime
        result['created_at'] = datetime.datetime.now().isoformat()
        print(f"✅ 时间戳: {result['created_at']}")

        # 步骤4: 转换为可视化格式
        print("\n" + "-"*60)
        print("步骤4: 转换为前端可视化格式")
        print("-"*60)

        start_time = time.time()
        visual_data = converter.json_to_visual(result)
        elapsed = time.time() - start_time

        print(f"✅ 转换成功 (耗时: {elapsed:.2f}秒)")
        print(f"节点数量: {len(visual_data.get('nodes', []))}")
        print(f"边数量: {len(visual_data.get('edges', []))}")

        # 步骤5: 构造响应
        print("\n" + "-"*60)
        print("步骤5: 构造JSON响应")
        print("-"*60)

        response = {
            'type': 'experiment_design',
            'experiment_json': result,
            'visual_data': visual_data,
            'reply': f"✅ 已生成实验设计方案：{result.get('experiment_name', '未命名实验')}\n\n{result.get('description', '')}\n\n共 {len(result.get('steps', []))} 个步骤，已推送到实验流程画布。"
        }

        print(f"✅ 响应构造成功")
        print(f"响应类型: {response['type']}")
        print(f"响应字段: {list(response.keys())}")

        # 步骤6: 序列化为JSON
        print("\n" + "-"*60)
        print("步骤6: 序列化为JSON字符串")
        print("-"*60)

        start_time = time.time()
        json_str = json.dumps(response, ensure_ascii=False, indent=2)
        elapsed = time.time() - start_time

        print(f"✅ 序列化成功 (耗时: {elapsed:.2f}秒)")
        print(f"JSON长度: {len(json_str)} 字符")
        print(f"\n响应JSON (前500字符):")
        print(json_str[:500])

        # 总结
        print("\n" + "="*80)
        print("✅ 完整流程测试通过")
        print("="*80)
        print("所有步骤均正常执行，问题可能在:")
        print("  1. Flask的超时配置")
        print("  2. 前端的请求超时设置")
        print("  3. 网络传输问题")
        print("  4. Flask的响应处理")

        return True

    except Exception as e:
        print(f"\n❌ 流程执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_timing():
    """带详细计时的测试"""
    print("\n" + "="*80)
    print("详细计时测试")
    print("="*80)

    timings = {}

    try:
        # 导入模块
        start = time.time()
        from core.field_inference import ExperimentDesignAgent
        from experiment.format import ExperimentFormatConverter
        timings['导入模块'] = time.time() - start

        # 创建实例
        start = time.time()
        agent = ExperimentDesignAgent()
        converter = ExperimentFormatConverter()
        timings['创建实例'] = time.time() - start

        # 生成实验设计
        start = time.time()
        success, result = agent.parse_experiment_design("设计一个简单的旋涂实验")
        timings['生成实验设计'] = time.time() - start

        if success:
            # 转换格式
            start = time.time()
            visual_data = converter.json_to_visual(result)
            timings['格式转换'] = time.time() - start

            # 构造响应
            start = time.time()
            response = {
                'type': 'experiment_design',
                'experiment_json': result,
                'visual_data': visual_data,
                'reply': "测试"
            }
            timings['构造响应'] = time.time() - start

            # JSON序列化
            start = time.time()
            json_str = json.dumps(response, ensure_ascii=False)
            timings['JSON序列化'] = time.time() - start

        # 打印计时结果
        print("\n计时结果:")
        print("-"*60)
        total = 0
        for step, duration in timings.items():
            print(f"{step:20s}: {duration:6.2f}秒")
            total += duration
        print("-"*60)
        print(f"{'总计':20s}: {total:6.2f}秒")

        if total > 30:
            print("\n⚠️ 总耗时超过30秒，可能触发前端超时")
        else:
            print("\n✅ 总耗时在合理范围内")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "="*80)
    print("实验设计完整流程测试")
    print("="*80)

    # 测试1: 完整流程
    result1 = test_complete_flow()

    # 测试2: 详细计时
    result2 = test_with_timing()

    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    print(f"完整流程测试: {'✅ 通过' if result1 else '❌ 失败'}")
    print(f"详细计时测试: {'✅ 通过' if result2 else '❌ 失败'}")

    if result1 and result2:
        print("\n✅ 所有测试通过")
        print("\n建议检查:")
        print("  1. Flask配置: 是否有超时限制")
        print("  2. 前端配置: fetch/axios的timeout设置")
        print("  3. 运行Flask测试: python platform_init/test/experiment_design_test/test_flask_route.py")
    else:
        print("\n❌ 测试失败，请检查错误信息")


if __name__ == "__main__":
    main()
