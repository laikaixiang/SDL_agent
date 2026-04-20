"""
测试JSON验证问题 - 查看LLM生成的原始JSON

诊断为什么validate_experiment_json返回False
"""

import sys
import os
import json
import io

# 设置stdout为UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from core.field_inference import ExperimentDesignAgent
from core.config import Config
from core.llm_client import LLMClient


def test_raw_llm_output():
    """测试LLM的原始输出"""
    print("\n" + "="*80)
    print("测试LLM原始输出")
    print("="*80)

    agent = ExperimentDesignAgent()
    user_description = "设计一个简单的旋涂实验"

    prompt = (
        f"{agent.system_prompt}\n\n"
        f"用户需求：{user_description}\n\n"
        "请根据上述需求设计实验方案，直接输出JSON格式。"
    )

    print(f"\n[提示词长度]: {len(prompt)} 字符")
    print(f"\n[提示词前500字符]:")
    print(prompt[:500])
    print("...")

    messages = [{"role": "user", "content": prompt}]

    print("\n[调用LLM API]...")
    result = agent.llm_client.call_api(
        model=agent.config.MODEL_NAME_TALK,
        messages=messages,
        temperature=0.3,
        max_tokens=2048
    )

    if result:
        content = result['choices'][0]['message']['content'].strip()
        print(f"\n[LLM原始输出长度]: {len(content)} 字符")
        print(f"\n[LLM原始输出]:")
        print("="*80)
        print(content)
        print("="*80)

        # 清理markdown标记
        cleaned = content.replace("```json", "").replace("```", "").strip()
        print(f"\n[清理后长度]: {len(cleaned)} 字符")

        # 尝试解析JSON
        try:
            experiment_json = json.loads(cleaned)
            print(f"\n✅ JSON解析成功")
            print(f"\n[解析后的JSON]:")
            print(json.dumps(experiment_json, ensure_ascii=False, indent=2))

            # 检查结构
            print(f"\n[JSON结构检查]:")
            print(f"  - 顶层字段: {list(experiment_json.keys())}")
            print(f"  - 是否有steps: {'steps' in experiment_json}")

            if 'steps' in experiment_json:
                steps = experiment_json['steps']
                print(f"  - steps类型: {type(steps)}")
                print(f"  - steps长度: {len(steps) if isinstance(steps, list) else 'N/A'}")

                if isinstance(steps, list) and len(steps) > 0:
                    print(f"\n[第一个步骤详情]:")
                    first_step = steps[0]
                    print(f"  - 类型: {type(first_step)}")
                    print(f"  - 字段: {list(first_step.keys()) if isinstance(first_step, dict) else 'N/A'}")
                    print(f"  - 是否有type: {'type' in first_step if isinstance(first_step, dict) else False}")
                    print(f"  - 是否有name: {'name' in first_step if isinstance(first_step, dict) else False}")
                    print(f"  - 是否有params: {'params' in first_step if isinstance(first_step, dict) else False}")
                    print(f"  - 是否有action: {'action' in first_step if isinstance(first_step, dict) else False}")
                    print(f"\n  完整内容:")
                    print(f"  {json.dumps(first_step, ensure_ascii=False, indent=4)}")

            # 运行验证
            print(f"\n[运行validate_experiment_json]:")
            is_valid = agent.validate_experiment_json(experiment_json)
            print(f"  结果: {'✅ 有效' if is_valid else '❌ 无效'}")

            if not is_valid:
                print(f"\n[验证失败原因分析]:")
                # 逐步检查
                if "steps" not in experiment_json:
                    print(f"  ❌ 缺少steps字段")
                else:
                    steps = experiment_json.get("steps", [])
                    if not isinstance(steps, list):
                        print(f"  ❌ steps不是列表类型")
                    elif len(steps) == 0:
                        print(f"  ❌ steps为空列表")
                    else:
                        print(f"  ✅ steps字段正常 (长度: {len(steps)})")

                        # 检查每个步骤
                        for i, step in enumerate(steps):
                            print(f"\n  步骤 {i+1}:")
                            if not isinstance(step, dict):
                                print(f"    ❌ 不是字典类型")
                            else:
                                missing = []
                                if "type" not in step:
                                    missing.append("type")
                                if "name" not in step:
                                    missing.append("name")
                                if "params" not in step:
                                    missing.append("params")

                                if missing:
                                    print(f"    ❌ 缺少字段: {', '.join(missing)}")
                                    print(f"    实际字段: {list(step.keys())}")
                                else:
                                    print(f"    ✅ 字段完整")

        except json.JSONDecodeError as e:
            print(f"\n❌ JSON解析失败: {e}")
            print(f"错误位置: 第{e.lineno}行, 第{e.colno}列")
            print(f"错误内容: {e.msg}")
        except Exception as e:
            print(f"\n❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n❌ API调用失败")


def main():
    """主函数"""
    test_raw_llm_output()


if __name__ == "__main__":
    main()
