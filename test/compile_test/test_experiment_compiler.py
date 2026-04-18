"""
实验编译器测试脚本

功能：
    - compile_experiment(json_file, output_py_file=None): 编译实验JSON为Python代码
    - run_experiment(json_file): 编译并运行实验

使用方法：
    1. 直接运行此文件进行测试
    2. 或在其他脚本中导入使用
"""
import sys
import os
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.experiment_manager import ExperimentManager


def load_experiment_json(filepath):
    """加载实验JSON文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 文件不存在 - {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ 错误: JSON格式错误 - {e}")
        return None
    except Exception as e:
        print(f"❌ 错误: 读取文件失败 - {e}")
        return None


def save_python_code(code, filepath):
    """保存生成的Python代码到文件"""
    try:
        # 确保目录存在
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"✅ Python代码已保存到: {filepath}")
        return True
    except Exception as e:
        print(f"❌ 错误: 保存文件失败 - {e}")
        return False


def compile_experiment(json_file, output_py_file=None):
    """
    编译实验JSON为Python代码

    Args:
        json_file: 输入的实验JSON文件路径
        output_py_file: 输出的Python代码文件路径（可选，默认自动生成）

    Returns:
        tuple: (success, python_code, output_file)
    """
    print(f"📖 正在读取实验JSON: {json_file}")
    experiment_json = load_experiment_json(json_file)

    if experiment_json is None:
        return False, None, None

    experiment_name = experiment_json.get("experiment_name", "未命名实验")
    steps_count = len(experiment_json.get("steps", []))
    print(f"✅ 实验名称: {experiment_name}")
    print(f"✅ 步骤数量: {steps_count}")

    # 如果没有指定输出文件，自动生成
    if output_py_file is None:
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        output_py_file = f"{base_name}_compiled.py"
        print(f"💡 自动生成输出文件名: {output_py_file}")

    # 创建编译器
    manager = ExperimentManager()

    # 编译为Python代码
    print("\n" + "=" * 60)
    print("🔧 正在编译实验JSON为Python代码...")
    print("=" * 60)

    try:
        python_code = manager.compile_to_python(experiment_json)
        print("✅ 编译成功！\n")
        print(python_code)
        print("\n" + "=" * 60)

        # 保存到文件
        if save_python_code(python_code, output_py_file):
            return True, python_code, output_py_file
        else:
            return False, python_code, output_py_file

    except Exception as e:
        print(f"❌ 编译失败: {e}")
        return False, None, None


def run_experiment(json_file):
    """
    编译并运行实验

    Args:
        json_file: 输入的实验JSON文件路径

    Returns:
        dict: 执行结果
    """
    print(f"📖 正在读取实验JSON: {json_file}")
    experiment_json = load_experiment_json(json_file)

    if experiment_json is None:
        return {"success": False, "error": "无法加载JSON文件"}

    experiment_name = experiment_json.get("experiment_name", "未命名实验")
    steps_count = len(experiment_json.get("steps", []))
    print(f"✅ 实验名称: {experiment_name}")
    print(f"✅ 步骤数量: {steps_count}")

    # 创建编译器
    manager = ExperimentManager()

    print("\n" + "=" * 60)
    print("⚡ 正在编译并运行实验...")
    print("=" * 60)

    result = manager.compile_and_run(experiment_json)

    if result["success"]:
        print("✅ 执行成功！\n")
        if result["output"]:
            print("📤 执行输出:")
            print(result["output"])
    else:
        print("❌ 执行失败！\n")
        if result["error"]:
            print("🔴 错误信息:")
            print(result["error"])

    return result


if __name__ == "__main__":
    """测试代码"""
    print("=" * 60)
    print("实验编译器测试")
    print("=" * 60)

    # 测试1: 编译简单实验
    print("\n【测试1】编译简单实验")
    print("-" * 60)
    success, code, output_file = compile_experiment(
        json_file="simple_experiment.json",
        output_py_file="simple_compiled.py"
    )

    if success:
        print(f"\n✅ 测试1通过: 代码已保存到 {output_file}")
    else:
        print("\n❌ 测试1失败")

    # 测试2: 编译复杂实验（自动生成输出文件名）
    print("\n\n【测试2】编译复杂实验（自动生成输出文件名）")
    print("-" * 60)
    success, code, output_file = compile_experiment(
        json_file="sample_experiment.json"
    )

    if success:
        print(f"\n✅ 测试2通过: 代码已保存到 {output_file}")
    else:
        print("\n❌ 测试2失败")

    # 测试3: 编译并运行简单实验
    print("\n\n【测试3】编译并运行简单实验")
    print("-" * 60)
    result = run_experiment("simple_experiment.json")

    if result["success"]:
        print("\n✅ 测试3通过: 实验执行成功")
    else:
        print("\n❌ 测试3失败")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

