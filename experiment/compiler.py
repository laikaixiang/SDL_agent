"""
实验编译器 - 将实验JSON编译为可执行Python代码

职责：
- 编译JSON为Python代码
- 执行编译后的代码
- 控制流转换（LOOP→for, CONDITION→if等）
"""
import subprocess
import tempfile
import os


class ExperimentCompiler:
    """
    实验编译器

    职责：
    - 将实验JSON编译为Python代码
    - 编译并执行代码
    - 支持控制流（LOOP/CONDITION/WAIT等）
    """

    def compile_to_python(self, experiment_json: dict) -> str:
        """
        将实验JSON编译为Python代码

        支持的控制结构：
        - LOOP: for循环
        - GROUP: 单次循环 (for i in range(1))
        - CONDITION: if-else条件判断
        - WAIT: time.sleep()
        - USER_INPUT: input()
        - END: 标志最近的循环/条件/组结束

        Args:
            experiment_json: 实验方案JSON

        Returns:
            str: 生成的Python代码
        """
        steps = experiment_json.get("steps", [])
        code_lines = [
            "# 自动生成的实验执行代码",
            "import time",
            "",
            "# 用户输入变量存储",
            "user_vars = {}",
            "",
            "def execute_experiment():",
        ]

        indent_level = 1
        stack = []  # 用于跟踪嵌套结构 (type, indent_level)

        for idx, step in enumerate(steps):
            step_type = step.get("type", "tool")
            step_name = step.get("name", "")
            params = step.get("params", {})
            description = step.get("description", "")

            indent = "    " * indent_level

            # 处理 END 标记
            if step_type == "helper" and step_name == "END":
                if stack:
                    stack.pop()
                    indent_level -= 1
                continue

            # 添加注释
            if description:
                code_lines.append(f"{indent}# {description}")

            # 处理不同类型的步骤
            if step_type == "helper":
                if step_name == "LOOP":
                    iterations = params.get("iterations", 3)
                    code_lines.append(f"{indent}for _loop_iter in range({iterations}):")
                    stack.append(("LOOP", indent_level))
                    indent_level += 1

                elif step_name == "GROUP":
                    group_name = params.get("name", "步骤组")
                    code_lines.append(f"{indent}# GROUP: {group_name}")
                    code_lines.append(f"{indent}for _group_iter in range(1):")
                    stack.append(("GROUP", indent_level))
                    indent_level += 1

                elif step_name == "CONDITION":
                    condition = params.get("condition", "True")
                    code_lines.append(f"{indent}if {condition}:")
                    stack.append(("CONDITION", indent_level))
                    indent_level += 1

                elif step_name == "WAIT":
                    duration_ms = params.get("duration", 1000)
                    duration_s = duration_ms / 1000.0
                    code_lines.append(f"{indent}time.sleep({duration_s})  # 等待 {duration_s} 秒")

                elif step_name == "USER_INPUT":
                    prompt = params.get("prompt", "请输入参数")
                    variable_name = params.get("variable_name", "user_value")
                    code_lines.append(f"{indent}user_vars['{variable_name}'] = input('{prompt}: ')")

            elif step_type == "tool":
                # 硬件工具调用
                code_lines.append(f"{indent}print('执行硬件操作: {step_name}')")
                code_lines.append(f"{indent}# TODO: 调用硬件函数 {step_name}({params})")

            elif step_type == "software":
                # 算法调用
                algo_name = step_name
                input_file = step.get("input_file", "")
                output_file = step.get("output_file", "")
                code_lines.append(f"{indent}print('执行算法: {algo_name}')")
                code_lines.append(f"{indent}# TODO: 调用算法 {algo_name}")
                if input_file:
                    code_lines.append(f"{indent}# 输入文件: {input_file}")
                if output_file:
                    code_lines.append(f"{indent}# 输出文件: {output_file}")

        # 关闭所有未闭合的结构
        while stack:
            stack.pop()
            indent_level -= 1
            indent = "    " * indent_level
            code_lines.append(f"{indent}pass  # 自动闭合")

        code_lines.append("")
        code_lines.append("if __name__ == '__main__':")
        code_lines.append("    execute_experiment()")
        code_lines.append("")

        return "\n".join(code_lines)

    def compile_and_run(self, experiment_json: dict) -> dict:
        """
        编译实验JSON为Python代码并执行

        Args:
            experiment_json: 实验方案JSON

        Returns:
            dict: 执行结果
                {
                    "success": bool,
                    "code": str,  # 生成的Python代码
                    "output": str,  # 执行输出
                    "error": str  # 错误信息（如果有）
                }
        """
        try:
            # 编译为Python代码
            python_code = self.compile_to_python(experiment_json)

            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(python_code)
                temp_file = f.name

            try:
                # 执行Python代码
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5分钟超时
                    encoding='utf-8'
                )

                return {
                    "success": result.returncode == 0,
                    "code": python_code,
                    "output": result.stdout,
                    "error": result.stderr if result.returncode != 0 else ""
                }
            finally:
                # 清理临时文件
                try:
                    os.unlink(temp_file)
                except:
                    pass

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "code": python_code if 'python_code' in locals() else "",
                "output": "",
                "error": "执行超时（超过5分钟）"
            }
        except Exception as e:
            return {
                "success": False,
                "code": python_code if 'python_code' in locals() else "",
                "output": "",
                "error": f"编译或执行失败: {str(e)}"
            }


if __name__ == "__main__":
    """测试编译器功能"""
    # 示例实验JSON
    test_experiment = {
        "experiment_name": "测试实验",
        "steps": [
            {"type": "helper", "name": "LOOP", "params": {"iterations": 3}, "description": "循环3次"},
            {"type": "helper", "name": "WAIT", "params": {"duration": 1000}, "description": "等待1秒"},
            {"type": "helper", "name": "USER_INPUT", "params": {"prompt": "请输入温度", "variable_name": "temperature"}, "description": "用户输入温度"},
            {"type": "helper", "name": "CONDITION", "params": {"condition": "int(user_vars.get('temperature', 0)) > 100"}, "description": "判断温度"},
            {"type": "tool", "name": "set_temperature", "params": {"temperature": 150}, "description": "设置温度"},
            {"type": "helper", "name": "END", "params": {}, "description": "结束条件"},
            {"type": "helper", "name": "END", "params": {}, "description": "结束循环"},
        ]
    }

    # 创建编译器
    compiler = ExperimentCompiler()

    # 编译为Python代码
    print("=" * 60)
    print("编译实验JSON为Python代码")
    print("=" * 60)
    python_code = compiler.compile_to_python(test_experiment)
    print(python_code)
    print("\n" + "=" * 60)

    # 可选：编译并运行
    run_test = input("\n是否运行生成的代码？(y/n): ").strip().lower()
    if run_test == 'y':
        print("\n" + "=" * 60)
        print("编译并运行实验")
        print("=" * 60)
        result = compiler.compile_and_run(test_experiment)
        if result["success"]:
            print("✅ 执行成功")
            print("\n输出:")
            print(result["output"])
        else:
            print("❌ 执行失败")
            print("\n错误:")
            print(result["error"])
