"""
实验编译器 - 将实验JSON编译为可执行Python代码

职责：
- 编译JSON为Python代码
- 执行编译后的代码
- 控制流转换（LOOP→for, CONDITION→if等）
- 变量解析（编译时将变量名替换为默认值）
"""
import subprocess
import tempfile
import os
import json
from pathlib import Path
from core.variable_resolver import VariableResolver


class ExperimentCompiler:
    """
    实验编译器

    职责：
    - 将实验JSON编译为Python代码
    - 编译并执行代码
    - 支持控制流（LOOP/CONDITION/WAIT等）
    """

    _registry_cache = None

    @classmethod
    def _load_registry(cls):
        """加载 REGISTRY.json，模块级缓存"""
        if cls._registry_cache is None:
            registry_path = Path(__file__).parent.parent / "hardware" / "tools" / "REGISTRY.json"
            with open(registry_path, encoding='utf-8') as f:
                cls._registry_cache = json.load(f)
        return cls._registry_cache

    @classmethod
    def _build_imports(cls, steps, registry):
        """收集tool步骤 → 去重 → 生成import语句"""
        tool_names = set()
        for step in steps:
            if step.get("type") == "tool":
                name = step.get("name") or step.get("action", "")
                if name and name in registry:
                    tool_names.add(name)
        if not tool_names:
            return ""
        func_names = sorted(f"execute_{n}" for n in tool_names)
        return f"from hardware import {', '.join(func_names)}"

    @classmethod
    def _build_tool_call(cls, tool_name, params_dict, registry, variables=None):
        """
        根据registry param顺序生成位置参数调用字符串

        Args:
            tool_name: 工具名
            params_dict: 参数字典（可能包含变量名引用）
            registry: 工具注册表
            variables: 变量定义字典（用于解析变量名），如 {"speed1": {"type": "int", "default_value": 3000}}

        Returns:
            str: 位置参数调用字符串，如 "execute_spin_coating(3000, \"Perovskite\")"

        Raises:
            ValueError: 未知工具或参数类型错误
        """
        if tool_name not in registry:
            raise ValueError(f"未知工具 '{tool_name}'，未在REGISTRY.json中注册")

        entry = registry[tool_name]

        # 构建变量值字典（默认值）
        var_values = {}
        if variables:
            for var_name, var_def in variables.items():
                var_values[var_name] = var_def.get("default_value")

        args = []
        for pname, pinfo in entry["params"].items():
            if pname in params_dict:
                raw_value = params_dict[pname]
            elif "default" in pinfo:
                raw_value = pinfo["default"]
            elif pinfo.get("required", False):
                raise ValueError(f"工具 '{tool_name}' 缺少必需参数 '{pname}'")
            else:
                raw_value = None

            # 变量解析：如果raw_value是字符串且variables存在，尝试解析变量/表达式
            if variables and isinstance(raw_value, str) and raw_value.strip():
                try:
                    resolved = VariableResolver._resolve_param_value(raw_value, var_values)
                    raw_value = resolved
                except (ValueError, SyntaxError):
                    pass  # 解析失败，保持原值

            ptype = pinfo.get("type", "str")
            try:
                if ptype == "int":
                    formatted = str(int(raw_value))
                elif ptype == "float":
                    formatted = str(float(raw_value))
                elif ptype == "str":
                    formatted = f'"{raw_value}"'
                elif ptype == "bool":
                    if isinstance(raw_value, str):
                        formatted = "True" if raw_value.lower() in ("true", "1", "yes") else "False"
                    else:
                        formatted = "True" if raw_value else "False"
                else:
                    formatted = repr(raw_value)
            except (ValueError, TypeError):
                raise ValueError(f"参数 '{pname}' 期望类型 {ptype}，但值为 '{raw_value}'")
            args.append(formatted)

        return f"execute_{tool_name}({', '.join(args)})"

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
        registry = self._load_registry()
        import_line = self._build_imports(steps, registry)

        code_lines = [
            "# 自动生成的实验执行代码",
            "import time",
        ]
        if import_line:
            code_lines.append(import_line)
        code_lines.extend([
            "",
            "# 用户输入变量存储",
            "user_vars = {}",
            "",
            "def execute_experiment():",
        ])

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
                if step_name in registry:
                    try:
                        variables = experiment_json.get("variables", {})
                        call_str = self._build_tool_call(step_name, params, registry, variables=variables)
                        code_lines.append(f"{indent}result = {call_str}")
                        code_lines.append(f"{indent}print(f'结果: {{result}}')")
                    except ValueError as e:
                        code_lines.append(f"{indent}# ERROR: {e}")
                else:
                    code_lines.append(f"{indent}# WARNING: 工具 '{step_name}' 未在REGISTRY.json中注册")
                    code_lines.append(f"{indent}print('错误: 未知工具 {step_name}')")

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
    test_experiment = {
        "experiment_name": "测试实验",
        "steps": [
            {"type": "tool", "name": "move_robot_arm", "params": {"x": 100, "y": 200, "z": 300}, "description": "移动机械臂到起点"},
            {"type": "helper", "name": "WAIT", "params": {"duration": 2000}, "description": "等待2秒"},
            {"type": "tool", "name": "spin_coating", "params": {"spin_speed": 3000, "reagent": "Perovskite"}, "description": "旋涂"},
        ]
    }
    compiler = ExperimentCompiler()
    print("=" * 60)
    print("编译实验JSON为Python代码")
    print("=" * 60)
    python_code = compiler.compile_to_python(test_experiment)
    print(python_code)
