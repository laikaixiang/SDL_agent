"""
变量解析器 - 实验设计系统的变量解析与校验模块

职责：
- 校验实验JSON中的变量声明和引用
- 将步骤参数中的变量名解析为默认值
- 批量模式支持（多组变量值生成多个执行方案）
- 安全的表达式求值（AST白名单）
- CSV导入（解析CSV生成变量和批量数据）
"""
import ast
import copy
import csv
import io
from typing import Any, Dict, List, Tuple, Optional


class VariableResolver:
    """
    变量解析器

    职责：
    - 变量校验：检查所有引用的变量是否已声明，类型和约束是否满足
    - 变量解析：将步骤params中的变量名替换为默认值
    - 批量解析：遍历batch_data生成多个resolved JSON
    - 表达式求值：安全的AST表达式计算
    - CSV解析：从CSV内容生成变量和批量数据
    """

    # 允许的AST节点白名单
    _ALLOWED_AST_NODES = {
        ast.Expression,
        ast.BinOp,      # 二元运算: +, -, *, /, %, **, etc.
        ast.BoolOp,     # 布尔运算: and, or
        ast.UnaryOp,    # 一元运算: -, not
        ast.Compare,    # 比较运算: ==, !=, <, >, <=, >=
        ast.Name,       # 变量名
        ast.Constant,   # 常量: 数字, 字符串, True/False/None
        ast.Load,       # 加载上下文
        ast.Add,        # +
        ast.Sub,        # -
        ast.Mult,       # *
        ast.Div,        # /
        ast.Mod,        # %
        ast.Pow,        # **
        ast.FloorDiv,   # //
        ast.And,        # and
        ast.Or,         # or
        ast.Not,        # not
        ast.Eq,         # ==
        ast.NotEq,      # !=
        ast.Lt,         # <
        ast.LtE,        # <=
        ast.Gt,         # >
        ast.GtE,        # >=
        ast.Is,         # is
        ast.IsNot,      # is not
        ast.In,         # in
        ast.NotIn,      # not in
        ast.USub,       # 一元负号
        ast.UAdd,       # 一元正号
        ast.Invert,     # 按位取反 ~
    }

    # 类型转换映射
    _TYPE_CONVERTERS = {
        "int":   lambda v: int(float(v)),   # 支持 "3.0" → 3
        "float": lambda v: float(v),
        "str":   lambda v: str(v),
        "bool":  lambda v: bool(v) if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes"),
    }

    # ========== 公开方法 ==========

    @staticmethod
    def _is_variable_name(s: str) -> bool:
        """
        判断字符串是否为纯变量名

        规则：ASCII字母或下划线开头，只含ASCII字母、数字、下划线

        Args:
            s: 待判断的字符串

        Returns:
            bool: 是否为纯变量名
        """
        if not isinstance(s, str):
            return False
        if not s:
            return False
        # 首字符：ASCII字母或下划线
        first = s[0]
        if not (('a' <= first <= 'z') or ('A' <= first <= 'Z') or first == '_'):
            return False
        # 其余字符：ASCII字母、数字、下划线
        for c in s:
            if not (('a' <= c <= 'z') or ('A' <= c <= 'Z') or ('0' <= c <= '9') or c == '_'):
                return False
        return True

    @staticmethod
    def _infer_type(value: Any) -> str:
        """
        从值推断变量类型

        优先级：int > float > bool > str

        Args:
            value: 待推断的值

        Returns:
            str: 类型名 ("int", "float", "bool", "str")
        """
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            # 检查是否可以无损转为int
            if value == int(value) and not isinstance(value, bool):
                return "int"
            return "float"
        if isinstance(value, str):
            # 尝试推断: 优先 int, 其次 float, 然后 bool, 最后 str
            s = value.strip()
            if not s:
                return "str"
            # 尝试 int
            try:
                int(s)
                return "int"
            except ValueError:
                pass
            # 尝试 float
            try:
                float(s)
                return "float"
            except ValueError:
                pass
            # 尝试 bool
            if s.lower() in ("true", "false"):
                return "bool"
            return "str"
        # 默认 str
        return "str"

    @staticmethod
    def _coerce_value(raw: Any, var_type: str) -> Any:
        """
        将原始值强制转换为指定类型

        Args:
            raw: 原始值（可能是字符串）
            var_type: 目标类型 ("int", "float", "str", "bool")

        Returns:
            Any: 转换后的值

        Raises:
            ValueError: 无法转换
        """
        converter = VariableResolver._TYPE_CONVERTERS.get(var_type)
        if converter is None:
            raise ValueError(f"不支持的变量类型: {var_type}")
        try:
            return converter(raw)
        except (ValueError, TypeError) as e:
            raise ValueError(f"无法将值 '{raw}' 转换为类型 {var_type}: {e}")

    @staticmethod
    def _check_value_against_type(name: str, value: Any, var_type: str,
                                   constraints: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        检查值是否符合声明的类型和约束

        Args:
            name: 变量名
            value: 待检查的值
            var_type: 声明的类型
            constraints: 约束字典, 如 {"min": 1000, "max": 6000}

        Returns:
            (bool, str): (是否通过, 错误信息)
        """
        # 类型检查
        type_checks = {
            "int":   lambda v: isinstance(v, (int, float)) and (isinstance(v, int) or v == int(v)),
            "float": lambda v: isinstance(v, (int, float)),
            "str":   lambda v: isinstance(v, str),
            "bool":  lambda v: isinstance(v, bool),
        }
        checker = type_checks.get(var_type)
        if checker is not None and not checker(value):
            return False, f"变量 '{name}' 期望类型 {var_type}，但值为 '{value}'（实际类型 {type(value).__name__}）"

        # 约束检查
        if constraints and isinstance(constraints, dict):
            numeric_types = ("int", "float")
            if var_type in numeric_types:
                try:
                    num_val = float(value)
                except (ValueError, TypeError):
                    return False, f"变量 '{name}' 期望类型 {var_type}，但值为 '{value}'"
                if "min" in constraints and num_val < float(constraints["min"]):
                    return False, f"变量 '{name}' 值 {value} 小于最小值 {constraints['min']}"
                if "max" in constraints and num_val > float(constraints["max"]):
                    return False, f"变量 '{name}' 值 {value} 大于最大值 {constraints['max']}"
            elif var_type == "str":
                if "min_length" in constraints and len(str(value)) < constraints["min_length"]:
                    return False, f"变量 '{name}' 长度 {len(str(value))} 小于最小长度 {constraints['min_length']}"
                if "max_length" in constraints and len(str(value)) > constraints["max_length"]:
                    return False, f"变量 '{name}' 长度 {len(str(value))} 大于最大长度 {constraints['max_length']}"
                if "options" in constraints:
                    options = constraints["options"]
                    if isinstance(options, list) and str(value) not in options:
                        return False, f"变量 '{name}' 值 '{value}' 不在允许选项中: {options}"

        return True, ""

    @staticmethod
    def _collect_referenced_variables(steps: List[Dict]) -> set:
        """
        从所有步骤的params中收集被引用的变量名

        扫描每个步骤的params值：
        - 字符串且为纯变量名 → 收集
        - 字符串且含运算符 → 用AST解析收集Name节点

        Args:
            steps: 步骤列表

        Returns:
            set: 被引用的变量名集合
        """
        referenced = set()

        def _collect_from_value(value):
            """递归收集值中的变量名"""
            if isinstance(value, str):
                # 检查是否为纯变量名
                if VariableResolver._is_variable_name(value):
                    referenced.add(value)
                else:
                    # 检查是否包含运算符表达式
                    try:
                        tree = ast.parse(value, mode='eval')
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Name):
                                referenced.add(node.id)
                    except SyntaxError:
                        pass  # 不是表达式，忽略
            elif isinstance(value, dict):
                for v in value.values():
                    _collect_from_value(v)
            elif isinstance(value, list):
                for item in value:
                    _collect_from_value(item)

        for step in steps:
            params = step.get("params", {})
            for val in params.values():
                _collect_from_value(val)

        return referenced

    @staticmethod
    def validate_variables(variables: Dict[str, Dict], steps: List[Dict]) -> Tuple[bool, str]:
        """
        校验变量定义和引用的一致性

        检查项：
        1. 所有被步骤引用的变量是否已在variables中声明
        2. 变量声明的默认值是否与声明类型匹配
        3. 默认值是否满足约束条件

        Args:
            variables: 变量定义字典
                格式: {"speed1": {"type": "int", "default_value": 3000, "constraints": {"min": 1000, "max": 6000}}}
            steps: 步骤列表

        Returns:
            (bool, str): (是否通过校验, 错误信息)
        """
        if not variables:
            # 没有变量定义，检查步骤中是否有变量引用
            referenced = VariableResolver._collect_referenced_variables(steps)
            if referenced:
                return False, f"变量 '{next(iter(referenced))}' 被引用但未声明"
            return True, ""

        # 规范化：为缺少 type 的变量从 default_value 推断类型
        for var_name, var_def in list(variables.items()):
            if isinstance(var_def, dict) and "type" not in var_def:
                dv = var_def.get("default_value")
                var_def["type"] = VariableResolver._infer_type(dv)

        # 收集步骤中引用的变量名
        referenced = VariableResolver._collect_referenced_variables(steps)

        # 检查所有引用的变量是否已声明
        for var_name in referenced:
            if var_name not in variables:
                return False, f"变量 '{var_name}' 被引用但未声明"

        # 检查每个已声明变量的默认值类型和约束
        for var_name, var_def in variables.items():
            if not isinstance(var_def, dict):
                return False, f"变量 '{var_name}' 的定义格式错误，应为字典"
            var_type = var_def.get("type", "str")
            default_value = var_def.get("default_value")
            constraints = var_def.get("constraints", {})

            if default_value is not None:
                ok, err = VariableResolver._check_value_against_type(
                    var_name, default_value, var_type, constraints
                )
                if not ok:
                    return False, err

        return True, ""

    @staticmethod
    def _resolve_param_value(value: Any, variables: Dict[str, Any]) -> Any:
        """
        解析单个参数值

        处理逻辑：
        - 数字/非字符串 → 直接返回
        - 字符串且为纯变量名 → 从 variables 中查找替换
        - 字符串且含运算符 → 表达式求值
        - 其他字符串 → 直接返回

        Args:
            value: 待解析的参数值
            variables: 当前生效的变量值字典 {"var_name": actual_value}

        Returns:
            Any: 解析后的值
        """
        # 非字符串直接返回
        if not isinstance(value, str):
            return value

        # 空字符串直接返回
        if not value.strip():
            return value

        # 纯变量名 → 查找替换
        if VariableResolver._is_variable_name(value):
            if value in variables:
                return variables[value]
            # 变量未找到：可能是纯字符串值，直接返回原值
            return value

        # 尝试作为表达式求值
        try:
            result = VariableResolver.evaluate_expression(value, variables)
            return result
        except (ValueError, SyntaxError):
            # 不是表达式，原样返回
            return value

    @staticmethod
    def resolve(experiment_json: dict) -> dict:
        """
        将实验JSON中步骤params的变量名替换为默认值

        使用 deepcopy 复制，不修改原对象

        Args:
            experiment_json: 实验JSON（可包含 variables 字段）

        Returns:
            dict: 解析后的实验JSON副本
        """
        result = copy.deepcopy(experiment_json)
        variables_def = result.get("variables", {})
        batch_data = result.get("batch_data", [])

        # 构建默认变量值字典
        var_values = {}
        for var_name, var_def in variables_def.items():
            var_values[var_name] = var_def.get("default_value")

        # 如果有 batch_data，使用第一行数据覆盖默认值
        if batch_data and isinstance(batch_data, list) and len(batch_data) > 0:
            first_row = batch_data[0]
            if isinstance(first_row, dict):
                for k, v in first_row.items():
                    var_values[k] = v

        # 解析每个步骤的params
        steps = result.get("steps", [])
        for step in steps:
            params = step.get("params", {})
            resolved_params = {}
            for key, val in params.items():
                resolved_params[key] = VariableResolver._resolve_param_value(val, var_values)
            step["params"] = resolved_params

        return result

    @staticmethod
    def resolve_batch(experiment_json: dict) -> List[dict]:
        """
        批量解析：遍历 batch_data，每行生成一个 resolved JSON

        Args:
            experiment_json: 实验JSON（必须包含 variables 和 batch_data）

        Returns:
            List[dict]: 解析后的实验JSON列表，每行一个

        Raises:
            ValueError: 缺少 batch_data 或 batch_data 格式错误
        """
        batch_data = experiment_json.get("batch_data", [])
        if not batch_data:
            raise ValueError("批量模式下缺少 batch_data")

        variables_def = experiment_json.get("variables", {})

        # 规范化：为缺少 type 的变量推断类型
        for var_name, var_def in list(variables_def.items()):
            if isinstance(var_def, dict) and "type" not in var_def:
                var_def["type"] = VariableResolver._infer_type(var_def.get("default_value"))

        results = []
        for idx, row in enumerate(batch_data):
            if not isinstance(row, dict):
                raise ValueError(f"batch_data 第 {idx + 1} 行格式错误，应为字典")

            # 校验行数据
            for var_name, val in row.items():
                if var_name not in variables_def:
                    raise ValueError(f"batch_data 第 {idx + 1} 行的变量 '{var_name}' 未在 variables 中声明")
                var_def = variables_def[var_name]
                var_type = var_def.get("type", "str")
                constraints = var_def.get("constraints", {})
                ok, err = VariableResolver._check_value_against_type(var_name, val, var_type, constraints)
                if not ok:
                    raise ValueError(f"batch_data 第 {idx + 1} 行: {err}")

            # 构建该行的变量值：默认值 + 行数据覆盖
            var_values = {}
            for var_name, var_def in variables_def.items():
                var_values[var_name] = var_def.get("default_value")
            for k, v in row.items():
                var_values[k] = v

            # 复制并解析
            resolved = copy.deepcopy(experiment_json)
            steps = resolved.get("steps", [])
            for step in steps:
                params = step.get("params", {})
                resolved_params = {}
                for key, val in params.items():
                    resolved_params[key] = VariableResolver._resolve_param_value(val, var_values)
                step["params"] = resolved_params

            # 标记行
            resolved["_batch_index"] = idx
            resolved["_batch_row"] = row
            results.append(resolved)

        return results

    @staticmethod
    def _eval_node(node, variables: Dict[str, Any]) -> Any:
        """
        递归AST节点求值

        白名单校验：遇到未允许的节点类型时抛出 ValueError

        Args:
            node: AST节点
            variables: 变量值字典

        Returns:
            Any: 求值结果
        """
        node_type = type(node)

        # 白名单检查
        if node_type not in VariableResolver._ALLOWED_AST_NODES:
            raise ValueError(f"不支持的操作: {node_type.__name__}")

        # 常量
        if isinstance(node, ast.Constant):
            return node.value

        # 变量名
        if isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            raise ValueError(f"变量 '{node.id}' 未定义")

        # 二元运算
        if isinstance(node, ast.BinOp):
            left = VariableResolver._eval_node(node.left, variables)
            right = VariableResolver._eval_node(node.right, variables)
            op_map = {
                ast.Add:      lambda a, b: a + b,
                ast.Sub:      lambda a, b: a - b,
                ast.Mult:     lambda a, b: a * b,
                ast.Div:      lambda a, b: a / b,
                ast.Mod:      lambda a, b: a % b,
                ast.Pow:      lambda a, b: a ** b,
                ast.FloorDiv: lambda a, b: a // b,
            }
            op_type = type(node.op)
            if op_type in op_map:
                return op_map[op_type](left, right)
            raise ValueError(f"不支持的二元运算: {op_type.__name__}")

        # 一元运算
        if isinstance(node, ast.UnaryOp):
            operand = VariableResolver._eval_node(node.operand, variables)
            op_map = {
                ast.USub:   lambda v: -v,
                ast.UAdd:   lambda v: +v,
                ast.Invert: lambda v: ~v,
                ast.Not:    lambda v: not v,
            }
            op_type = type(node.op)
            if op_type in op_map:
                return op_map[op_type](operand)
            raise ValueError(f"不支持的一元运算: {op_type.__name__}")

        # 布尔运算
        if isinstance(node, ast.BoolOp):
            values = [VariableResolver._eval_node(v, variables) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
            raise ValueError(f"不支持的布尔运算: {type(node.op).__name__}")

        # 比较运算
        if isinstance(node, ast.Compare):
            left = VariableResolver._eval_node(node.left, variables)
            for op, comparator in zip(node.ops, node.comparators):
                right = VariableResolver._eval_node(comparator, variables)
                cmp_map = {
                    ast.Eq:    lambda a, b: a == b,
                    ast.NotEq: lambda a, b: a != b,
                    ast.Lt:    lambda a, b: a < b,
                    ast.LtE:   lambda a, b: a <= b,
                    ast.Gt:    lambda a, b: a > b,
                    ast.GtE:   lambda a, b: a >= b,
                    ast.Is:    lambda a, b: a is b,
                    ast.IsNot: lambda a, b: a is not b,
                    ast.In:    lambda a, b: a in b,
                    ast.NotIn: lambda a, b: a not in b,
                }
                cmp_type = type(op)
                if cmp_type in cmp_map:
                    if not cmp_map[cmp_type](left, right):
                        return False
                else:
                    raise ValueError(f"不支持的比较运算: {cmp_type.__name__}")
                left = right
            return True

        raise ValueError(f"不支持的AST节点: {node_type.__name__}")

    @staticmethod
    def evaluate_expression(expr: str, variables: Dict[str, Any]) -> Any:
        """
        安全的表达式求值

        仅允许白名单AST节点，禁止函数调用、import、属性访问等

        Args:
            expr: 表达式字符串，如 "speed1 * 2 + 100"
            variables: 变量值字典

        Returns:
            Any: 求值结果

        Raises:
            ValueError: 表达式包含不允许的操作
            SyntaxError: 表达式语法错误
        """
        if not isinstance(expr, str) or not expr.strip():
            return expr

        try:
            tree = ast.parse(expr.strip(), mode='eval')
        except SyntaxError:
            raise SyntaxError(f"表达式语法错误: {expr}")

        return VariableResolver._eval_node(tree.body, variables)

    @staticmethod
    def parse_csv(csv_content: str) -> Tuple[Dict[str, Dict], List[Dict], Optional[str]]:
        """
        解析CSV内容，生成变量定义和批量数据

        CSV格式：
        - 第一行为表头（变量名）
        - 后续行为数据
        - 第一行数据用于推断变量类型

        Args:
            csv_content: CSV文本内容

        Returns:
            Tuple:
                - variables: 变量定义字典
                - batch_data: 批量数据列表
                - error: 错误信息（成功时为None）
        """
        if not csv_content or not csv_content.strip():
            return {}, [], "CSV内容为空"

        try:
            reader = csv.DictReader(io.StringIO(csv_content.strip()))
            rows = list(reader)
        except Exception as e:
            return {}, [], f"CSV解析失败: {str(e)}"

        if not rows:
            return {}, [], "CSV中没有数据行"

        # 获取字段名（表头）
        fieldnames = reader.fieldnames
        if not fieldnames:
            return {}, [], "CSV表头为空"

        # 清理字段名（去空格）
        fieldnames = [f.strip() for f in fieldnames]

        # 使用第一行数据推断每个变量的类型
        variables: Dict[str, Dict] = {}
        first_row = rows[0]

        for var_name in fieldnames:
            raw_value = first_row.get(var_name, "").strip()
            var_type = VariableResolver._infer_type(raw_value)

            # 转换第一行值为正确类型，作为示例默认值
            try:
                default_value = VariableResolver._coerce_value(raw_value, var_type)
            except ValueError:
                # 无法转换，使用字符串
                var_type = "str"
                default_value = raw_value

            variables[var_name] = {
                "type": var_type,
                "default_value": default_value,
                "constraints": {},
            }

        # 转换所有行为正确的类型
        batch_data: List[Dict] = []
        for row in rows:
            converted_row = {}
            for var_name in fieldnames:
                raw_value = row.get(var_name, "").strip()
                var_type = variables[var_name]["type"]
                try:
                    converted_row[var_name] = VariableResolver._coerce_value(raw_value, var_type)
                except ValueError:
                    # 类型转换失败，保留原始字符串
                    converted_row[var_name] = raw_value
            batch_data.append(converted_row)

        return variables, batch_data, None
