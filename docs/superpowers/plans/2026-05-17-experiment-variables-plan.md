# 实验设计变量系统 — 实现计划

> **For agentic workers:** 每个 Task 分配给对应的专业 agent 执行。步骤使用 checkbox (`- [ ]`) 语法追踪。

**目标:** 为实验设计系统添加变量支持——参数可用变量名代替字面量，变量栏管理，CSV 批量导入/执行。

**架构:** 后端新增 VariableResolver 模块做变量校验/解析/表达式求值；前端新增 VariableBar 组件 + StepEditor blur 检测；prompt 告知 AI 何时/如何输出变量；执行/编译路径集成变量解析。

**技术栈:** Python (ast safe_eval, Flask), Vue 3 + TypeScript (Pinia, Composition API)

**Spec:** `docs/superpowers/specs/2026-05-17-experiment-variables-design.md`

---

## Agent 分工

| Agent | 职责 | Task |
|-------|------|------|
| **prompt-engineer** | 修改 `_system.yaml` | Task 1 |
| **backend-engineer** | `variable_resolver.py`, `executor.py`, `compiler.py`, `format.py`, `app.py` | Task 2-6 |
| **frontend-engineer** | `types/experiment.ts`, `api/experiment.ts`, `stores/experiment.ts`, `StepEditor.vue`, `VariableBar.vue`, `ExperimentPage.vue` | Task 7-12 |

---

### Task 1: Prompt — 变量声明规则

**Agent:** prompt-engineer
**Files:** `prompts/experiment_design/_system.yaml`

在输出格式部分新增变量规则。当前文件内容已在会话中确认，修改 `template` 中的"输出格式"段，在 `## 输出格式` 之后、"🚨 必须输出纯JSON"之前，插入变量规则段落。

修改后 `## 输出格式` 到 `🚨 必须输出纯JSON` 之间的内容变为：

```yaml
  ## 输出格式

  ### 变量使用规则

  **何时使用变量：**
  - 单次实验（固定参数）→ 直接使用数字/字符串字面量
  - 多轮实验（梯度变化、筛选优化）→ 变化的参数使用变量名代替数字
  - 用户明确指定某参数用变量时，优先使用变量
  - 不确定或疑似有误时，在 reply 中询问用户，不要自行决定

  **变量声明格式（在 experiment JSON 顶层）：**
  ```json
  {
    "experiment_name": "...",
    "variables": {
      "var_name": {
        "type": "int",
        "default_value": 3000,
        "constraints": { "min": 1000, "max": 6000 }
      }
    },
    "steps": [...]
  }
  ```
  - type: int / float / str / bool
  - default_value: 必须是整数（int/float类型下）
  - constraints 可选字段: min, max, required, options

  **步骤中引用变量（params 内，无需 $ 前缀）：**
  ```json
  {
    "type": "tool",
    "name": "spin_coating",
    "params": {
      "spin_speed": "speed1",
      "spin_dur": "duration",
      "reagent": "reagent_name",
      "spin_acc": 500,
      "volume": 60
    }
  }
  ```
  - `"spin_speed": "speed1"` → 引用变量 speed1
  - `"spin_acc": 500` → 固定字面量
  - 变量名规则：英文+数字，语义化，不重复

  **JSON 输出格式：**
  🚨 必须输出纯JSON，不要有Markdown标记（如```json）、代码块或解释文字。
  🚨 JSON格式：
  {
    "experiment_name": "实验名称（简洁描述）",
    "description": "实验目的和方法的简要描述",
    "variables": {
      "变量名": {"type": "int|float|str|bool", "default_value": 默认值, "constraints": {}}
    },
    "steps": [...],
    "notes": "注意事项、假设说明、安全提醒"
  }
```

---

### Task 2: Backend — `core/variable_resolver.py`

**Agent:** backend-engineer
**Files:** Create `core/variable_resolver.py`

```python
"""
变量解析器 - 实验设计变量校验、解析、表达式求值

职责：
- 校验变量声明完整性（引用的变量是否已声明）
- 校验变量类型匹配 + 约束满足
- 将步骤参数中的变量引用替换为实际值
- 表达式求值（ast safe_eval，仅算术+逻辑）
- CSV 批量数据解析
"""
import ast
import operator
import csv
import io
from typing import Any, Dict, List, Optional, Tuple


class VariableResolver:
    """变量解析器 — 无状态，所有方法为静态或实例方法"""

    # 表达式引擎：允许的 AST 节点白名单
    _ALLOWED_NODES = {
        ast.Expression, ast.BinOp, ast.BoolOp, ast.UnaryOp, ast.Compare,
        ast.Name, ast.Constant, ast.Load,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq,
        ast.And, ast.Or, ast.Not, ast.USub,
        ast.BoolOp,
    }

    # 允许的二元运算符
    _OPERATORS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    # 允许的比较运算符
    _COMPARATORS = {
        ast.Gt: operator.gt, ast.Lt: operator.lt,
        ast.GtE: operator.ge, ast.LtE: operator.le,
        ast.Eq: operator.eq, ast.NotEq: operator.ne,
    }

    # ========== 公开 API ==========

    def validate_variables(self, variables: Dict[str, dict], steps: List[dict]) -> Tuple[bool, str]:
        """
        执行前校验：所有步骤参数中引用的变量是否已声明、类型是否匹配、约束是否满足。

        Returns:
            (is_valid, error_message) — error_message 为空字符串表示通过
        """
        if not variables:
            return True, ""

        # 1. 收集所有步骤中引用的变量名
        referenced = self._collect_referenced_variables(steps)

        # 2. 检查每个引用的变量是否已声明
        for var_name in referenced:
            if var_name not in variables:
                return False, f"变量 '{var_name}' 被引用但未声明"

        # 3. 检查变量类型和约束
        for var_name, var_def in variables.items():
            var_type = var_def.get("type", "str")
            default = var_def.get("default_value")
            constraints = var_def.get("constraints", {})

            if default is not None:
                ok, msg = self._check_value_against_type(var_name, default, var_type, constraints)
                if not ok:
                    return False, msg

        return True, ""

    def resolve(self, experiment_json: dict) -> dict:
        """
        将实验 JSON 中的变量引用替换为默认值，计算表达式。

        修改步骤 params 中的变量引用，返回新的 experiment_json（不修改原对象）。
        """
        import copy
        resolved = copy.deepcopy(experiment_json)
        variables = resolved.get("variables", {})
        if not variables:
            return resolved

        for step in resolved.get("steps", []):
            params = step.get("params", {})
            for key, value in list(params.items()):
                if isinstance(value, str):
                    new_value = self._resolve_param_value(value, variables)
                    params[key] = new_value

        return resolved

    def resolve_batch(self, experiment_json: dict) -> List[dict]:
        """批量模式：遍历 batch_data，每行生成一个 resolved JSON"""
        batch_data = experiment_json.get("batch_data", [])
        if not batch_data:
            return [self.resolve(experiment_json)]

        results = []
        for row in batch_data:
            import copy
            one = copy.deepcopy(experiment_json)
            # 用行数据覆盖 variables 的 default_value
            variables = one.get("variables", {})
            for col_name, col_value in row.items():
                if col_name in variables:
                    variables[col_name]["default_value"] = col_value
                else:
                    # CSV 有但变量栏没有 → 自动推断类型并新增
                    inferred_type = self._infer_type(col_value)
                    variables[col_name] = {
                        "type": inferred_type,
                        "default_value": col_value,
                    }
            resolved = self.resolve(one)
            resolved["_batch_row"] = row
            results.append(resolved)

        return results

    def evaluate_expression(self, expr: str, variables: Dict[str, Any]) -> Any:
        """
        Safe eval 表达式。仅支持算术 + 比较 + 布尔逻辑。

        Raises:
            ValueError: 表达式语法不合法或使用了不允许的操作
        """
        expr = expr.strip()
        tree = ast.parse(expr, mode='eval')

        # 节点白名单检查
        for node in ast.walk(tree):
            if type(node) not in self._ALLOWED_NODES:
                raise ValueError(f"表达式包含不支持的操作: {type(node).__name__}")

        return self._eval_node(tree.body, variables)

    def parse_csv(self, csv_content: str) -> Tuple[Dict[str, dict], List[Dict[str, Any]], str]:
        """
        解析 CSV 内容，返回 (variables_dict, batch_data_list, error_message)

        CSV 第一行为 header（变量名），后续行为数据。
        从第一行数据推断变量类型。
        """
        try:
            reader = csv.DictReader(io.StringIO(csv_content))
            headers = reader.fieldnames
            if not headers:
                return {}, [], "CSV 文件为空或没有表头"

            rows = list(reader)
            if not rows:
                return {}, [], "CSV 文件没有数据行"

            # 从第一行推断类型
            variables = {}
            for col in headers:
                sample = rows[0].get(col, "")
                var_type = self._infer_type(sample)
                variables[col] = {
                    "type": var_type,
                    "default_value": self._coerce_value(sample, var_type),
                }

            # 构建 batch_data
            batch_data = []
            for row in rows:
                typed_row = {}
                for col in headers:
                    raw = row.get(col, "")
                    var_type = variables[col]["type"]
                    typed_row[col] = self._coerce_value(raw, var_type)
                batch_data.append(typed_row)

            return variables, batch_data, ""

        except Exception as e:
            return {}, [], f"CSV 解析失败: {str(e)}"

    def _resolve_param_value(self, value: str, variables: Dict[str, dict]) -> Any:
        """解析单个参数值：若为变量名则替换为默认值，若含运算符则为表达式求值，否则返回原值"""
        stripped = value.strip()

        # 检查是否为纯变量名（仅字母+数字+下划线，无运算符）
        if self._is_variable_name(stripped):
            if stripped in variables:
                return variables[stripped].get("default_value")
            return stripped  # 未声明的保持原样（校验阶段会报错）

        # 检查是否包含运算符 → 表达式求值
        if any(op in stripped for op in '+-*/<>!=()'):
            var_values = {
                name: v.get("default_value", 0)
                for name, v in variables.items()
            }
            try:
                return self.evaluate_expression(stripped, var_values)
            except (ValueError, SyntaxError):
                return stripped  # 表达式错误保持原样（校验时报告）

        # 既不是变量名也不是表达式 → 数字？字符串？
        try:
            return int(stripped)
        except (ValueError, TypeError):
            pass
        try:
            return float(stripped)
        except (ValueError, TypeError):
            pass
        return stripped

    def _is_variable_name(self, s: str) -> bool:
        """判断是否为纯变量名（字母或下划线开头，只含字母数字下划线）"""
        return bool(s) and (s[0].isalpha() or s[0] == '_') and all(c.isalnum() or c == '_' for c in s)

    def _collect_referenced_variables(self, steps: List[dict]) -> set:
        """从所有步骤 params 中收集引用的变量名"""
        refs = set()
        for step in steps:
            for value in step.get("params", {}).values():
                if isinstance(value, str):
                    stripped = value.strip()
                    if self._is_variable_name(stripped):
                        refs.add(stripped)
        return refs

    def _check_value_against_type(self, name: str, value: Any, var_type: str, constraints: dict) -> Tuple[bool, str]:
        """检查值是否符合类型和约束"""
        if var_type in ("int", "float"):
            if not isinstance(value, (int, float)):
                return False, f"变量 '{name}' 期望类型 {var_type}，但默认值为 '{value}'"
            if var_type == "int" and isinstance(value, float) and value != int(value):
                return False, f"变量 '{name}' 类型为 int，默认值不允许小数"
            num = value
            if "min" in constraints and num < constraints["min"]:
                return False, f"变量 '{name}' 值为 {num}，低于最小值 {constraints['min']}"
            if "max" in constraints and num > constraints["max"]:
                return False, f"变量 '{name}' 值为 {num}，超过最大值 {constraints['max']}"
        elif var_type == "str":
            if not isinstance(value, str):
                return False, f"变量 '{name}' 期望类型 str"
            if "options" in constraints and value not in constraints["options"]:
                return False, f"变量 '{name}' 值 '{value}' 不在允许选项中: {constraints['options']}"
        elif var_type == "bool":
            if not isinstance(value, bool):
                return False, f"变量 '{name}' 期望类型 bool"
        return True, ""

    def _infer_type(self, value: Any) -> str:
        """从值推断变量类型"""
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        s = str(value).strip()
        try:
            int(s)
            return "int"
        except (ValueError, TypeError):
            pass
        try:
            float(s)
            return "float"
        except (ValueError, TypeError):
            pass
        low = s.lower()
        if low in ("true", "false"):
            return "bool"
        return "str"

    def _coerce_value(self, raw: str, var_type: str) -> Any:
        """将字符串值强制转换为目标类型"""
        s = raw.strip()
        if var_type == "int":
            return int(float(s))  # 处理 "3.0" 这种情况
        elif var_type == "float":
            return float(s)
        elif var_type == "bool":
            return s.lower() in ("true", "1", "yes")
        return s

    # ========== AST 表达式求值 ==========

    def _eval_node(self, node, variables: Dict[str, Any]) -> Any:
        """递归求值 AST 节点"""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            raise ValueError(f"变量 '{node.id}' 未定义")
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, variables)
            right = self._eval_node(node.right, variables)
            op_func = self._OPERATORS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
            return op_func(left, right)
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, variables)
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left, variables)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator, variables)
                cmp_func = self._COMPARATORS.get(type(op))
                if cmp_func is None:
                    raise ValueError(f"不支持的比较: {type(op).__name__}")
                if not cmp_func(left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                for value_node in node.values:
                    if not self._eval_node(value_node, variables):
                        return False
                return True
            elif isinstance(node.op, ast.Or):
                for value_node in node.values:
                    if self._eval_node(value_node, variables):
                        return True
                return False
        raise ValueError(f"不支持的 AST 节点: {type(node).__name__}")
```

---

### Task 3: Backend — `experiment/format.py` variables 透传

**Agent:** backend-engineer
**Files:** Modify `experiment/format.py`

修改 `json_to_visual()` 方法，在返回的 dict 中透传 `variables` 字段（如果存在的话）。

在 `json_to_visual()` 的 return 语句处（第100-107行），将 `variables` 加入返回 dict：

原代码（第100-107行）：
```python
        return {
            "experiment_name": experiment_json.get("experiment_name", "未命名实验"),
            "created_at": experiment_json.get("created_at", ""),
            "description": experiment_json.get("description", ""),
            "nodes": nodes,
            "edges": edges,
            "notes": experiment_json.get("notes", "")
        }
```

改为：
```python
        result = {
            "experiment_name": experiment_json.get("experiment_name", "未命名实验"),
            "created_at": experiment_json.get("created_at", ""),
            "description": experiment_json.get("description", ""),
            "nodes": nodes,
            "edges": edges,
            "notes": experiment_json.get("notes", "")
        }
        if "variables" in experiment_json:
            result["variables"] = experiment_json["variables"]
        return result
```

---

### Task 4: Backend — `experiment/executor.py` 集成 VariableResolver

**Agent:** backend-engineer
**Files:** Modify `experiment/executor.py`

在 `execute_plan()` 方法开头（第89行之后，步骤执行循环之前），加入变量校验和解析。

修改位置：第88-90行，在 `results = []` 之后、`try:` 块内 `steps = plan_json.get("steps", [])` 之后。

在第 90 行 `steps = plan_json.get("steps", [])` 之后插入：

```python
            # === 变量解析 ===
            from core.variable_resolver import VariableResolver
            resolver = VariableResolver()
            
            # 执行前校验
            variables = plan_json.get("variables", {})
            if variables:
                is_valid, err_msg = resolver.validate_variables(variables, steps)
                if not is_valid:
                    return {
                        "success": False,
                        "results": [],
                        "error": err_msg
                    }
            
            # 变量解析（替换引用为默认值）
            batch_mode = plan_json.get("batch_mode", False)
            batch_data = plan_json.get("batch_data", [])
            
            if batch_mode and batch_data:
                # 批量模式：逐行解析执行，汇总结果
                all_results = []
                resolved_list = resolver.resolve_batch(plan_json)
                for batch_idx, resolved_plan in enumerate(resolved_list):
                    row_label = resolved_plan.get("_batch_row", {})
                    if progress_callback:
                        progress_callback("batch", "info", f"执行第 {batch_idx + 1}/{len(resolved_list)} 组: {row_label}")
                    # 对每一行做单次执行（递归调用自身）
                    single_result = self._execute_single_plan(
                        resolved_plan, progress_callback, row_label
                    )
                    all_results.extend(single_result.get("results", []))
                return {
                    "success": all(r.get("success", False) for r in all_results),
                    "results": all_results,
                    "error": None
                }
            else:
                # 单次模式：解析后直接使用
                plan_json = resolver.resolve(plan_json)
                steps = plan_json.get("steps", [])
            # === 变量解析结束 ===
```

然后在 `ExperimentExecutor` 类中新增 `_execute_single_plan()` 方法（放在 `execute_plan` 和 `_check_success` 之间）：

```python
    def _execute_single_plan(self, plan_json: dict, progress_callback=None, row_label=None) -> dict:
        """执行单个已解析的实验计划（内部方法，供批量模式调用）"""
        results = []
        steps = plan_json.get("steps", [])
        
        for idx, step in enumerate(steps):
            step_num = step.get("step_number", idx + 1)
            step_type = step.get("type", "tool")
            action = step.get("action") or step.get("name")
            params = step.get("params", {})
            description = step.get("description", "")
            
            if progress_callback:
                progress_callback(step_num, "running", f"正在执行: {description}")
            
            if step_type == "software":
                sw_result = self._execute_software_algorithm(step)
                is_success = sw_result.get("success", False)
                result_msg = sw_result.get("message", "算法执行完成" if is_success else "算法执行失败")
                results.append({
                    "step": step_num, "action": action, "description": description,
                    "result": result_msg, "detail": sw_result.get("result"), "success": is_success
                })
                if progress_callback:
                    progress_callback(step_num, "completed" if is_success else "error", result_msg)
                continue
            
            if step_type == "helper":
                if action in self.helper_map:
                    try:
                        result = self.helper_map[action](params)
                        results.append({
                            "step": step_num, "action": action, "description": description,
                            "result": result, "success": True
                        })
                        if progress_callback:
                            progress_callback(step_num, "completed", result)
                    except Exception as e:
                        results.append({
                            "step": step_num, "action": action, "description": description,
                            "result": f"执行失败: {str(e)}", "success": False
                        })
                        if progress_callback:
                            progress_callback(step_num, "error", str(e))
                continue
            
            if self._hardware_agent.is_known_tool(action):
                try:
                    agent_result = self._hardware_agent.execute_tool_call({"name": action, "params": params})
                    result = agent_result.get("result", "") or agent_result.get("message", "")
                    is_success = agent_result.get("status") == "success"
                    results.append({
                        "step": step_num, "action": action, "description": description,
                        "result": result, "success": is_success
                    })
                    if progress_callback:
                        progress_callback(step_num, "completed" if is_success else "error", result)
                except Exception as e:
                    results.append({
                        "step": step_num, "action": action, "description": description,
                        "result": f"执行失败: {str(e)}", "success": False
                    })
                    if progress_callback:
                        progress_callback(step_num, "error", str(e))
            else:
                results.append({
                    "step": step_num, "action": action, "description": description,
                    "result": f"未知操作类型: {action}", "success": False
                })
                if progress_callback:
                    progress_callback(step_num, "error", f"未知操作类型: {action}")
        
        # 自动启动实验序列
        has_spin_coating = any(r.get("action") == "spin_coating" for r in results)
        if has_spin_coating:
            start_result = self._hardware_agent.execute_tool_call({"name": "start_experiment", "params": {}}).get("result", "")
            results.append({
                "step": "final", "action": "start_experiment", "description": "启动实验序列",
                "result": start_result, "success": "成功" in start_result or "已发送" in start_result
            })
        
        return {"success": all(r.get("success", False) for r in results), "results": results, "error": None}
```

---

### Task 5: Backend — `experiment/compiler.py` 变量引用处理

**Agent:** backend-engineer
**Files:** Modify `experiment/compiler.py`

修改 `_build_tool_call()` 方法（第51-88行），在取值后、类型转换前，先检查是否为变量引用并解析。

在 `_build_tool_call()` 方法中，第59-67行获取 `raw_value` 后，加入变量解析逻辑。同时给方法增加 `variables` 参数。

修改 `_build_tool_call` 签名和取值逻辑：

```python
    @classmethod
    def _build_tool_call(cls, tool_name, params_dict, registry, variables=None):
        """根据registry param顺序生成位置参数调用字符串"""
        if variables is None:
            variables = {}
        
        if tool_name not in registry:
            raise ValueError(f"未知工具 '{tool_name}'，未在REGISTRY.json中注册")

        entry = registry[tool_name]
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

            # === 变量解析 ===
            if isinstance(raw_value, str) and variables:
                from core.variable_resolver import VariableResolver
                resolver = VariableResolver()
                raw_value = resolver._resolve_param_value(raw_value, variables)
            # === 变量解析结束 ===

            ptype = pinfo.get("type", "str")
            # ... 后续类型转换逻辑不变
```

同时修改 `compile_to_python()` 调用 `_build_tool_call` 处（第184行），传入 `variables`：

```python
            # 第179-185行，修改 _build_tool_call 调用：
            elif step_type == "tool":
                variables = experiment_json.get("variables", {})
                code_lines.append(f"{indent}print('执行硬件操作: {step_name}')")
                if step_name in registry:
                    try:
                        call_str = cls._build_tool_call(step_name, params, registry, variables)
                        code_lines.append(f"{indent}result = {call_str}")
                        code_lines.append(f"{indent}print(f'结果: {{result}}')")
                    except ValueError as e:
                        code_lines.append(f"{indent}# ERROR: {e}")
```

---

### Task 6: Backend — `app.py` 新增 `/api/variables/import_csv` + 执行/编译集成

**Agent:** backend-engineer
**Files:** Modify `app.py`

#### 6a. 新增 CSV 导入路由

在 `app.py` 中 `export_experiment_json` 路由之前（约第1660行）新增：

```python
@app.route('/api/variables/import_csv', methods=['POST'])
def import_variables_csv():
    """
    解析 CSV 文件，返回变量定义 + batch_data
    
    POST body: { "csv_content": "name,value\\nspeed,3000\\n..." }
    
    Returns:
    {
        "type": "variables_csv",
        "variables": { "speed": {"type": "int", "default_value": 3000}, ... },
        "batch_data": [ {"speed": 3000}, ... ],
        "reply": "CSV 解析完成，新增 N 个变量，M 行数据"
    }
    """
    data = request.json
    if data is None:
        return jsonify({'type': 'error', 'reply': '请求体为空或JSON格式错误'}), 400
    
    csv_content = data.get('csv_content', '').strip()
    if not csv_content:
        return jsonify({'type': 'error', 'reply': 'CSV 内容为空'}), 400
    
    try:
        from core.variable_resolver import VariableResolver
        resolver = VariableResolver()
        variables, batch_data, err = resolver.parse_csv(csv_content)
        
        if err:
            return jsonify({'type': 'error', 'reply': err}), 400
        
        return jsonify({
            'type': 'variables_csv',
            'variables': variables,
            'batch_data': batch_data,
            'reply': f"✅ CSV 解析完成：新增 {len(variables)} 个变量，{len(batch_data)} 行数据\n变量: {', '.join(variables.keys())}"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'type': 'error', 'reply': f'CSV 导入失败: {str(e)}'}), 500
```

#### 6b. 修改 `experiment_chat()` 的 reply 文案

在 `experiment_chat()` 的 reply 中（约第1114-1118行），如果 result 包含 variables，在 reply 中提示变量数量：

原代码：
```python
                'reply': (
                    f"✅ 已生成实验设计方案：{result.get('experiment_name', '未命名实验')}\n\n"
                    f"{result.get('description', '')}\n\n"
                    f"共 {len(result.get('steps', []))} 个步骤，已推送到实验流程画布。"
                )
```

改为：
```python
                var_count = len(result.get('variables', {}))
                var_hint = f"\n\n📊 包含 {var_count} 个可配置变量，可在变量栏中修改默认值或导入CSV批量执行。" if var_count else ""
                'reply': (
                    f"✅ 已生成实验设计方案：{result.get('experiment_name', '未命名实验')}\n\n"
                    f"{result.get('description', '')}\n\n"
                    f"共 {len(result.get('steps', []))} 个步骤，已推送到实验流程画布。{var_hint}"
                )
```

---

### Task 7: Frontend — `types/experiment.ts` 类型定义

**Agent:** frontend-engineer
**Files:** Modify `frontend/src/types/experiment.ts`

新增 `VariableDefinition` interface，更新 `ExperimentPlan` 加三个可选字段：

```typescript
export type StepType = 'tool' | 'software' | 'helper'
export type HelperType = 'LOOP' | 'GROUP' | 'WAIT' | 'CONDITION' | 'END' | 'USER_INPUT'

export interface ExperimentStep {
  type: StepType
  name: string
  params: Record<string, unknown>
  description?: string
  input_file?: string
  output_file?: string
}

export interface VariableConstraint {
  min?: number
  max?: number
  required?: boolean
  options?: string[]
}

export interface VariableDefinition {
  name: string
  type: 'int' | 'float' | 'str' | 'bool'
  default_value: number | string | boolean
  constraints?: VariableConstraint
  used_in_steps?: string[]  // "步骤1: spin_speed"
}

export interface ExperimentPlan {
  experiment_name: string
  description?: string
  steps: ExperimentStep[]
  created_at?: string
  notes?: string
  variables?: Record<string, VariableDefinition>
  batch_data?: Record<string, unknown>[]
  batch_mode?: boolean
}
```

---

### Task 8: Frontend — `api/experiment.ts` API 层

**Agent:** frontend-engineer
**Files:** Modify `frontend/src/api/experiment.ts`

新增 `importCSV` 函数：

```typescript
export interface ImportCSVResult {
  type: string
  variables: Record<string, VariableDefinition>
  batch_data: Record<string, unknown>[]
  reply: string
}

export async function importCSV(csvContent: string): Promise<ImportCSVResult> {
  return request('/api/variables/import_csv', {
    method: 'POST',
    body: { csv_content: csvContent },
  })
}
```

在文件顶部 import 区域添加：
```typescript
import type { ExperimentPlan, VariableDefinition } from '@/types/experiment'
```

（`VariableDefinition` 在此仅用于 API 返回类型标注）

---

### Task 9: Frontend — `stores/experiment.ts` 变量状态管理

**Agent:** frontend-engineer
**Files:** Modify `frontend/src/stores/experiment.ts`

#### 9a. 新增状态

在 `codeAreaFullscreen` 之后（约第52行）插入：

```typescript
  // === 变量系统 ===
  const variables = ref<Record<string, VariableDefinition>>({})
  const batchData = ref<Record<string, unknown>[]>([])
  const batchMode = ref(false)
  const selectedVariable = ref<string | null>(null)
```

需要导入 `VariableDefinition`：
```typescript
import type { ExperimentStep, ExperimentPlan, HelperType, VariableDefinition } from '@/types/experiment'
```

#### 9b. 更新 plan computed

修改 `plan` computed（第63-67行）以包含变量字段：

```typescript
  const plan = computed<ExperimentPlan>(() => ({
    experiment_name: experimentName.value,
    steps: steps.value,
    created_at: new Date().toISOString(),
    variables: variables.value,
    batch_data: batchData.value.length > 0 ? batchData.value : undefined,
    batch_mode: batchMode.value,
  }))
```

#### 9c. 更新 loadFromJSON

修改 `loadFromJSON()`（第454行）以加载变量：

```typescript
  function loadFromJSON(json: ExperimentPlan) {
    experimentName.value = json.experiment_name || '未命名实验'
    steps.value = (json.steps || []).map((s) => ({
      ...s,
      type: s.type || 'tool',
      params: s.params || {},
    }))
    // 加载变量
    variables.value = json.variables || {}
    batchData.value = json.batch_data || []
    batchMode.value = json.batch_mode || false
    editingStepIndex.value = null
    addLog(`已加载实验: ${experimentName.value}，共 ${steps.value.length} 步`)
  }
```

#### 9d. 新增变量操作方法

在 `clear()` 函数之前（约第475行）新增：

```typescript
  // === 变量操作 ===

  function addVariable(name: string, varType: VariableDefinition['type'] = 'int') {
    if (variables.value[name]) {
      addLog(`变量 "${name}" 已存在`)
      return
    }
    variables.value = {
      ...variables.value,
      [name]: { name, type: varType, default_value: varType === 'str' ? '' : 0 },
    }
    selectedVariable.value = name
  }

  function removeVariable(name: string) {
    const updated = { ...variables.value }
    delete updated[name]
    variables.value = updated
    if (selectedVariable.value === name) selectedVariable.value = null
  }

  function updateVariable(name: string, updates: Partial<VariableDefinition>) {
    if (!variables.value[name]) return
    variables.value = {
      ...variables.value,
      [name]: { ...variables.value[name], ...updates },
    }
  }

  function selectVariable(name: string | null) {
    selectedVariable.value = name
  }

  function isVariableDeclared(name: string): boolean {
    return name in variables.value
  }

  // CSV 导入
  async function importCSVFile(file: File) {
    try {
      const text = await file.text()
      const data = await importCSV(text)
      if (data.variables) {
        // 合并变量：CSV 的变量覆盖同名变量，新增独有变量
        const merged = { ...variables.value }
        for (const [k, v] of Object.entries(data.variables)) {
          merged[k] = { ...v, name: k }
        }
        variables.value = merged
        batchData.value = data.batch_data || []
        addLog(data.reply)
      }
    } catch (err) {
      addLog((err as Error).message)
    }
  }

  // 获取变量被哪些步骤引用
  function getVariableReferences(name: string): string[] {
    const refs: string[] = []
    steps.value.forEach((step, idx) => {
      for (const [paramKey, paramVal] of Object.entries(step.params || {})) {
        if (typeof paramVal === 'string' && paramVal.trim() === name) {
          refs.push(`步骤${idx + 1}: ${paramKey}`)
        }
      }
    })
    return refs
  }
```

#### 9e. 更新 return 导出

在 return 对象中（第535-548行）添加新增的状态和方法：

```typescript
    // 变量系统
    variables, batchData, batchMode, selectedVariable,
    addVariable, removeVariable, updateVariable, selectVariable,
    isVariableDeclared, importCSVFile, getVariableReferences,
```

#### 9f. 导入 importCSV

在顶部 import 处添加：
```typescript
import { generateExperimentStream, compileExperiment, compileAndRun, executeExperiment, saveExperiment, importCSV } from '@/api/experiment'
```

#### 9g. 更新 clear 函数

在 `clear()` 函数中添加变量重置：

```typescript
  function clear() {
    // ... 现有逻辑 ...
    variables.value = {}
    batchData.value = []
    batchMode.value = false
    selectedVariable.value = null
  }
```

---

### Task 10: Frontend — `components/experiment/StepEditor.vue` blur 检测

**Agent:** frontend-engineer
**Files:** Modify `frontend/src/components/experiment/StepEditor.vue`

当前 StepEditor 使用 JSON 字符串编辑 params。需要改为针对 tool 步骤的参数逐个渲染输入框，并加入 blur 检测。

重写 template 中的 tool params 部分（第48-71行区域），为每个参数输入框增加 `@blur` 处理：

```html
      <!-- Tool params: use tool definition to render fields -->
      <template v-if="step.type === 'tool' && toolDef">
        <label>参数</label>
        <div class="param-grid">
          <div v-for="(v, k) in toolDef.params" :key="k" class="param-field">
            <span class="param-label">{{ k }}
              <span v-if="v.required" class="param-req">*</span>
              <span v-else class="param-opt">可选</span>
            </span>
            <span class="param-type">{{ v.type }}</span>
            <div class="param-input-row">
              <input
                class="editor-input param-input"
                :class="{
                  'param-undeclared': paramState[k] === 'undeclared',
                  'param-linked': paramState[k] === 'linked'
                }"
                :placeholder="v.description || k"
                :value="String(step.params[k] ?? v.default ?? '')"
                @input="(e) => {
                  const val = (e.target as HTMLInputElement).value
                  const p = { ...step.params }
                  p[k] = val // 保留原始字符串，不做 Number 转换
                  params = JSON.stringify(p, null, 2)
                }"
                @blur="(e) => onParamBlur(k, (e.target as HTMLInputElement).value)"
              />
              <button
                v-if="paramState[k] === 'undeclared'"
                class="btn-declare"
                @click="onDeclareVariable(k, String(step.params[k] ?? ''))"
              >声明</button>
              <span v-if="paramState[k] === 'linked'" class="param-linked-hint">
                → {{ variablesHint[k] }}
              </span>
            </div>
          </div>
        </div>
      </template>
```

在 `<script setup>` 中新增响应式状态和函数：

```typescript
import { ref, reactive } from 'vue'

// --- 在 props 和 store 声明之后，onSave 函数之前 ---

// 变量检测状态: key = 参数名, value = 'normal' | 'undeclared' | 'linked'
const paramState = reactive<Record<string, 'normal' | 'undeclared' | 'linked'>>({})
const variablesHint = reactive<Record<string, string>>({})

function onParamBlur(paramKey: string, value: string) {
  const trimmed = value.trim()
  if (!trimmed) {
    paramState[paramKey] = 'normal'
    return
  }
  
  // 纯数字 → 正常
  const num = Number(trimmed)
  if (!isNaN(num) && String(num) === trimmed) {
    paramState[paramKey] = 'normal'
    return
  }
  
  // 检查是否为纯变量名（字母开头，无运算符）
  const isVarName = /^[a-zA-Z_][a-zA-Z0-9_]*$/.test(trimmed)
  
  if (isVarName) {
    if (store.isVariableDeclared(trimmed)) {
      paramState[paramKey] = 'linked'
      const v = store.variables[trimmed]
      variablesHint[paramKey] = `→ ${v?.default_value ?? '?'}`
    } else {
      paramState[paramKey] = 'undeclared'
    }
  } else {
    // 含运算符的表达式或其他 → 正常（后端表达式引擎处理）
    paramState[paramKey] = 'normal'
  }
}

function onDeclareVariable(paramKey: string, varName: string) {
  if (!varName.trim()) return
  
  // 从工具参数定义推断类型
  let varType: 'int' | 'float' | 'str' | 'bool' = 'int'
  if (toolDef?.value) {
    const paramDef = toolDef.value.params?.[paramKey]
    if (paramDef?.type === 'str') varType = 'str'
    else if (paramDef?.type === 'float') varType = 'float'
    else if (paramDef?.type === 'bool') varType = 'bool'
  }
  
  store.addVariable(varName.trim(), varType)
  paramState[paramKey] = 'linked'
  variablesHint[paramKey] = `→ 需填写默认值`
}
```

在 `<style scoped>` 中新增样式：

```css
.param-input-row {
  display: flex; align-items: center; gap: 4px;
}

.param-input.param-undeclared {
  border-color: var(--color-error);
  background: rgba(var(--color-error-rgb, 220, 38, 38), 0.05);
}

.param-input.param-linked {
  border-color: var(--color-success, #10b981);
  background: rgba(16, 185, 129, 0.05);
}

.param-linked-hint {
  font-size: 11px; color: var(--color-success, #10b981);
  white-space: nowrap; min-width: 40px;
}

.btn-declare {
  padding: 2px 8px;
  border: 1px solid var(--color-error);
  border-radius: var(--radius-sm);
  background: var(--color-error);
  color: #fff;
  font-size: 11px;
  cursor: pointer;
  white-space: nowrap;
  flex-shrink: 0;
}

.btn-declare:hover { opacity: 0.85; }
```

---

### Task 11: Frontend — `components/experiment/VariableBar.vue` 新组件

**Agent:** frontend-engineer
**Files:** Create `frontend/src/components/experiment/VariableBar.vue`

```html
<script setup lang="ts">
import { useExperimentStore } from '@/stores/experiment'
import { Plus, Upload, Download, Trash2 } from 'lucide-vue-next'

const store = useExperimentStore()

const variableList = computed(() =>
  Object.entries(store.variables).map(([name, def]) => ({ name, ...def }))
)

const selectedIndex = ref<number | null>(null)

function onSelect(index: number) {
  selectedIndex.value = selectedIndex.value === index ? null : index
  const v = variableList.value[index]
  store.selectVariable(v ? v.name : null)
}

function onDelete() {
  if (selectedIndex.value === null) return
  const v = variableList.value[selectedIndex.value]
  if (v && confirm(`确定删除变量 "${v.name}"？`)) {
    store.removeVariable(v.name)
    selectedIndex.value = null
  }
}

function onAddVar() {
  const name = prompt('变量名（英文+数字，如 speed1）：')
  if (!name?.trim()) return
  store.addVariable(name.trim())
}

function onImportCSV() {
  const input = document.createElement('input')
  input.type = 'file'; input.accept = '.csv'
  input.onchange = async () => {
    const file = input.files?.[0]
    if (file) await store.importCSVFile(file)
  }
  input.click()
}

function onExportCSV() {
  const vars = store.variables
  if (!Object.keys(vars).length) return
  const names = Object.keys(vars)
  let csv = names.join(',') + '\n'
  const rowCount = store.batchData.length
  if (rowCount > 0) {
    for (const row of store.batchData) {
      csv += names.map(n => row[n] ?? vars[n].default_value ?? '').join(',') + '\n'
    }
  } else {
    csv += names.map(n => vars[n].default_value ?? '').join(',') + '\n'
  }
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = 'variables.csv'; a.click()
  URL.revokeObjectURL(url)
}

function onDefaultChange(name: string, value: string) {
  const v = store.variables[name]
  if (!v) return
  let coerced: number | string | boolean = value
  if (v.type === 'int' || v.type === 'float') {
    const n = Number(value)
    if (!isNaN(n)) coerced = v.type === 'int' ? Math.floor(n) : n
  }
  store.updateVariable(name, { default_value: coerced })
}

function onConstraintChange(name: string, value: string) {
  const v = store.variables[name]
  if (!v) return
  const constraints: Record<string, unknown> = { ...(v.constraints || {}) }
  // 解析 "1000-6000" 格式
  const rangeMatch = value.match(/^(\d+)-(\d+)$/)
  if (rangeMatch) {
    constraints.min = Number(rangeMatch[1])
    constraints.max = Number(rangeMatch[2])
  } else if (value === '必填' || value === 'required') {
    constraints.required = true
  }
  store.updateVariable(name, { constraints: constraints as VariableConstraint })
}

function getReferences(name: string): string {
  return store.getVariableReferences(name).join(', ')
}
</script>

<script lang="ts">
import { computed, ref } from 'vue'
import type { VariableConstraint } from '@/types/experiment'
</script>

<template>
  <div class="variable-bar">
    <div class="vb-header">
      <span class="vb-title">变量</span>
      <div class="vb-actions">
        <button class="vb-btn" @click="onAddVar" title="添加变量">
          <Plus :size="14" /> 添加
        </button>
        <button class="vb-btn" @click="onImportCSV" title="CSV导入">
          <Upload :size="14" /> CSV导入
        </button>
        <button
          class="vb-btn vb-btn-danger"
          :disabled="selectedIndex === null"
          @click="onDelete"
          title="删除选中变量"
        >
          <Trash2 :size="14" /> 删除
        </button>
        <button class="vb-btn" @click="onExportCSV" title="CSV导出">
          <Download :size="14" /> CSV导出
        </button>
        <label class="vb-check">
          <input type="checkbox" v-model="store.batchMode" />
          批量模式
        </label>
      </div>
    </div>

    <div v-if="variableList.length === 0" class="vb-empty">
      暂无变量。在步骤参数中输入非数字值，或将 CSV 导入。
    </div>

    <div v-else class="vb-table-wrap">
      <table class="vb-table">
        <thead>
          <tr>
            <th>名称</th>
            <th>默认值</th>
            <th>约束</th>
            <th>引用步骤</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="(v, i) in variableList"
            :key="v.name"
            :class="{ 'vb-row-selected': selectedIndex === i }"
            @click="onSelect(i)"
          >
            <td>{{ v.name }}</td>
            <td>
              <input
                class="vb-input"
                :value="String(v.default_value ?? '')"
                @input="(e) => onDefaultChange(v.name, (e.target as HTMLInputElement).value)"
                @click.stop
                :placeholder="v.default_value === undefined || v.default_value === '' ? '?' : ''"
              />
            </td>
            <td>
              <input
                class="vb-input"
                :value="v.constraints?.min !== undefined ? `${v.constraints.min}-${v.constraints.max ?? ''}` : (v.constraints?.required ? '必填' : '')"
                placeholder="无约束"
                @input="(e) => onConstraintChange(v.name, (e.target as HTMLInputElement).value)"
                @click.stop
              />
            </td>
            <td class="vb-refs">{{ getReferences(v.name) || '—' }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.variable-bar {
  border-top: 1px solid var(--color-border);
  border-bottom: 1px solid var(--color-border);
  background: var(--color-bg-soft);
  padding: var(--space-sm) var(--space-md);
  max-height: 160px;
  display: flex; flex-direction: column;
}

.vb-header {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: var(--space-xs);
  flex-shrink: 0;
}

.vb-title { font-size: 13px; font-weight: 600; color: var(--color-text-secondary); }

.vb-actions { display: flex; align-items: center; gap: 6px; }

.vb-btn {
  display: flex; align-items: center; gap: 3px;
  padding: 3px 8px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-surface);
  color: var(--color-text);
  font-size: 11px;
  cursor: pointer;
}

.vb-btn:hover { background: var(--color-bg-mute); }

.vb-btn-danger:disabled { opacity: 0.35; cursor: not-allowed; }
.vb-btn-danger:not(:disabled):hover { background: rgba(var(--color-error-rgb, 220, 38, 38), 0.1); border-color: var(--color-error); color: var(--color-error); }

.vb-check {
  display: flex; align-items: center; gap: 4px;
  font-size: 11px; color: var(--color-text-secondary); cursor: pointer;
}

.vb-empty {
  font-size: 12px; color: var(--color-text-tertiary); padding: var(--space-sm);
  text-align: center;
}

.vb-table-wrap {
  overflow-x: auto; flex: 1;
}

.vb-table {
  width: 100%; border-collapse: collapse; font-size: 12px;
  table-layout: auto;
}

.vb-table th {
  text-align: left; font-weight: 600; color: var(--color-text-secondary);
  padding: 4px 8px; border-bottom: 1px solid var(--color-border);
  white-space: nowrap;
}

.vb-table td {
  padding: 3px 8px; border-bottom: 1px solid var(--color-border-light);
  white-space: nowrap;
}

.vb-table tbody tr { cursor: pointer; }
.vb-table tbody tr:hover { background: var(--color-bg-mute); }
.vb-row-selected { background: var(--color-primary-soft, rgba(59, 130, 246, 0.08)); }

.vb-input {
  width: 100%; min-width: 60px;
  padding: 2px 6px;
  border: 1px solid transparent;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--color-text);
  font-size: 12px;
  font-family: monospace;
}

.vb-input:focus {
  outline: none; border-color: var(--color-primary);
  background: var(--color-surface);
}

.vb-input::placeholder { color: var(--color-text-tertiary); }

.vb-refs {
  color: var(--color-text-tertiary); font-size: 11px; max-width: 200px;
  overflow: hidden; text-overflow: ellipsis;
}
</style>
```

---

### Task 12: Frontend — `pages/ExperimentPage.vue` 集成 VariableBar

**Agent:** frontend-engineer
**Files:** Modify `frontend/src/pages/ExperimentPage.vue`

在 ExperimentPage 模板中，在 CodeArea 上方插入 VariableBar。需要找到 CodeArea 组件的引用位置。

根据现有架构，ExperimentPage 的结构为：ElementPanel + StepCanvas 为主区域，CodeArea 在底部。在 CodeArea 上方插入 VariableBar。

模板中 `<CodeArea>` 之前插入：

```html
<VariableBar />
```

在 script 开头导入 VariableBar：

```typescript
import VariableBar from '@/components/experiment/VariableBar.vue'
```

---

## 验证清单

- [ ] 在 StepEditor 参数中输入非数字变量名 → blur → 标红 + 声明按钮出现
- [ ] 点击 [声明] → VariableBar 新增一行
- [ ] CSV 导入 → 变量列表更新 + batchData 填充
- [ ] 开启批量模式 → 执行实验遍历每行数据
- [ ] LLM 生成的实验 JSON 包含 variables 字段
- [ ] 变量已声明 → 参数输入框显示绿色链接状态
- [ ] 删除变量 → VariableBar 和参数引用同步
- [ ] compile API → 生成的 Python 代码变量被替换为实际值
- [ ] `npx vue-tsc -b` 无类型错误
- [ ] `npm run build:flask` 构建成功
