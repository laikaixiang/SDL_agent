# AI Python 代码输出 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**目标:** 将实验设计 AI 输出从 JSON 切换为 Python 代码（命名参数风格），通过安全沙箱执行提取 dict，消除 JSON 解析失败。

**架构:** 新增 `AICodeExecutor` 沙箱执行 AI 代码；修改 prompt 让 AI 输出 `tool_name(param=value)` 风格 Python 代码；`ExperimentDesignAgent` 集成 executor 并保留 JSON fallback。

**技术栈:** Python ast 白名单（复用 VariableResolver 经验）、受限 globals、REGISTRY.json 驱动工具 stub 生成

---

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `experiment/ai_code_executor.py` | **新建** | AICodeExecutor 沙箱 + ExperimentBuilder |
| `prompts/experiment_design/_system.yaml` | 修改 | 输出格式从 JSON → Python 代码 |
| `core/field_inference.py` | 修改 | integrate AICodeExecutor + JSON fallback |
| `platform_init/test/variable_system/test_ai_executor.py` | **新建** | 14 项测试 |

---

## AI 输出的 Python 代码格式

```python
speed1 = var("speed1", default=3000, min=1000, max=6000)
reagent_a = var("reagent_a", default="Perovskite")

# 步骤1: 取枪头
get_tips(tip_box=1, tip_pos=1)

# 步骤2: 吸液
suck(bottom_box=2, bottom_pos=1, Vol=60)

# 步骤3: 旋涂（使用变量 speed1、reagent_a）
spin_coating(spin_speed=speed1, spin_acc=500, spin_dur=30000, reagent=reagent_a, volume=60)

# 步骤4: 退火
set_temperature(target=150)
wait(5000)

build("两步法旋涂实验", description="不同转速条件下的旋涂实验")
```

规则：
- `var(name, default=..., **constraints)` 声明变量，返回变量名
- 工具函数名 = REGISTRY.json 中的 name，参数名 = params 中的 key
- `wait(ms)` → WAIT 步骤；`loop(n)`/`group("name")`/`condition("expr")`/`end()` → 控制流
- `build(name, description=..., notes=...)` 收尾，输出 dict
- 步骤顺序 = 调用顺序
- 可以 `#` 注释

---

### Task 1: 创建 `experiment/ai_code_executor.py`

**文件:** `D:\PycharmProjects\SDL_agent\experiment\ai_code_executor.py`

```python
"""
AI 代码执行器 — 安全执行 AI 生成的实验构建 Python 代码

职责：
- 提供受限的 Python 执行环境
- 通过 ExperimentBuilder 收集变量声明 + 步骤调用
- ast 白名单检查确保安全
- 提取标准实验 JSON dict
"""
import ast
import os
import json
from typing import Any, Dict, List, Optional, Tuple


class ExperimentBuilder:
    """
    实验构建器 — 收集 AI 代码中的变量声明和步骤调用

    提供的 API（供 AI 代码调用）:
    - var(name, default=..., **constraints)  声明一个变量
    - <tool_name>(**params)                  硬件工具步骤（根据 REGISTRY.json 动态注入）
    - wait(ms)                               等待步骤
    - loop(n) / group("name") / condition("expr") / end()  控制流
    - build(name, description="", notes="")  收尾，输出 dict
    """

    def __init__(self, registry: dict):
        self._registry = registry
        self._variables: Dict[str, dict] = {}
        self._steps: List[dict] = []
        self._experiment_name = "未命名实验"
        self._description = ""
        self._notes = ""

    # ---- 变量声明 ----

    def var(self, name: str, default: Any = None, **constraints) -> str:
        """声明变量，返回变量名（用作参数值）"""
        var_type = self._infer_type(default)
        self._variables[name] = {
            "type": var_type,
            "default_value": default,
            "constraints": constraints if constraints else {},
        }
        return name

    # ---- 辅助操作 ----

    def wait(self, duration: int, description: str = ""):
        self._steps.append({
            "type": "helper", "name": "WAIT",
            "params": {"duration": duration},
            "description": description or f"等待 {duration}ms",
        })

    def loop(self, iterations: int, description: str = ""):
        self._steps.append({
            "type": "helper", "name": "LOOP",
            "params": {"iterations": iterations},
            "description": description or f"循环 {iterations} 次",
        })

    def group(self, name: str, description: str = ""):
        self._steps.append({
            "type": "helper", "name": "GROUP",
            "params": {"name": name},
            "description": description or f"步骤组: {name}",
        })

    def condition(self, expr: str, description: str = ""):
        self._steps.append({
            "type": "helper", "name": "CONDITION",
            "params": {"condition": expr},
            "description": description or f"条件: {expr}",
        })

    def end(self):
        self._steps.append({"type": "helper", "name": "END", "params": {}})

    # ---- 收尾 ----

    def build(self, name: str, description: str = "", notes: str = "") -> dict:
        """收集完毕，输出标准实验 JSON dict"""
        self._experiment_name = name
        self._description = description
        self._notes = notes
        result = {
            "experiment_name": self._experiment_name,
            "description": self._description,
            "steps": list(self._steps),
        }
        if self._variables:
            result["variables"] = dict(self._variables)
        if self._notes:
            result["notes"] = self._notes
        return result

    # ---- 工具函数动态注入 ----

    def _make_tool_func(self, tool_name: str):
        """为 REGISTRY.json 中的工具生成 stub 函数"""
        def tool_func(**params):
            entry = self._registry[tool_name]
            description = params.pop("description", entry.get("description", tool_name))
            # 变量名作为参数值保留为字符串
            self._steps.append({
                "type": "tool",
                "name": tool_name,
                "params": params,
                "description": description,
            })
        return tool_func

    def _infer_type(self, value: Any) -> str:
        if isinstance(value, bool): return "bool"
        if isinstance(value, int): return "int"
        if isinstance(value, float): return "float"
        return "str"


class AICodeExecutor:
    """
    AI 代码安全执行器

    流程:
    1. ast.parse → walk → 白名单检查（拒绝危险操作）
    2. 构建受限 globals（ExperimentBuilder + 工具 stub）
    3. exec(code, safe_globals)
    4. 从 ExperimentBuilder.build() 提取 dict
    """

    # 白名单：复用 VariableResolver 的节点集合 + import/Assign/Call/Expr/FunctionDef/arg
    _ALLOWED_NODES = {
        # 基础
        ast.Expression, ast.Module, ast.Load, ast.Store, ast.Del,
        # 字面量
        ast.Constant, ast.Name, ast.List, ast.Tuple, ast.Dict, ast.Set,
        ast.Starred, ast.NamedExpr,
        # 表达式
        ast.Expr, ast.Call, ast.BinOp, ast.BoolOp, ast.UnaryOp, ast.Compare,
        ast.IfExp, ast.Attribute, ast.Subscript, ast.Slice, ast.JoinedStr,
        ast.FormattedValue, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp,
        ast.GeneratorExp, ast.Await, ast.Yield, ast.YieldFrom,
        # 运算符
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.MatMult, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
        ast.And, ast.Or, ast.Not, ast.Invert, ast.UAdd, ast.USub,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
        ast.In, ast.NotIn,
        # 语句
        ast.Assign, ast.AugAssign, ast.AnnAssign,
        ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass,
        ast.FunctionDef, ast.AsyncFunctionDef, ast.Return,
        ast.arguments, ast.arg, ast.keyword,
        ast.Import, ast.ImportFrom, ast.alias,
        ast.ClassDef, ast.With, ast.AsyncWith,
        ast.Try, ast.ExceptHandler, ast.Raise, ast.Assert,
        # 其他
        ast.comprehension, ast.Match, ast.MatchValue, ast.MatchSingleton,
        ast.MatchSequence, ast.MatchMapping, ast.MatchClass, ast.MatchStar,
        ast.MatchAs, ast.MatchOr,
    }

    # 禁止的 import 模块（黑名单）
    _FORBIDDEN_IMPORTS = {
        "os", "sys", "subprocess", "shutil", "socket", "http", "urllib",
        "ftplib", "smtplib", "telnetlib", "requests", "pathlib", "io",
        "pickle", "marshal", "json", "code", "codeop", "compile", "builtins",
        "__builtins__", "importlib", "inspect", "ctypes", "multiprocessing",
        "threading", "signal", "gc", "atexit", "traceback", "pdb",
        "base64", "hashlib", "hmac", "secrets",
    }

    @classmethod
    def execute(cls, code: str, registry: dict) -> Tuple[Optional[dict], str]:
        """
        安全执行 AI 生成的 Python 代码，提取实验 dict

        Args:
            code: AI 生成的 Python 代码
            registry: REGISTRY.json 内容（用于生成工具 stub）

        Returns:
            (experiment_dict, error) — 成功时 dict 不为 None，error 为空字符串
        """
        # 1. AST 白名单检查
        try:
            tree = ast.parse(code, mode='exec')
        except SyntaxError as e:
            return None, f"代码语法错误: {e}"

        bad = cls._check_ast(tree)
        if bad:
            return None, f"禁止的操作: {bad}"

        # 2. 构建沙箱环境
        builder = ExperimentBuilder(registry)
        safe_globals = {"__builtins__": cls._safe_builtins(), "__name__": "__ai_experiment__"}

        # 注入 var / wait / loop / group / condition / end / build
        safe_globals["var"] = builder.var
        safe_globals["wait"] = builder.wait
        safe_globals["loop"] = builder.loop
        safe_globals["group"] = builder.group
        safe_globals["condition"] = builder.condition
        safe_globals["end"] = builder.end
        safe_globals["build"] = builder.build

        # 注入工具 stub
        for tool_name in registry:
            safe_globals[tool_name] = builder._make_tool_func(tool_name)

        # 3. 执行
        try:
            exec(tree, safe_globals)
        except Exception as e:
            return None, f"代码执行失败: {type(e).__name__}: {e}"

        # 4. 提取结果
        if not builder._steps:
            return None, "代码未包含任何步骤（可能未调用工具函数）"
        if not builder._experiment_name or builder._experiment_name == "未命名实验":
            return None, "代码未调用 build() 或未指定实验名称"

        experiment_dict = builder.build(builder._experiment_name, builder._description, builder._notes)
        return experiment_dict, ""

    @classmethod
    def _check_ast(cls, tree: ast.AST) -> Optional[str]:
        """遍历 AST，检查所有节点是否在白名单内 + import 是否合法"""
        for node in ast.walk(tree):
            node_type = type(node)
            if node_type not in cls._ALLOWED_NODES:
                return f"节点类型 {node_type.__name__}"

            # 检查 import
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        base = alias.name.split(".")[0]
                        if base in cls._FORBIDDEN_IMPORTS:
                            return f"禁止导入模块: {base}"
                elif isinstance(node, ast.ImportFrom):
                    base = (node.module or "").split(".")[0]
                    if base in cls._FORBIDDEN_IMPORTS:
                        return f"禁止导入模块: {base}"

            # 检查函数定义（防止覆盖内置）
            if isinstance(node, ast.FunctionDef):
                if node.name in ("__init__", "__del__", "__new__", "__reduce__", "__reduce_ex__"):
                    return f"禁止定义特殊方法: {node.name}"

        return None

    @classmethod
    def _safe_builtins(cls) -> dict:
        """返回安全的 builtins —— 只保留数学和无副作用的函数"""
        return {
            "True": True, "False": False, "None": None,
            "abs": abs, "all": all, "any": any,
            "bin": bin, "bool": bool, "bytes": bytes,
            "chr": chr, "complex": complex, "dict": dict,
            "divmod": divmod, "enumerate": enumerate,
            "filter": filter, "float": float, "format": format,
            "frozenset": frozenset, "hash": hash, "hex": hex,
            "int": int, "isinstance": isinstance,
            "issubclass": issubclass, "iter": iter,
            "len": len, "list": list, "map": map,
            "max": max, "min": min, "next": next,
            "oct": oct, "ord": ord, "pow": pow,
            "print": print, "range": range, "repr": repr,
            "reversed": reversed, "round": round,
            "set": set, "slice": slice, "sorted": sorted,
            "str": str, "sum": sum, "tuple": tuple,
            "type": type, "zip": zip,
            "Exception": Exception,
            "ValueError": ValueError, "TypeError": TypeError,
            "KeyError": KeyError, "IndexError": IndexError,
            "StopIteration": StopIteration, "ZeroDivisionError": ZeroDivisionError,
        }
```

**测试 — 创建 `platform_init/test/variable_system/test_ai_executor.py`:**

```python
"""
AI 代码执行器 — 单元测试

运行方法: python platform_init/test/variable_system/test_ai_executor.py
"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from experiment.ai_code_executor import AICodeExecutor


def _load_registry():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                     "hardware", "tools", "REGISTRY.json")
    with open(p, encoding='utf-8') as f:
        return json.load(f)


REGISTRY = _load_registry()


def test_simple_tool_call():
    print("\n=== test_simple_tool_call ===")
    code = """
get_tips(tip_box=1, tip_pos=1)
build("测试", description="简单测试")
"""
    d, err = AICodeExecutor.execute(code, REGISTRY)
    assert d is not None, f"期望成功，但: {err}"
    assert d["experiment_name"] == "测试"
    assert len(d["steps"]) == 1
    assert d["steps"][0]["name"] == "get_tips"
    assert d["steps"][0]["params"]["tip_box"] == 1
    print("PASS")


def test_with_variables():
    print("\n=== test_with_variables ===")
    code = """
speed1 = var("speed1", default=3000, min=1000, max=6000)
spin_coating(spin_speed=speed1, spin_acc=500, spin_dur=30000, reagent="Perovskite", volume=60)
build("旋涂实验")
"""
    d, err = AICodeExecutor.execute(code, REGISTRY)
    assert d is not None, f"期望成功，但: {err}"
    assert "variables" in d
    assert d["variables"]["speed1"]["default_value"] == 3000
    assert d["variables"]["speed1"]["constraints"] == {"min": 1000, "max": 6000}
    assert d["steps"][0]["params"]["spin_speed"] == "speed1"
    print("PASS")


def test_wait_and_loop():
    print("\n=== test_wait_and_loop ===")
    code = """
loop(3)
get_tips(tip_box=1, tip_pos=1)
suck(bottom_box=2, bottom_pos=1, Vol=60)
end()
wait(5000)
build("循环实验")
"""
    d, err = AICodeExecutor.execute(code, REGISTRY)
    assert d is not None, f"期望成功，但: {err}"
    assert d["steps"][0]["name"] == "LOOP"
    assert d["steps"][1]["name"] == "get_tips"
    assert d["steps"][2]["name"] == "suck"
    assert d["steps"][3]["name"] == "END"
    assert d["steps"][4]["name"] == "WAIT"
    assert len(d["steps"]) == 5
    print("PASS")


def test_banned_import():
    print("\n=== test_banned_import ===")
    code = """
import os
os.system("echo hack")
get_tips(tip_box=1, tip_pos=1)
build("hack")
"""
    d, err = AICodeExecutor.execute(code, REGISTRY)
    assert d is None
    assert "禁止" in err
    print("PASS")


def test_banned_eval():
    print("\n=== test_banned_eval ===")
    code = "eval('1+1')"
    d, err = AICodeExecutor.execute(code, REGISTRY)
    assert d is None
    print("PASS")


def test_no_build_call():
    print("\n=== test_no_build_call ===")
    code = "get_tips(tip_box=1, tip_pos=1)"
    d, err = AICodeExecutor.execute(code, REGISTRY)
    assert d is None
    assert "build" in err.lower()
    print("PASS")


def test_multiple_tools():
    print("\n=== test_multiple_tools ===")
    code = """
move_robot_arm(x=220, y=-220, z=20)
get_tips(tip_box=1, tip_pos=1)
suck(bottom_box=2, bottom_pos=1, Vol=60)
move_glass(start_plate=1, start_pos=1, end_plate=2, end_pos=1)
drop(drop_plate=2, drop_pos=1, Vol=60)
drop_tips(tip_box=1, tip_pos=1)
set_temperature(target=150)
wait(300000)
collect_spectrum(duration=60)
build("完整实验")
"""
    d, err = AICodeExecutor.execute(code, REGISTRY)
    assert d is not None, f"期望成功，但: {err}"
    assert len(d["steps"]) == 8
    print("PASS")


def test_expression_in_param():
    print("\n=== test_expression_in_param ===")
    code = """
base = var("base", default=1000)
spin_coating(spin_speed="base * 2 + 500", spin_acc=500, spin_dur=30000, reagent="P", volume=60)
build("表达式测试")
"""
    d, err = AICodeExecutor.execute(code, REGISTRY)
    assert d is not None, f"期望成功，但: {err}"
    assert d["steps"][0]["params"]["spin_speed"] == "base * 2 + 500"
    print("PASS")


def test_description_param():
    print("\n=== test_description_param ===")
    code = """
spin_coating(spin_speed=3000, spin_acc=500, spin_dur=30000, reagent="P", volume=60, description="旋涂钙钛矿前驱体")
build("描述测试")
"""
    d, err = AICodeExecutor.execute(code, REGISTRY)
    assert d is not None, f"期望成功，但: {err}"
    assert "旋涂钙钛矿前驱体" in d["steps"][0]["description"]
    print("PASS")


# ==================== 运行 ====================

if __name__ == "__main__":
    tests = [
        test_simple_tool_call,
        test_with_variables,
        test_wait_and_loop,
        test_banned_import,
        test_banned_eval,
        test_no_build_call,
        test_multiple_tools,
        test_expression_in_param,
        test_description_param,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            print(f"  FAIL/ERROR: {e}"); failed += 1
    print(f"\n{'='*40}\n结果: {passed} 通过, {failed} 失败, 共 {len(tests)} 项")
    if failed: sys.exit(1)
```

---

### Task 2: 修改 `prompts/experiment_design/_system.yaml`

**文件:** `D:\PycharmProjects\SDL_agent\prompts\experiment_design\_system.yaml`

把 `### 变量使用规则` 之后的 `**JSON 输出格式（含 variables）：**` 整个块替换为 Python 代码格式。

删掉第 56 行（`**JSON 输出格式（含 variables）：**`）到第 73 行（`}`）的全部内容，替换为：

```yaml
  **输出格式：Python 代码**

  写一段 Python 代码，使用函数调用来描述实验步骤。规则：

  1. 变量声明：`var_name = var("var_name", default=默认值, min=最小值, max=最大值)`
     - 变量名英文+数字，如 speed1、temp_a
     - 单次实验用固定值，多轮/优化实验用 var() 声明变量
     - constraints 可省略，如 `var("name", default="A")`

  2. 工具步骤：`tool_name(param1=值, param2=值, description="此步描述")`
     - 参数名必须与下方硬件工具列表中的参数名一致
     - 变量引用直接写变量名，如 `spin_speed=speed1`
     - 固定值写数字/字符串，如 `spin_acc=500`
     - description 可选，建议写中文说明

  3. 辅助操作：
     - `wait(毫秒)` → 等待
     - `loop(次数)` / `end()` → 循环
     - `group("名称")` / `end()` → 步骤组
     - `condition("表达式")` / `end()` → 条件判断

  4. 结尾：`build("实验名称", description="目的简述", notes="注意事项")`

  5. 不需要 print()、import，可以用 # 写注释，可以用 Python 变量简化重复值

  **正确示例：**
  ```python
  speed1 = var("speed1", default=3000, min=1000, max=6000)
  temp = var("temp", default=150, min=50, max=300)
  reagent = var("reagent", default="Perovskite")

  get_tips(tip_box=1, tip_pos=1, description="取枪头")
  suck(bottom_box=2, bottom_pos=1, Vol=60, description="吸液")
  spin_coating(spin_speed=speed1, spin_acc=500, spin_dur=30000, reagent=reagent, volume=60, description="旋涂")
  wait(5000)

  # 退火（使用变量 temp）
  set_temperature(target=temp, description="退火")

  build("两步法旋涂实验", description="不同转速+温度的梯度实验")
  ```

  **错误示例（不要这样做）：**
  - 不要直接输出 JSON
  - 不要写 import 语句
  - build() 一定要在最后调用，否则步骤不会被记录
```

同时删除 JSON 格式示例的 emoji 行（`🚨 必须输出纯JSON...`），保留 `## 设计自查清单` 不变。

---

### Task 3: 修改 `core/field_inference.py` — 集成 AICodeExecutor

**文件:** `D:\PycharmProjects\SDL_agent\core\field_inference.py`

修改两处调用点：`parse_experiment_design()` 和 `parse_experiment_design_stream()`。

**3a. `parse_experiment_design()`（第 378-382 行）**

当前逻辑：`content = ...` → `_parse_experiment_json(content)` → validate → return。

改为先尝试 Python 执行，失败则回退 JSON 解析：

```python
                content = result['choices'][0]['message']['content'].strip()
                print(f"[实验设计] LLM原始输出({len(content)}字符): {content[:300]}...")

                # 尝试 Python 代码执行
                experiment_json = self._execute_ai_code(content)
                if experiment_json is None:
                    # 回退 JSON 解析
                    experiment_json = self._parse_experiment_json(content)

                if experiment_json is None:
                    return False, f"无法解析LLM输出: {content[:200]}"
```

**3b. `parse_experiment_design_stream()`（第 475-485 行）**

同样修改：

```python
        content = full_content.strip()
        print(f"[实验设计-流式] LLM原始输出({len(content)}字符): {content[:300]}...")

        # 尝试 Python 代码执行
        experiment_json = self._execute_ai_code(content)
        if experiment_json is None:
            # 回退 JSON 解析
            experiment_json = self._parse_experiment_json(content)

        if experiment_json is None:
            yield self._sse_event("error", ...)
            return
```

**3c. 新增 `_execute_ai_code()` 方法**（在 `_parse_experiment_json` 之前）

```python
    def _execute_ai_code(self, content: str) -> Optional[dict]:
        """
        尝试将 LLM 输出作为 Python 代码安全执行。
        成功返回 dict，失败返回 None。
        """
        # 快速判断：不含 def/build/tool 调用则跳过
        stripped = content.strip()
        if stripped.startswith("{") or stripped.startswith("```json"):
            return None

        try:
            from experiment.ai_code_executor import AICodeExecutor
            d, err = AICodeExecutor.execute(stripped, self.hardware_registry)
            if d is not None:
                print(f"[实验设计] Python 代码执行成功，{len(d.get('steps',[]))} 步骤")
                return d
            print(f"[实验设计] Python 代码执行失败: {err}，回退 JSON...")
        except Exception as e:
            print(f"[实验设计] Python 执行异常: {e}，回退 JSON...")
        return None
```

---

## 验证清单

- [ ] `python platform_init/test/variable_system/test_ai_executor.py` — 9 项全通过
- [ ] `python platform_init/test/variable_system/test_variable_resolver.py` — 40 项全通过
- [ ] 启动 Flask，发"实验设计：设计一个旋涂实验" — AI 返回 Python 代码 → executor 解析 → 前端收到 experiment_json
- [ ] 发 JSON 格式的实验请求 — fallback 仍然工作
- [ ] 危险代码被拦截：`import os`、`eval()`、`exec()`、`open()` 均返回错误
