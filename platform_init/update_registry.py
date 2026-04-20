"""
更新 hardware/tools 注册表的工具函数。

同时维护三处：
  1. hardware/tools/REGISTRY.json  — 供 LLM/ExperimentDesignAgent 离线读取
  2. hardware/tools/{name}.py      — 带 @register_tool 装饰器的工具函数文件
  3. hardware/tools/__init__.py    — 运行时导入和 __all__
"""

import json
import os
import re

_ROOT = os.path.join(os.path.dirname(__file__), "..")
REGISTRY_PATH = os.path.join(_ROOT, "hardware", "tools", "REGISTRY.json")
TOOLS_DIR = os.path.join(_ROOT, "hardware", "tools")
INIT_PATH = os.path.join(TOOLS_DIR, "__init__.py")


# ── JSON 注册表 ────────────────────────────────────────────────────────────────

def _load_registry() -> dict:
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(registry: dict) -> None:
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


# ── __init__.py 同步 ───────────────────────────────────────────────────────────

def _add_to_init(func_name: str, module_name: str) -> None:
    """在 __init__.py 中添加 import 行和 __all__ 条目（幂等）。"""
    with open(INIT_PATH, "r", encoding="utf-8") as f:
        src = f.read()

    import_line = f"from .{module_name} import {func_name}"
    if import_line not in src:
        # 插到最后一个 from .xxx import 行之后
        last = max((m.end() for m in re.finditer(r"^from \.\w+ import \w+", src, re.M)), default=0)
        src = src[:last] + "\n" + import_line + src[last:]

    all_entry = f"    '{func_name}',"
    if all_entry not in src:
        src = src.replace("__all__ = [", f"__all__ = [\n{all_entry}", 1)
        # 去掉多余的空行
        src = src.replace("__all__ = [\n\n", "__all__ = [\n")

    with open(INIT_PATH, "w", encoding="utf-8") as f:
        f.write(src)


def _remove_from_init(func_name: str, module_name: str) -> None:
    """从 __init__.py 中移除 import 行和 __all__ 条目（幂等）。"""
    with open(INIT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    filtered = [
        l for l in lines
        if f"from .{module_name} import {func_name}" not in l
        and f"    '{func_name}'," not in l
    ]
    with open(INIT_PATH, "w", encoding="utf-8") as f:
        f.writelines(filtered)


# ── .py 工具文件生成 ───────────────────────────────────────────────────────────

def _params_to_signature(params: dict) -> str:
    type_map = {"int": "int", "float": "float", "str": "str", "bool": "bool"}
    parts = []
    for pname, pdef in params.items():
        t = type_map.get(pdef.get("type", "str"), "str")
        if "default" in pdef:
            parts.append(f"{pname}: {t} = {repr(pdef['default'])}")
        else:
            parts.append(f"{pname}: {t}")
    return ", ".join(parts)


def _generate_tool_file(name: str, description: str, params: dict) -> str:
    func_name = f"execute_{name}"
    params_json = json.dumps(params, ensure_ascii=False, indent=8)
    sig = _params_to_signature(params)
    return f"""\
from .registry import register_tool
from ..mqtt import get_mqtt_client


@register_tool(
    name="{name}",
    description="{description}",
    params={params_json}
)
def {func_name}({sig}) -> str:
    # TODO: implement MQTT payload and publish logic
    raise NotImplementedError("{func_name} is not implemented yet")
"""


# ── 公开 API ───────────────────────────────────────────────────────────────────

def add_tool(name: str, description: str, params: dict) -> None:
    """
    添加或覆盖一个工具，同步更新 REGISTRY.json、{name}.py、__init__.py。

    Args:
        name: 工具名称（对应 execute_{name} 函数，以及 {name}.py 文件）
        description: 工具描述（供 LLM 理解）
        params: 参数字典，格式：
            {
                "param_name": {
                    "type": "int|float|str|bool",
                    "description": "...",
                    "required": True,
                    "default": ...  # 可选
                }
            }
    """
    # 1. REGISTRY.json
    registry = _load_registry()
    registry[name] = {"name": name, "description": description, "params": params}
    _save_registry(registry)

    # 2. {name}.py（仅在文件不存在时生成，避免覆盖已有实现）
    py_path = os.path.join(TOOLS_DIR, f"{name}.py")
    if not os.path.exists(py_path):
        with open(py_path, "w", encoding="utf-8") as f:
            f.write(_generate_tool_file(name, description, params))
        print(f"[Registry] 已生成工具文件: hardware/tools/{name}.py")
    else:
        print(f"[Registry] 工具文件已存在，跳过生成: hardware/tools/{name}.py")

    # 3. __init__.py
    _add_to_init(f"execute_{name}", name)

    print(f"[Registry] 已添加工具: {name}")


def remove_tool(name: str) -> None:
    """从 REGISTRY.json 和 __init__.py 中移除工具（不删除 .py 文件）。"""
    registry = _load_registry()
    if name not in registry:
        print(f"[Registry] 工具不存在: {name}")
        return
    del registry[name]
    _save_registry(registry)
    _remove_from_init(f"execute_{name}", name)
    print(f"[Registry] 已移除工具: {name}（.py 文件保留，请手动删除）")


def update_tool_description(name: str, description: str) -> None:
    """仅更新 REGISTRY.json 中的描述字段。"""
    registry = _load_registry()
    if name not in registry:
        print(f"[Registry] 工具不存在: {name}")
        return
    registry[name]["description"] = description
    _save_registry(registry)
    print(f"[Registry] 已更新描述: {name}")


def update_tool_param(name: str, param_name: str, param_def: dict) -> None:
    """添加或更新 REGISTRY.json 中工具的某个参数。"""
    registry = _load_registry()
    if name not in registry:
        print(f"[Registry] 工具不存在: {name}")
        return
    registry[name]["params"][param_name] = param_def
    _save_registry(registry)
    print(f"[Registry] 已更新参数 {name}.{param_name}")


def list_tools() -> list:
    """返回所有已注册工具的名称列表。"""
    return list(_load_registry().keys())


def scan_and_sync() -> None:
    """
    扫描 hardware/tools/ 下所有 .py 文件，提取 @register_tool 装饰器信息，
    同步到 REGISTRY.json 和 __init__.py。已存在的条目会被覆盖。
    跳过 registry.py 和 __init__.py 本身。
    """
    import ast

    SKIP = {"registry.py", "__init__.py"}
    registry = _load_registry()
    found = {}

    for fname in os.listdir(TOOLS_DIR):
        if not fname.endswith(".py") or fname in SKIP:
            continue
        fpath = os.path.join(TOOLS_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            src = f.read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            print(f"[Registry] 跳过（语法错误）: {fname}")
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for deco in node.decorator_list:
                # 匹配 @register_tool(name=..., description=..., params=...)
                if not (isinstance(deco, ast.Call) and
                        getattr(getattr(deco, "func", None), "id", None) == "register_tool"):
                    continue
                kwargs = {kw.keyword: kw for kw in [
                    type("kw", (), {"keyword": k.arg, "value": k.value})()
                    for k in deco.keywords
                ]}
                name_node = kwargs.get("name")
                desc_node = kwargs.get("description")
                params_node = kwargs.get("params")
                if not (name_node and desc_node and params_node):
                    continue
                try:
                    tool_name = ast.literal_eval(name_node.value)
                    tool_desc = ast.literal_eval(desc_node.value)
                    tool_params = ast.literal_eval(params_node.value)
                except Exception:
                    print(f"[Registry] 跳过（无法解析装饰器参数）: {fname}:{node.name}")
                    continue

                module_name = fname[:-3]
                func_name = node.name
                found[tool_name] = {
                    "name": tool_name,
                    "description": tool_desc,
                    "params": tool_params,
                    "_module": module_name,
                    "_func": func_name,
                }

    # 同步 REGISTRY.json
    for tool_name, info in found.items():
        registry[tool_name] = {
            "name": info["name"],
            "description": info["description"],
            "params": info["params"],
        }
    _save_registry(registry)

    # 同步 __init__.py
    for tool_name, info in found.items():
        _add_to_init(info["_func"], info["_module"])

    print(f"[Registry] 扫描完成，共发现 {len(found)} 个工具: {list(found.keys())}")


if __name__ == "__main__":
    scan_and_sync()
