"""
核心 - 统一工具执行器 (core/agent_tools.py)
===========================================

UnifiedToolExecutor 合并 hardware/tools/, software/algorithms/ 和内置
agent 工具到统一的 OpenAI tools 格式，供 AgentLoop 使用。

组件:
    AgentTool dataclass        — 统一工具描述（含 func / category / dangerous）
    BUILTIN_TOOLS              — ask_user（AgentLoop 会在 dispatch 时拦截）
    scan_hardware_tools()      — 从 hardware.ToolRegistry 导入硬件工具
    scan_software_algorithms() — 从 SoftwareController.list_algorithms() 导入软件算法
    UnifiedToolExecutor        — dispatch / build_openai_tools / is_hardware_tool / get / names
    create_main_executor()     — 工厂函数，合并全部三类工具

使用示例::

    from core.agent_tools import create_main_executor

    executor = create_main_executor()
    openai_tools = executor.build_openai_tools()
    result = executor.dispatch("spin_coating", {"spin_speed": 3000, ...})
"""

import json
from dataclasses import dataclass
from typing import Callable


# =============================================================================
# AgentTool 数据类
# =============================================================================

@dataclass
class AgentTool:
    """统一工具描述，包含 OpenAI JSON Schema 和执行函数引用"""
    name: str
    description: str
    parameters: dict              # OpenAI JSON Schema object
    required: list[str]           # 必填参数名
    func: Callable[[dict], str]   # 接收 args dict，返回结果字符串
    category: str                 # "builtin" | "hardware" | "software"
    dangerous: bool = False       # 是否为危险操作（硬件工具默认 True）


# =============================================================================
# 参数 schema 转换工具
# =============================================================================

_TYPE_MAP = {
    "int":   "integer",
    "float": "number",
    "str":   "string",
    "bool":  "boolean",
    "list":  "array",
    "dict":  "object",
}


def _param_to_json_schema(param_def: dict) -> dict:
    """将单个参数定义转为 JSON Schema property"""
    ptype = param_def.get("type", "string")
    schema_type = _TYPE_MAP.get(ptype, "string")
    prop = {"type": schema_type, "description": param_def.get("description", "")}
    if "default" in param_def:
        prop["default"] = param_def["default"]
    if ptype == "list":
        prop["items"] = {"type": "string"}
    return prop


def _params_to_json_schema(params: dict) -> dict:
    """将 Registry 格式的 params dict 转为 OpenAI JSON Schema"""
    properties = {}
    required_list = []
    for name, param_def in params.items():
        properties[name] = _param_to_json_schema(param_def)
        if param_def.get("required"):
            required_list.append(name)

    schema: dict = {"type": "object", "properties": properties}
    if required_list:
        schema["required"] = required_list
    return schema


# =============================================================================
# BUILTIN_TOOLS --- 内置 agent 工具
# =============================================================================

def _ask_user_func(args: dict) -> str:
    """No-op: AgentLoop 在 dispatch 时检测此返回值并拦截，暂停等待用户输入"""
    return "__ASK_USER_PENDING__"


BUILTIN_TOOLS: list[AgentTool] = [
    AgentTool(
        name="ask_user",
        description=(
            "向用户提问以澄清意图、确认危险操作或在多个策略中选择。"
            "当指令不够明确或存在多种可行方案时，应使用此工具请求用户确认。"
            "注意：不要使用此工具询问已由其他工具参数明确指定的范围或取值。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "向用户提出的问题，用于澄清意图或确认操作",
                },
                "options": {
                    "type": "string",
                    "description": "可选的 JSON 数组字符串，列出供用户选择的选项",
                },
            },
            "required": ["question"],
        },
        required=["question"],
        func=_ask_user_func,
        category="builtin",
        dangerous=False,
    ),
]


# =============================================================================
# 硬件工具扫描与分发
# =============================================================================

def _dispatch_hardware(name: str, args: dict) -> str:
    """
    分发硬件工具调用

    从 hardware.ToolRegistry 查找工具定义，按 registry params 顺序构建
    kwargs（对缺失的 optional 参数填入 default），调用实际函数。

    Args:
        name: 工具名称
        args: LLM 传入的参数 dict

    Returns:
        硬件工具执行结果字符串
    """
    from hardware import ToolRegistry

    entry = ToolRegistry.get_tool(name)
    if entry is None:
        return f"错误: 未找到硬件工具 '{name}'"

    params_def = entry.get("params", {})
    kwargs = {}

    for param_name, param_def in params_def.items():
        if param_name in args:
            kwargs[param_name] = args[param_name]
        elif not param_def.get("required", False):
            # 可选参数：有 default 就填 default
            if "default" in param_def:
                kwargs[param_name] = param_def["default"]
        # required 但未提供：不填 kwargs，让函数自身处理（触发 TypeError）

    try:
        func = entry["function"]
        result = func(**kwargs)
        return str(result)
    except Exception as e:
        return f"硬件工具 '{name}' 执行错误: {str(e)}"


def scan_hardware_tools() -> list[AgentTool]:
    """
    扫描 hardware.ToolRegistry 中所有已注册工具，转为 AgentTool 列表

    注意: hardware.__init__.py 在 import 时已调用 discover_tools()，
    因此导入时 ToolRegistry 已填充完毕。

    Returns:
        AgentTool 列表，每个工具的 category="hardware", dangerous=True
    """
    from hardware import ToolRegistry

    tools: list[AgentTool] = []
    entries = ToolRegistry.get_all()

    for name, entry in entries.items():
        params_def = entry.get("params", {})
        schema = _params_to_json_schema(params_def)
        required_list = schema.get("required", [])

        # 闭包捕获 tool_name，避免循环变量延迟绑定问题
        def _make_func(tool_name: str):
            def _func(args: dict) -> str:
                return _dispatch_hardware(tool_name, args)
            return _func

        tool = AgentTool(
            name=name,
            description=entry.get("description", ""),
            parameters=schema,
            required=required_list,
            func=_make_func(name),
            category="hardware",
            dangerous=True,
        )
        tools.append(tool)

    return tools


# =============================================================================
# 软件算法扫描与分发
# =============================================================================

def _dispatch_software(name: str, args: dict) -> str:
    """
    分发软件算法调用

    创建 SoftwareController 实例，将 args 中的 data 字段提取为算法输入，
    其余字段作为算法参数传入。

    Args:
        name: 算法名称
        args: LLM 传入的参数 dict（包含 data 和算法参数）

    Returns:
        JSON 字符串，格式: {"success": bool, "algorithm": str, "result": ..., "message": str}
    """
    from software.software_controller import SoftwareController

    controller = SoftwareController()
    args_copy = dict(args)  # 不修改调用方的 dict
    data = args_copy.pop("data", None)
    params = args_copy if args_copy else None

    try:
        result = controller.run_algorithm(name, data=data, params=params)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "success": False,
            "algorithm": name,
            "result": None,
            "message": f"算法 '{name}' 执行异常: {str(e)}",
        }, ensure_ascii=False)


def scan_software_algorithms() -> list[AgentTool]:
    """
    扫描 SoftwareController 中所有已注册算法，转为 AgentTool 列表

    自动向每个算法的 OpenAI schema 中添加 "data" 必填字段
    （算法执行必需的输入数据）。

    Returns:
        AgentTool 列表，每个工具的 category="software", dangerous=False
    """
    from software.software_controller import SoftwareController

    controller = SoftwareController()
    algos = controller.list_algorithms()

    tools: list[AgentTool] = []
    for algo in algos:
        params_schema = algo.get("params_schema", {})
        schema = _params_to_json_schema(params_schema)

        # 软件算法需要输入数据 —— 向 schema 中添加 data 字段
        schema.setdefault("properties", {})
        schema["properties"]["data"] = {
            "type": "object",
            "description": "算法的输入数据（dict / list / 由具体算法定义格式）",
        }
        required_list = schema.get("required", [])
        required_list.append("data")
        schema["required"] = required_list

        name = algo["name"]

        def _make_func(tool_name: str):
            def _func(args: dict) -> str:
                return _dispatch_software(tool_name, args)
            return _func

        tool = AgentTool(
            name=name,
            description=algo.get("description", ""),
            parameters=schema,
            required=required_list,
            func=_make_func(name),
            category="software",
            dangerous=False,
        )
        tools.append(tool)

    return tools


# =============================================================================
# UnifiedToolExecutor --- 统一工具执行器
# =============================================================================

class UnifiedToolExecutor:
    """
    统一工具执行器

    合并内置工具 / 硬件工具 / 软件算法，提供统一的 dispatch + 查询接口。
    AgentLoop 通过此对象与所有工具交互。

    使用示例::

        from core.agent_tools import create_main_executor
        exec = create_main_executor()

        # 构建 OpenAI tools 参数
        openai_tools = exec.build_openai_tools()

        # LLM 返回 tool_call 后分发执行
        result = exec.dispatch("spin_coating", {"spin_speed": 3000, ...})

        # 查询
        exec.is_hardware_tool("drop")  # → True
        exec.get("data_statistics")     # → AgentTool | None
        exec.names                      # → ["ask_user", "drop", ...]
    """

    def __init__(self, tools: list[AgentTool]):
        self._tools: dict[str, AgentTool] = {t.name: t for t in tools}

    @property
    def names(self) -> list[str]:
        """所有已注册工具的名称列表"""
        return list(self._tools.keys())

    def build_openai_tools(self) -> list[dict]:
        """
        构建 OpenAI tools 参数格式

        Returns:
            [{"type":"function","function":{"name":...,"description":...,"parameters":...}}, ...]
        """
        result: list[dict] = []
        for tool in self._tools.values():
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        return result

    def dispatch(self, name: str, arguments: dict) -> str:
        """
        按名称查找工具并执行

        Args:
            name: 工具名称
            arguments: 工具参数 dict

        Returns:
            工具执行结果字符串
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"错误: 未找到工具 '{name}'"
        try:
            return tool.func(arguments)
        except Exception as e:
            return f"工具 '{name}' 执行错误: {str(e)}"

    def is_hardware_tool(self, name: str) -> bool:
        """判断给定名称是否对应一个硬件工具"""
        tool = self._tools.get(name)
        return tool is not None and tool.category == "hardware"

    def get(self, name: str) -> AgentTool | None:
        """按名称查找 AgentTool，未找到返回 None"""
        return self._tools.get(name)


# =============================================================================
# Factory
# =============================================================================

def create_main_executor() -> UnifiedToolExecutor:
    """
    工厂函数：扫描硬件工具 + 软件算法 + 合并内置工具，创建 UnifiedToolExecutor

    启动时打印工具扫描摘要:
        [AgentTools] Found N hardware tools: [...]
        [AgentTools] Found N software algorithms: [...]
        [AgentTools] Total: N tools

    Returns:
        已填充所有工具的 UnifiedToolExecutor 实例
    """
    hw_tools = scan_hardware_tools()
    hw_names = [t.name for t in hw_tools]
    print(f"[AgentTools] Found {len(hw_tools)} hardware tools: {hw_names}")

    sw_tools = scan_software_algorithms()
    sw_names = [t.name for t in sw_tools]
    print(f"[AgentTools] Found {len(sw_tools)} software algorithms: {sw_names}")

    all_tools = list(BUILTIN_TOOLS) + hw_tools + sw_tools
    print(f"[AgentTools] Total: {len(all_tools)} tools")

    return UnifiedToolExecutor(all_tools)
