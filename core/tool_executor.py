"""
工具调用执行器 — 对接 LLM tool_call 到实际硬件/软件功能。

架构：
  LLM 返回 tool_calls → ToolExecutor.dispatch() → 查找 TOOL_REGISTRY → 调用 → 返回结果字符串

TOOL_REGISTRY 格式（OpenAI 兼容）：
  {"name": "get_weather", "function": get_weather_impl, "schema": {...}}

TODO: 从 hardware/tools/REGISTRY.json 和 software/algorithms 自动导入工具
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List

# ---- Tool 定义 ----
ToolFunction = Callable[..., str]
ToolSchema = Dict[str, Any]   # OpenAI-format function schema


@dataclass
class ToolDef:
    """单个工具定义"""
    name: str
    description: str
    parameters: dict          # JSON Schema for parameters
    func: ToolFunction        # 实际执行函数


# ---- 全局工具注册表 ----
# TODO: 从 hardware/tools/REGISTRY.json 导入硬件工具
# TODO: 从 software/algorithms 导入软件算法
TOOL_REGISTRY: Dict[str, ToolDef] = {}


def register_tool(name: str, description: str, parameters: dict, func: ToolFunction):
    """注册一个 tool 到全局注册表"""
    TOOL_REGISTRY[name] = ToolDef(
        name=name,
        description=description,
        parameters=parameters,
        func=func,
    )


def build_openai_tools() -> List[dict]:
    """将 TOOL_REGISTRY 转为 OpenAI tools 参数格式"""
    tools = []
    for td in TOOL_REGISTRY.values():
        tools.append({
            "type": "function",
            "function": {
                "name": td.name,
                "description": td.description,
                "parameters": td.parameters,
            }
        })
    return tools


class ToolExecutor:
    """工具执行器 — 根据 tool_name 查找并执行"""

    def dispatch(self, tool_name: str, arguments: dict) -> str:
        """
        执行单个 tool，返回结果字符串

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            执行结果字符串
        """
        td = TOOL_REGISTRY.get(tool_name)
        if td is None:
            return f"错误: 未找到工具 '{tool_name}'"
        try:
            return str(td.func(**arguments))
        except Exception as e:
            return f"工具 '{tool_name}' 执行错误: {str(e)}"

    def dispatch_all(self, tool_calls: list) -> list:
        """
        批量执行 tool_calls，返回 tool result messages

        Args:
            tool_calls: OpenAI 格式的 tool_calls 列表

        Returns:
            [{"role": "tool", "tool_call_id": ..., "content": ...}, ...]
        """
        results = []
        for tc in tool_calls:
            func_name = tc['function']['name']
            try:
                import json
                arguments = json.loads(tc['function']['arguments'])
            except Exception:
                arguments = {}
            result_content = self.dispatch(func_name, arguments)
            results.append({
                "role": "tool",
                "tool_call_id": tc['id'],
                "content": result_content,
            })
        return results
