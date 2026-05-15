"""
硬件控制模块 - 基于 LLM 命令解析的同步工具调用（"硬件控制：" 前缀触发）
"""

import json
import re
import threading
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field

from .config import Config
from .llm_client import LLMClient
from hardware import ToolRegistry
from hardware.utils.reagent import find_reagent

# 配置日志
logger = logging.getLogger(__name__)


class HardwareTool(BaseModel):
    """
    硬件工具元数据定义

    Attributes:
        name        : 工具名称，如 "do_experiment"
        description : 工具描述，注入 LLM prompt 帮助 AI 选择工具
        params      : 参数定义，每个参数含 type/description/required/default
        function    : 对应 hardware/tools.py 中底层函数名称
    """
    name: str = Field(description="硬件工具名称")
    description: str = Field(description="硬件工具描述")
    params: Dict[str, Dict[str, Any]] = Field(description="参数定义")
    function: str = Field(description="对应的底层函数名称")


class HardwareAgent:
    """
    硬件智能体 - 通过 LLM 解析自然语言命令，映射到已注册工具并执行

    工作流：用户命令 -> LLM 解析为 JSON -> 参数验证 -> 调用底层函数
    """

    def __init__(self):
        self.config = Config()
        self.llm_client = LLMClient()
        self.hardware_tools = self._load_hardware_tools()

    def _load_hardware_tools(self) -> List[HardwareTool]:
        """
        从 ToolRegistry 自动加载所有已注册的硬件工具定义

        新工具只需在 hardware/tools/ 中放一个带 @register_tool 装饰器的 .py 文件，
        启动时自动发现并加载，无需修改此处代码。
        """
        tools = []
        for name, entry in ToolRegistry.get_all().items():
            tools.append(HardwareTool(
                name=name,
                description=entry["description"],
                params=entry["params"],
                function=f"execute_{name}",
            ))
        # LLM-facing aliases（同一底层 spin_coating 函数的不同入口名称）
        spin_entry = ToolRegistry.get_tool("spin_coating")
        if spin_entry:
            spin_params = {k: dict(v) for k, v in spin_entry["params"].items()}
            tools.append(HardwareTool(
                name="do_experiment",
                description="执行旋涂实验（单步）",
                params=spin_params,
                function="execute_spin_coating",
            ))
            tools.append(HardwareTool(
                name="save_experiment_step",
                description="注册一步旋涂实验参数（多步实验用）",
                params=spin_params,
                function="execute_spin_coating",
            ))
        return tools

    def get_tools_schema(self) -> str:
        """将工具列表转为 JSON 字符串，注入到 LLM prompt 中"""
        tools_info = []
        for tool in self.hardware_tools:
            params_info = {}
            for param_name, param_info in tool.params.items():
                params_info[param_name] = {
                    "type": param_info.get("type"),
                    "description": param_info.get("description"),
                    "required": param_info.get("required", False),
                }
                if param_info.get("default") is not None:
                    params_info[param_name]["default"] = param_info["default"]
            tools_info.append({
                "name": tool.name,
                "description": tool.description,
                "params": params_info,
            })
        return json.dumps(tools_info, ensure_ascii=False)

    def parse_complex_command(self, command_text: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        用 LLM 解析自然语言命令为工具调用 JSON 列表

        Args:
            command_text: 用户自然语言命令

        Returns:
            (是否成功, 工具调用列表 [{"name": "xxx", "params": {...}}, ...])
        """
        tools_schema = self.get_tools_schema()
        # 已迁移至 prompts/hardware/_command_parse.yaml
        from prompts import create_prompt_manager
        pm = create_prompt_manager()
        prompt = pm.get(
            "hardware_command_parse",
            tools_schema=tools_schema,
            user_command=command_text,
        )
        messages = [{"role": "user", "content": prompt}]
        result = self.llm_client.call_api(
            model=self.config.MODEL_NAME_TALK,
            messages=messages,
            temperature=0.1,
            max_tokens=None,
            response_format={"type": "json_object"},
        )
        if not result:
            return False, []

        try:
            content = result['choices'][0]['message']['content'].strip()
            clean_json = re.sub(r'```json\n|\n```|```', '', content).strip()
            tool_calls = json.loads(clean_json)
            if isinstance(tool_calls, list):
                return True, tool_calls
            return False, []
        except Exception as e:
            print(f"解析复杂命令失败: {e}")
            return False, []

    def execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行单个工具调用，通过 ToolRegistry 动态分派到硬件函数

        Args:
            tool_call: {"name": "工具名", "params": {参数}}

        Returns:
            {"status": "success"/"error", "result"/"message": ...}
        """
        try:
            tool_name = tool_call.get("name")
            params = dict(tool_call.get("params", {}))

            # Alias map: LLM-facing names -> registry names
            ALIASES = {"do_experiment": "spin_coating", "save_experiment_step": "spin_coating"}
            resolved = ALIASES.get(tool_name, tool_name)

            # Param rename: plan JSON keys -> function arg names
            PARAM_RENAME = {"set_temperature": {"temperature": "target"}}

            logger.info(f"[硬件调用] 工具: {tool_name}, 参数: {params}")
            print(f"[硬件调用] 工具: {tool_name}, 参数: {params}")

            entry = ToolRegistry.get_tool(resolved)
            if not entry:
                logger.warning(f"[硬件调用] 未知工具: {tool_name}")
                return {"status": "error", "message": f"未知工具: {tool_name}"}

            # Apply param renames
            rename = PARAM_RENAME.get(resolved, {})
            for old, new in rename.items():
                if old in params and new not in params:
                    params[new] = params.pop(old)

            # Build kwargs using registry param order
            kwargs = {}
            for pname, pinfo in entry["params"].items():
                if pname in params:
                    kwargs[pname] = params[pname]
                elif "default" in pinfo:
                    kwargs[pname] = pinfo["default"]

            result = entry["function"](**kwargs)
            logger.info(f"[硬件调用] 工具 {tool_name} 执行完成")
            return {"status": "success", "result": result}

        except Exception as e:
            return {"status": "error", "message": f"执行工具调用失败: {str(e)}"}

    def is_known_tool(self, name: str) -> bool:
        """检查工具名称是否已注册（含别名映射）"""
        ALIASES = {"do_experiment": "spin_coating", "save_experiment_step": "spin_coating"}
        resolved = ALIASES.get(name, name)
        return ToolRegistry.get_tool(resolved) is not None

    def check_reagent(self, name: str) -> str:
        """检查试剂是否存在，返回试剂位置"""
        return find_reagent(name)

    def execute_complex_command(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """依次执行多个工具调用，返回每个调用的结果列表"""
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            params = tool_call.get("params", {})
            valid, error_msg = self.validate_tool_params(tool_name, params)
            if not valid:
                results.append({"tool": tool_name, "status": "error", "message": error_msg})
                continue
            result = self.execute_tool_call(tool_call)
            results.append({"tool": tool_name, "status": result.get("status"), "result": result})
        return results

    def validate_tool_params(self, tool_name: str, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        验证工具参数（类型检查、必填检查、转速上限等）

        Returns:
            (是否合法, 错误信息)
        """
        tool = next((t for t in self.hardware_tools if t.name == tool_name), None)
        if not tool:
            return False, f"未知工具: {tool_name}"

        for param_name, param_info in tool.params.items():
            if param_info.get("required", False) and param_name not in params:
                return False, f"缺少必需参数: {param_name}"

        for param_name, param_value in params.items():
            if param_name in tool.params:
                expected_type = tool.params[param_name].get("type")
                try:
                    if expected_type == "int":
                        params[param_name] = int(param_value)
                    elif expected_type == "float":
                        params[param_name] = float(param_value)
                    elif expected_type == "str":
                        params[param_name] = str(param_value)
                except (ValueError, TypeError):
                    return False, f"参数 {param_name} 类型错误，应为 {expected_type}"

        if tool_name in ("do_experiment", "save_experiment_step"):
            if params.get("spin_speed", 0) > 6000:
                return False, "转速不能超过6000rpm"

        return True, ""


class HardwareController:
    """
    硬件控制器 - HardwareAgent 的高层封装

    提供防重复提交、实验确认信息生成、硬件状态查询等功能。
    由 app.py 的 "硬件控制：" 前缀命令触发。
    """

    def __init__(self):
        self.agent = HardwareAgent()
        self.config = Config()
        self.is_running = False
        self._state_lock = threading.Lock()
        self._last_command_signature: Optional[str] = None
        self._last_command_time = 0.0
        self._duplicate_window_seconds = 2.0

    def is_hardware_running(self) -> bool:
        """硬件是否正在执行任务"""
        return self.is_running

    def control_hardware(self, user_command: str) -> tuple[bool, Dict[str, Any]]:
        """
        处理用户硬件控制命令（主入口）

        流程：LLM 解析命令 -> 参数验证 -> 执行

        Args:
            user_command: 自然语言硬件命令

        Returns:
            (是否成功, 执行结果)
        """
        success, tool_calls = self.agent.parse_complex_command(user_command)
        if not success:
            return False, {"status": "error", "message": "命令解析失败，请检查指令格式"}
        return self.execute_tool_calls(tool_calls)

    def execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> tuple[bool, Dict[str, Any]]:
        """执行工具调用列表（带防重复提交保护）"""
        command_signature = json.dumps(tool_calls, sort_keys=True, ensure_ascii=False)
        now = time.monotonic()

        with self._state_lock:
            if self.is_running:
                return False, {"status": "rejected", "message": "硬件任务正在执行，请勿重复提交。"}
            if (
                self._last_command_signature == command_signature
                and now - self._last_command_time < self._duplicate_window_seconds
            ):
                return False, {"status": "rejected", "message": "检测到重复的硬件指令，已拦截。"}
            self.is_running = True
            self._last_command_signature = command_signature
            self._last_command_time = now

        try:
            if len(tool_calls) == 1:
                result = self.agent.execute_tool_call(tool_calls[0])
                return result.get("status") == "success", result
            else:
                results = self.agent.execute_complex_command(tool_calls)
                all_success = all(r.get("status") == "success" for r in results)
                return all_success, {
                    "status": "success" if all_success else "partial_error",
                    "results": results,
                    "message": "复杂命令执行完成",
                }
        finally:
            with self._state_lock:
                self.is_running = False

    def get_hardware_status(self) -> Dict[str, Any]:
        """获取硬件连接状态和可用工具列表"""
        try:
            from hardware import local_client
            return {
                "status": "connected" if local_client.is_connected else "disconnected",
                "available_tools": len(self.agent.hardware_tools),
                "tools": [tool.name for tool in self.agent.hardware_tools],
            }
        except Exception as e:
            return {"status": "error", "message": f"获取硬件状态失败: {str(e)}"}

    def ask_for_experiment_confirmation(self, tool_calls: List[Dict[str, Any]]) -> str:
        """根据工具调用列表生成人类可读的实验确认信息"""
        if not tool_calls:
            return "没有需要确认的实验操作"

        parts = []
        for tc in tool_calls:
            name = tc.get("name")
            p = tc.get("params", {})
            if name in ("do_experiment", "save_experiment_step"):
                parts.append(
                    f"  - 旋涂实验：试剂={p.get('reagent','未知')}, "
                    f"转速={p.get('spin_speed','未知')}rpm, "
                    f"时长={p.get('spin_dur','未知')}ms, "
                    f"体积={p.get('volume','未知')}ul"
                )
            elif name == "set_temperature":
                parts.append(f"  - 设置温度：{p.get('target','未知')} ℃")
            elif name == "move_robot_arm":
                parts.append(f"  - 移动机械臂：X={p.get('x')}, Y={p.get('y')}, Z={p.get('z')}")
            elif name == "start_experiment":
                parts.append("  - 启动已注册的实验序列")
            elif name == "collect_spectrum":
                parts.append(f"  - 启动光谱仪采集（{p.get('duration',60)}秒）")

        return "检测到以下硬件操作，请确认是否继续：\n" + "\n".join(parts) if parts else "检测到硬件操作，请确认是否继续"

    def supports_complex_operations(self) -> bool:
        """是否支持复杂操作（多工具调用）"""
        return True

    def get_available_hardware(self) -> List[str]:
        """返回所有已注册的硬件工具名称"""
        return [tool.name for tool in self.agent.hardware_tools]
