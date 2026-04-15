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
        加载所有已注册的硬件工具定义

        扩展方法：
        1. 在 hardware/tools.py 中编写底层执行函数
        2. 在此方法中添加 HardwareTool 定义
        3. 在 execute_tool_call() 中添加分派逻辑
        """
        return [
            HardwareTool(
                name="set_temperature",
                description="设置设备温度",
                params={
                    "target": {"type": "float", "description": "目标温度值（℃）", "required": True, "default": None},
                },
                function="execute_set_temperature",
            ),
            HardwareTool(
                name="move_robot_arm",
                description="移动机械臂到指定位置",
                params={
                    "x": {"type": "float", "description": "X坐标", "required": True, "default": None},
                    "y": {"type": "float", "description": "Y坐标", "required": True, "default": None},
                    "z": {"type": "float", "description": "Z坐标", "required": True, "default": None},
                },
                function="execute_move_robot_arm",
            ),
            HardwareTool(
                name="do_experiment",
                description="执行旋涂实验",
                params={
                    "reagent":    {"type": "str", "description": "试剂名称", "required": True, "default": ""},
                    "spin_speed": {"type": "int", "description": "转速(rpm)，最大6000", "required": True, "default": 3000},
                    "spin_acc":   {"type": "int", "description": "加速度(rpm/s)", "required": False, "default": 1000},
                    "spin_dur":   {"type": "int", "description": "持续时间(ms)", "required": True, "default": 30000},
                    "volume":     {"type": "int", "description": "体积(µl)", "required": False, "default": 10},
                },
                function="execute_spin_coating",
            ),
            # ---- 合并 AutonomousPlatform 后新增 ----
            HardwareTool(
                name="save_experiment_step",
                description="注册一步旋涂实验参数（多步实验需多次调用，最后用 start_experiment 启动）",
                params={
                    "reagent":    {"type": "str", "description": "试剂名称", "required": True, "default": ""},
                    "spin_speed": {"type": "int", "description": "转速(rpm)，最大6000", "required": True, "default": 3000},
                    "spin_acc":   {"type": "int", "description": "加速度(rpm/s)", "required": False, "default": 1000},
                    "spin_dur":   {"type": "int", "description": "持续时间(ms)", "required": True, "default": 30000},
                    "volume":     {"type": "int", "description": "体积(µl)", "required": False, "default": 10},
                },
                function="execute_spin_coating",
            ),
            HardwareTool(
                name="start_experiment",
                description="启动已注册的多步实验序列",
                params={},
                function="execute_start_experiment",
            ),
            HardwareTool(
                name="collect_spectrum",
                description="启动光谱仪数据采集",
                params={
                    "duration": {"type": "int", "description": "采集时长(秒)", "required": False, "default": 60},
                },
                function="execute_collect_spectrum",
            ),
        ]

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
        prompt = f"""
你是一个智能实验室助手，需要根据用户指令调用相应的硬件工具。

可用工具：
{tools_schema}

用户指令：{command_text}

请分析用户指令，确定需要调用哪些工具以及它们的参数。
输出格式：JSON数组，每个元素是一个工具调用，包含name和params。

示例：
- 简单调用: [{{"name": "set_temperature", "params": {{"target": 25.0}}}}]
- 多工具调用: [{{"name": "set_temperature", "params": {{"target": 25.0}}}}, {{"name": "move_robot_arm", "params": {{"x": 10.0, "y": 20.0, "z": 30.0}}}}]
"""
        messages = [{"role": "user", "content": prompt}]
        result = self.llm_client.call_api(
            model=self.config.MODEL_NAME_TALK,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
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
        执行单个工具调用，分派到 hardware/tools.py 底层函数

        Args:
            tool_call: {"name": "工具名", "params": {参数}}

        Returns:
            {"status": "success"/"error", "result"/"message": ...}
        """
        try:
            from hardware.tools import (
                execute_spin_coating,
                execute_set_temperature,
                execute_move_robot_arm,
                execute_start_experiment,
                execute_collect_spectrum,
            )

            tool_name = tool_call.get("name")
            params = tool_call.get("params", {})

            # 记录硬件函数调用
            logger.info(f"[硬件调用] 工具: {tool_name}, 参数: {params}")
            print(f"[硬件调用] 工具: {tool_name}, 参数: {params}")

            if tool_name == "set_temperature":
                result = execute_set_temperature(float(params["target"]))
            elif tool_name == "move_robot_arm":
                result = execute_move_robot_arm(
                    float(params["x"]), float(params["y"]), float(params["z"]),
                )
            elif tool_name in ("do_experiment", "save_experiment_step"):
                result = execute_spin_coating(
                    int(params.get("spin_speed", 3000)),
                    int(params.get("spin_acc", 1000)),
                    int(params.get("spin_dur", 30000)),
                    str(params.get("reagent", "")),
                    int(params.get("volume", 10)),
                )
            elif tool_name == "start_experiment":
                result = execute_start_experiment()
            elif tool_name == "collect_spectrum":
                result = execute_collect_spectrum(int(params.get("duration", 60)))
            else:
                logger.warning(f"[硬件调用] 未知工具: {tool_name}")
                return {"status": "error", "message": f"未知工具: {tool_name}"}

            logger.info(f"[硬件调用] 工具 {tool_name} 执行完成")
            return {"status": "success", "result": result}

        except ImportError as e:
            return {"status": "error", "message": f"硬件工具模块导入失败: {str(e)}"}
        except Exception as e:
            return {"status": "error", "message": f"执行工具调用失败: {str(e)}"}

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
            from hardware.tools import local_client
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
