"""
硬件控制模块
负责与硬件设备交互，执行硬件控制命令，支持智能硬件选择和复杂调用
"""

import json
import re
import threading
import time
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field, ValidationError, create_model
from typing import Literal

from .config import Config
from .llm_client import LLMClient


class HardwareTool(BaseModel):
    """
    硬件工具定义
    """
    name: str = Field(description="硬件工具名称")
    description: str = Field(description="硬件工具描述")
    params: Dict[str, Dict[str, Any]] = Field(description="参数定义")
    function: str = Field(description="对应的底层函数名称")


class HardwareAgent:
    """
    硬件智能体 - 负责智能硬件选择和调用
    """

    def __init__(self):
        """初始化硬件智能体"""
        self.config = Config()
        self.llm_client = LLMClient()
        self.hardware_tools = self._load_hardware_tools()

    def _load_hardware_tools(self) -> List[HardwareTool]:
        """
        加载硬件工具列表，对应root/hardware/tools.py中的函数

        Returns:
            硬件工具列表
        """
        return [
            HardwareTool(
                name="set_temperature",
                description="设置设备温度",
                params={
                    "target": {
                        "type": "float",
                        "description": "目标温度值",
                        "required": True,
                        "default": None
                    }
                },
                function="execute_set_temperature"
            ),
            HardwareTool(
                name="move_robot_arm",
                description="移动机械臂到指定位置",
                params={
                    "x": {
                        "type": "float",
                        "description": "X坐标",
                        "required": True,
                        "default": None
                    },
                    "y": {
                        "type": "float",
                        "description": "Y坐标",
                        "required": True,
                        "default": None
                    },
                    "z": {
                        "type": "float",
                        "description": "Z坐标",
                        "required": True,
                        "default": None
                    }
                },
                function="execute_move_robot_arm"
            ),
            HardwareTool(
                name="do_experiment",
                description="执行旋涂实验",
                params={
                    "reagent": {
                        "type": "str",
                        "description": "试剂名称",
                        "required": True,
                        "default": ""
                    },
                    "spin_speed": {
                        "type": "int",
                        "description": "转速(rpm)，最大6000rpm",
                        "required": True,
                        "default": 3000
                    },
                    "spin_acc": {
                        "type": "int",
                        "description": "加速度(rpm/s)",
                        "required": False,
                        "default": 1000
                    },
                    "spin_dur": {
                        "type": "int",
                        "description": "持续时间(毫秒)",
                        "required": True,
                        "default": 30000
                    },
                    "volume": {
                        "type": "int",
                        "description": "体积(ul)",
                        "required": False,
                        "default": 10
                    }
                },
                function="execute_spin_coating"
            )
        ]

    def get_tools_schema(self) -> str:
        """
        获取工具Schema供LLM使用

        Returns:
            工具Schema字符串
        """
        tools_info = []
        for tool in self.hardware_tools:
            params_info = {}
            for param_name, param_info in tool.params.items():
                params_info[param_name] = {
                    "type": param_info.get("type"),
                    "description": param_info.get("description"),
                    "required": param_info.get("required", False)
                }
                if param_info.get("default") is not None:
                    params_info[param_name]["default"] = param_info["default"]

            tools_info.append({
                "name": tool.name,
                "description": tool.description,
                "params": params_info
            })

        return json.dumps(tools_info, ensure_ascii=False)

    def parse_complex_command(self, command_text: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        解析复杂硬件命令，支持多工具调用

        Args:
            command_text: 命令文本

        Returns:
            (成功状态, 工具调用列表)
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
            response_format={"type": "json_object"}
        )

        if not result:
            return False, []

        try:
            content = result['choices'][0]['message']['content'].strip()
            clean_json = re.sub(r'```json\n|\n```|```', '', content).strip()

            # 尝试解析为工具调用列表
            tool_calls = json.loads(clean_json)

            if isinstance(tool_calls, list):
                return True, tool_calls
            else:
                return False, []

        except Exception as e:
            print(f"解析复杂命令失败: {e}")
            return False, []

    def execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行单个工具调用，调用root/hardware/tools.py中的底层函数

        Args:
            tool_call: 工具调用

        Returns:
            执行结果
        """
        try:
            # 导入root/hardware/tools.py中的底层函数
            from hardware.tools import (
                execute_spin_coating,
                execute_set_temperature,
                execute_move_robot_arm
            )

            tool_name = tool_call.get("name")
            params = tool_call.get("params", {})

            # 根据工具名称选择对应的函数
            if tool_name == "set_temperature":
                result = execute_set_temperature(float(params["target"]))
            elif tool_name == "move_robot_arm":
                result = execute_move_robot_arm(
                    float(params["x"]),
                    float(params["y"]),
                    float(params["z"])
                )
            elif tool_name == "do_experiment":
                result = execute_spin_coating(
                    int(params["spin_speed"]),
                    int(params.get("spin_acc", 1000)),
                    int(params["spin_dur"]),
                    params["reagent"],
                    int(params.get("volume", 10))
                )
            else:
                return {
                    "status": "error",
                    "message": f"未知工具: {tool_name}"
                }

            return {
                "status": "success",
                "result": result
            }

        except ImportError as e:
            return {
                "status": "error",
                "message": f"硬件工具模块导入失败: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"执行工具调用失败: {str(e)}"
            }

    def execute_complex_command(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        执行复杂命令（多工具调用）

        Args:
            tool_calls: 工具调用列表

        Returns:
            执行结果列表
        """
        results = []

        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call.get("name")
            params = tool_call.get("params", {})

            # 验证参数
            valid, error_msg = self.validate_tool_params(tool_name, params)
            if not valid:
                results.append({
                    "tool": tool_name,
                    "status": "error",
                    "message": error_msg
                })
                continue

            # 执行调用
            result = self.execute_tool_call(tool_call)
            results.append({
                "tool": tool_name,
                "status": result.get("status"),
                "result": result
            })

        return results

    def validate_tool_params(self, tool_name: str, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        验证工具参数

        Args:
            tool_name: 工具名称
            params: 参数

        Returns:
            (是否有效, 错误信息)
        """
        tool = next((t for t in self.hardware_tools if t.name == tool_name), None)
        if not tool:
            return False, f"未知工具: {tool_name}"

        # 检查必需参数
        for param_name, param_info in tool.params.items():
            if param_info.get("required", False) and param_name not in params:
                return False, f"缺少必需参数: {param_name}"

        # 验证参数类型
        for param_name, param_value in params.items():
            if param_name in tool.params:
                expected_type = tool.params[param_name].get("type")
                if expected_type == "int":
                    try:
                        params[param_name] = int(param_value)
                    except (ValueError, TypeError):
                        return False, f"参数{param_name}必须是整数"
                elif expected_type == "float":
                    try:
                        params[param_name] = float(param_value)
                    except (ValueError, TypeError):
                        return False, f"参数{param_name}必须是数字"
                elif expected_type == "str":
                    params[param_name] = str(param_value)

        # 特殊验证：do_experiment的转速限制
        if tool_name == "do_experiment":
            spin_speed = params.get("spin_speed", 0)
            if spin_speed > 6000:
                return False, "转速不能超过6000rpm"

        return True, ""


class HardwareController:
    """
    硬件控制器类 - 负责硬件设备控制

    职责：
    - 智能硬件选择和调用
    - 支持简单和复杂硬件操作
    - 参数验证和错误处理
    - 与LLM交互进行硬件控制
    """

    def __init__(self):
        """初始化硬件控制器"""
        self.agent = HardwareAgent()
        self.config = Config()
        self.is_running = False
        self._state_lock = threading.Lock()
        self._last_command_signature: Optional[str] = None
        self._last_command_time = 0.0
        self._duplicate_window_seconds = 2.0

    def is_hardware_running(self) -> bool:
        """硬件是否正在执行"""
        return self.is_running

    def control_hardware(self, user_command: str) -> tuple[bool, Dict[str, Any]]:
        """
        控制硬件设备

        Args:
            user_command: 用户命令

        Returns:
            (成功状态, 执行结果)
        """
        # 解析复杂命令
        success, tool_calls = self.agent.parse_complex_command(user_command)

        if not success:
            return False, {
                "status": "error",
                "message": "命令解析失败，请检查指令格式"
            }

        return self.execute_tool_calls(tool_calls)

    def execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> tuple[bool, Dict[str, Any]]:
        """
        执行工具调用列表

        Args:
            tool_calls: 工具调用列表

        Returns:
            (成功状态, 执行结果)
        """
        command_signature = json.dumps(tool_calls, sort_keys=True, ensure_ascii=False)
        now = time.monotonic()

        with self._state_lock:
            if self.is_running:
                return False, {
                    "status": "rejected",
                    "message": "硬件任务正在执行，请勿重复提交。"
                }

            if (
                self._last_command_signature == command_signature
                and now - self._last_command_time < self._duplicate_window_seconds
            ):
                return False, {
                    "status": "rejected",
                    "message": "检测到重复的硬件指令，已拦截本次重复发送。"
                }

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
                    "message": "复杂命令执行完成"
                }
        finally:
            with self._state_lock:
                self.is_running = False

    def get_hardware_status(self) -> Dict[str, Any]:
        """
        获取硬件状态

        Returns:
            硬件状态信息
        """
        try:
            # 导入local_client变量（注意：这是一个模块级变量）
            from hardware.tools import local_client
            return {
                "status": "connected" if local_client.is_connected else "disconnected",
                "available_tools": len(self.agent.hardware_tools),
                "tools": [tool.name for tool in self.agent.hardware_tools]
            }
        except ImportError as e:
            return {
                "status": "error",
                "message": f"硬件工具模块导入失败: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"获取硬件状态失败: {str(e)}"
            }

    def ask_for_experiment_confirmation(self, tool_calls: List[Dict[str, Any]]) -> str:
        """
        生成实验确认信息

        Args:
            tool_calls: 工具调用列表

        Returns:
            确认信息
        """
        if not tool_calls:
            return "没有需要确认的实验操作"

        confirmation_parts = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            params = tool_call.get("params", {})

            if tool_name == "do_experiment":
                reagent = params.get("reagent", "未知")
                speed = params.get("spin_speed", "未知")
                duration = params.get("spin_dur", "未知")
                volume = params.get("volume", "未知")

                confirmation_parts.append(
                    f"🔬 执行实验：试剂={reagent}，转速={speed}rpm，持续时间={duration}ms，体积={volume}ul"
                )
            elif tool_name == "set_temperature":
                temp = params.get("target", "未知")
                confirmation_parts.append(f"🌡️ 设置温度：{temp}°C")
            elif tool_name == "move_robot_arm":
                x = params.get("x", "未知")
                y = params.get("y", "未知")
                z = params.get("z", "未知")
                confirmation_parts.append(f"🦾 移动机械臂：X={x}, Y={y}, Z={z}")

        if confirmation_parts:
            return "检测到以下硬件操作，请确认是否继续：\n" + "\n".join(confirmation_parts)
        else:
            return "检测到硬件操作，请确认是否继续"

    def supports_complex_operations(self) -> bool:
        """
        是否支持复杂操作

        Returns:
            是否支持
        """
        return True

    def get_available_hardware(self) -> List[str]:
        """
        获取可用的硬件列表

        Returns:
            硬件名称列表
        """
        return [tool.name for tool in self.agent.hardware_tools]
