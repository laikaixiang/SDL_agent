"""
============================================================
 硬件工具模板 —— 复制此文件即可添加新硬件工具
============================================================

使用步骤：
  1. 复制此文件到 hardware/tools/ 目录
  2. 重命名文件（如 my_tool.py），注意文件名不能以 _ 开头
  3. 修改下方 @register_tool 中的内容
  4. 实现底层函数
  5. 运行一次 ToolRegistry.export_to_json("hardware/tools/REGISTRY.json") 同步 JSON

规则：
  - 函数名必须为 execute_xxx 格式，如 execute_my_tool
  - 注册名 (name) 不带 execute_ 前缀，如 "my_tool"
  - 参数顺序：所有 required=True 的在前，required=False 的在后
  - 文件名不能以 _ 开头（以 _ 开头的模块不会被自动扫描）
"""

from .registry import register_tool


# ============================================================================
# @register_tool 装饰器 —— 所有参数说明
# ============================================================================
# name        : 工具唯一标识（不含 execute_ 前缀），LLM 和前端用此名引用工具
# description : 工具功能的一句话描述，会注入到 LLM prompt 中
# params      : 参数字典，key 是参数名，value 是参数定义:
#     type        : "int" | "float" | "str" | "bool" —— 参数类型
#     description : 参数的中文说明
#     required    : True（必填）| False（可选）
#     default     : 默认值（仅 required=False 时需要），类型必须与 type 一致
# ============================================================================

@register_tool(
    name="example_tool",
    description="示例工具 —— 演示标准注册格式",
    params={
        # ---- 必填参数 ----
        "target_temp": {
            "type": "float",
            "description": "目标温度(℃)",
            "required": True,
        },
        "reagent": {
            "type": "str",
            "description": "试剂名称",
            "required": True,
        },
        # ---- 可选参数 ----
        "speed": {
            "type": "int",
            "description": "转速(rpm)",
            "required": False,
            "default": 3000,
        },
        "duration_ms": {
            "type": "int",
            "description": "持续时间(毫秒)",
            "required": False,
            "default": 30000,
        },
        "enable_log": {
            "type": "bool",
            "description": "是否启用日志",
            "required": False,
            "default": False,
        },
    }
)
def execute_example_tool(
    target_temp: float,
    reagent: str,
    speed: int = 3000,
    duration_ms: int = 30000,
    enable_log: bool = False,
) -> str:
    """
    示例工具 —— 演示标准实现格式

    函数签名必须与 @register_tool 的 params 完全对应（顺序、类型、默认值）。

    Args:
        target_temp : 目标温度(℃)
        reagent     : 试剂名称
        speed       : 转速(rpm)，默认3000
        duration_ms : 持续时间(毫秒)，默认30000
        enable_log  : 是否启用日志，默认False

    Returns:
        str: 执行结果消息（成功以 "✅" 或结果描述开头，失败以错误描述开头）
    """
    try:
        # ================================================================
        #  在此处写实际的硬件控制逻辑，如 MQTT 发布、subprocess 调用等
        # ================================================================
        # from ..mqtt import get_mqtt_client
        # client = get_mqtt_client()
        # client.publish("do_experiment", f"set_temp_{target_temp}")
        # ================================================================

        print(f"[example_tool] 设置温度={target_temp}℃, 试剂={reagent}, "
              f"转速={speed}rpm, 时长={duration_ms}ms, 日志={'开' if enable_log else '关'}")
        return f"✅ 示例工具执行成功: 温度={target_temp}℃, 试剂={reagent}"

    except Exception as e:
        return f"执行失败: {e}"
