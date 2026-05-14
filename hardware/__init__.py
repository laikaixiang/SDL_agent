"""
硬件通信层包 - MQTT通信、硬件操控函数、光谱仪采集、数据可视化

统一导出所有硬件相关功能，从 hardware 包顶层即可导入：
    from hardware import execute_spin_coating, get_mqtt_client, Deps
"""

# MQTT 连接器
from .agent_client import MQTTConnector, Client_Conf

# 工具注册系统（装饰器 + 自动发现 + 热加载）
from .tools.registry import ToolRegistry, register_tool, discover_tools, reload_tools

# === 自动发现 tools/ 目录下的工具模块 ===
# 添加新工具只需在 hardware/tools/ 中新建 .py 文件并加 @register_tool 装饰器，无需改此文件
discover_tools()

# 将所有已注册工具函数提升到包命名空间（同 register_tool 装饰器效果，确保顶层可导入）
for _name, _entry in ToolRegistry.get_all().items():
    _func = _entry["function"]
    globals()[_func.__name__] = _func

# MQTT 客户端和配置
from .mqtt import get_mqtt_client, local_client, EXPERIMENT_TOPIC, REAGENT_LAYOUT_PATH

# 试剂查找
from .utils.reagent import find_reagent, get_reagent

# PydanticAI 异步工具
from .pydantic_ai import (
    Deps,
    read_pdf,
    get_all_reagents,
    save_experiment_step,
    start_experiment,
    do_experiment,
)

# 兼容旧代码：保留 topic 和 json_path 变量别名
topic = EXPERIMENT_TOPIC
json_path = REAGENT_LAYOUT_PATH

__all__ = [
    # MQTT 连接器
    'MQTTConnector',
    'Client_Conf',
    # 工具注册
    'ToolRegistry',
    'register_tool',
    'discover_tools',
    'reload_tools',
    # 底层同步工具函数（自动发现，新工具在 hardware/tools/ 添加后在此补充）
    'execute_spin_coating',
    'execute_set_temperature',
    'execute_move_robot_arm',
    'execute_start_experiment',
    'execute_collect_spectrum',
    'get_tips',
    # MQTT 客户端
    'get_mqtt_client',
    'local_client',
    # 配置常量
    'EXPERIMENT_TOPIC',
    'REAGENT_LAYOUT_PATH',
    'topic',
    'json_path',
    # 试剂工具
    'find_reagent',
    'get_reagent',
    # PydanticAI 异步工具
    'Deps',
    'read_pdf',
    'get_all_reagents',
    'save_experiment_step',
    'start_experiment',
    'do_experiment',
]
