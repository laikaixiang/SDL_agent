"""
硬件通信层包 - MQTT通信、硬件操控函数、光谱仪采集、数据可视化

统一导出所有硬件相关功能，从 hardware 包顶层即可导入：
    from hardware import execute_spin_coating, get_mqtt_client, Deps
"""

# MQTT 连接器
from .agent_client import MQTTConnector, Client_Conf

# 工具注册系统
from .tools.registry import ToolRegistry, register_tool

# 底层同步工具函数
from .tools.spin_coating import execute_spin_coating
from .tools.temperature import execute_set_temperature
from .tools.robot_arm import execute_move_robot_arm
from .tools.experiment_control import execute_start_experiment
from .tools.spectrum import execute_collect_spectrum

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
    # 底层同步工具函数
    'execute_spin_coating',
    'execute_set_temperature',
    'execute_move_robot_arm',
    'execute_start_experiment',
    'execute_collect_spectrum',
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
