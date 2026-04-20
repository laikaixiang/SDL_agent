"""
硬件工具模块 - 统一导出接口

这个包导出所有硬件相关功能：
- 实验操作工具（本包内的子模块）
- MQTT通信（hardware.mqtt）
- 试剂查找（hardware.utils）
- PydanticAI工具（hardware.pydantic_ai）
"""

# 先导入registry，再导入使用它的工具
try:
    from .registry import ToolRegistry
except ImportError:
    # 如果registry.py不存在或有问题，提供一个空的占位符
    class ToolRegistry:
        pass

# 导入本包内的实验操作工具
from .spin_coating import execute_spin_coating
from .temperature import execute_set_temperature
from .robot_arm import execute_move_robot_arm
from .experiment_control import execute_start_experiment
from .spectrum import execute_collect_spectrum

# 从其他hardware子模块导入
from ..mqtt import get_mqtt_client, local_client, EXPERIMENT_TOPIC, REAGENT_LAYOUT_PATH
from ..utils.reagent import find_reagent, get_reagent
from ..pydantic_ai import (
    Deps,
    read_pdf,
    get_all_reagents,
    save_experiment_step,
    start_experiment,
    do_experiment,
)

# 兼容旧代码：保留topic和json_path变量
topic = EXPERIMENT_TOPIC
json_path = REAGENT_LAYOUT_PATH

__all__ = [
    # 工具函数（同步执行）
    'execute_spin_coating',
    'execute_set_temperature',
    'execute_move_robot_arm',
    'execute_start_experiment',
    'execute_collect_spectrum',
    'ToolRegistry',
    # MQTT客户端
    'get_mqtt_client',
    'local_client',
    # 配置常量
    'EXPERIMENT_TOPIC',
    'REAGENT_LAYOUT_PATH',
    'topic',  # 兼容旧代码
    'json_path',  # 兼容旧代码
    # 试剂工具
    'find_reagent',
    'get_reagent',
    # PydanticAI工具
    'Deps',
    'read_pdf',
    'get_all_reagents',
    'save_experiment_step',
    'start_experiment',
    'do_experiment',
]
