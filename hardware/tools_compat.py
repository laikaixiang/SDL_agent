"""
(后续要删)
硬件工具模块 - 向后兼容层
================================

本文件作为兼容层，重新导出所有从hardware/tools.py拆分到各子模块的函数。
现有代码中的 `from hardware.tools import xxx` 无需修改即可继续工作。

新架构：
- hardware/tools/          : 实验操作工具（注册系统）
- hardware/mqtt/           : MQTT通信层
- hardware/utils/          : 辅助工具（试剂查找等）
- hardware/pydantic_ai/    : PydanticAI异步工具

使用示例（向后兼容）::

    from hardware.tools import execute_spin_coating, Deps, read_pdf
    # 以上导入方式继续有效

使用示例（新架构）::

    from hardware.tools import execute_spin_coating
    from hardware.pydantic_ai import Deps, read_pdf
    from hardware.mqtt import get_mqtt_client
"""

# 从新位置导入所有函数
# 注意：使用相对导入避免循环导入
from .tools import (
    execute_spin_coating,
    execute_set_temperature,
    execute_move_robot_arm,
    execute_start_experiment,
    execute_collect_spectrum,
    ToolRegistry,
)

from .mqtt import get_mqtt_client, local_client, EXPERIMENT_TOPIC, REAGENT_LAYOUT_PATH

from .utils.reagent import find_reagent, get_reagent

from .pydantic_ai import (
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
