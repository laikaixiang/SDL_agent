"""
硬件工具模块 - 向后兼容层

⚠️ 所有导出内容已迁移到 hardware/__init__.py
   新代码请使用 from hardware import xxx 代替 from hardware.tools import xxx

本文件通过延迟加载从 hardware 顶层重新导出，保持旧代码兼容。
"""

import importlib

# 子模块仍需要 ToolRegistry（虽然不通过 __init__.py 导入，但保留直接访问路径）
try:
    from .registry import ToolRegistry
except ImportError:
    class ToolRegistry:
        pass


def __getattr__(name):
    """延迟从 hardware 顶层获取属性，避免循环导入"""
    import sys
    mod = sys.modules.get(__name__)
    if mod and name in mod.__dict__:
        return mod.__dict__[name]

    hw = importlib.import_module('hardware')
    try:
        return getattr(hw, name)
    except AttributeError:
        raise ImportError(f"cannot import name '{name}' from 'hardware.tools'")


__all__ = [
    'ToolRegistry',
    'execute_spin_coating',
    'execute_set_temperature',
    'execute_move_robot_arm',
    'execute_start_experiment',
    'execute_collect_spectrum',
    'get_mqtt_client',
    'local_client',
    'EXPERIMENT_TOPIC',
    'REAGENT_LAYOUT_PATH',
    'topic',
    'json_path',
    'find_reagent',
    'get_reagent',
    'Deps',
    'read_pdf',
    'get_all_reagents',
    'save_experiment_step',
    'do_experiment',
]
