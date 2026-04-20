"""
硬件通信层包 - MQTT通信、硬件操控函数、光谱仪采集、数据可视化

TODO: 将 hardware/tools/__init__.py 中的所有导出内容迁移到此文件，
      统一从 hardware 包顶层导入，消除多层 __init__ 的冗余。
"""

# 从 agent_client 模块导入 MQTT 相关类
from .agent_client import MQTTConnector   # MQTT 连接管理器
from .agent_client import Client_Conf     # MQTT 连接配置（IP、端口、用户名密码等）

__all__ = [
    'MQTTConnector',
    'Client_Conf',
]
