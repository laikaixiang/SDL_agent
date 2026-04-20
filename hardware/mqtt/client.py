"""
MQTT客户端管理 - 懒加载模式
"""

from ..agent_client import MQTTConnector

# 全局MQTT客户端实例（模块级单例）
_local_client = None


def get_mqtt_client() -> MQTTConnector:
    """
    获取全局MQTT客户端实例（懒加载 + 自动重连）

    首次调用时会创建MQTTConnector实例并尝试连接。
    后续调用时如果连接已断开，会自动尝试重连。

    Returns:
        MQTTConnector: 已连接（或已尝试连接）的MQTT客户端实例

    使用示例::

        client = get_mqtt_client()
        if client.is_connected:
            client.publish("do_experiment", "pstart")
    """
    global _local_client
    if _local_client is None:
        # 首次调用：创建新的MQTT连接器实例
        _local_client = MQTTConnector()
    if not _local_client.is_connected:
        # 如果当前未连接，尝试重新连接（超时2秒）
        _local_client.connect(timeout=2)
    return _local_client


class _LazyClient:
    """
    懒加载代理类，使 `local_client.is_connected` 等属性访问时
    自动触发 get_mqtt_client()，避免模块导入时就连接MQTT
    """
    @property
    def is_connected(self) -> bool:
        """获取当前MQTT连接状态"""
        return get_mqtt_client().is_connected

    def connect(self, timeout=5) -> bool:
        """连接MQTT服务器"""
        return get_mqtt_client().connect(timeout)

    def check_connect(self) -> bool:
        """检查MQTT连接是否正常"""
        return get_mqtt_client().check_connect()

    def publish(self, topic: str, msg: str):
        """发布MQTT消息"""
        get_mqtt_client().publish(topic, msg)


# local_client: 供外部模块（如core/hardware_controller.py）直接引用的MQTT客户端
# 通过_LazyClient代理实现懒加载
local_client = _LazyClient()
