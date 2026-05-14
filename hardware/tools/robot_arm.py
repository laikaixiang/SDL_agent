"""
机械臂移动工具
"""
from ..mqtt import get_mqtt_client
from .registry import register_tool


@register_tool(
    name="move_robot_arm",
    description="移动机械臂到指定坐标",
    params={
        "x": {"type": "float", "description": "X坐标", "required": True, "default": 220},
        "y": {"type": "float", "description": "Y坐标", "required": True, "default": -220},
        "z": {"type": "float", "description": "Z坐标", "required": True, "default": 20},
        "r": {"type": "float", "description": "r", "required": True, "default": 0}
    }
)
def execute_move_robot_arm(x: float, y: float, z: float, r: float) -> str:
    """
    底层同步函数：移动机械臂到指定坐标位置

    当前为模拟实现，实际部署时需取消注释subprocess调用。

    Args:
        x : X轴坐标
        y : Y轴坐标
        z : Z轴坐标

    Returns:
        str: 机械臂移动结果消息
    """
    try:
        client = get_mqtt_client()
        experiment_topic = "do_experiment"
        if client.is_connected:
            client.publish(experiment_topic, f"a{x},{y},{z},{r}")
            client.publish(experiment_topic, "astart")
            return f"机械臂已精准移动至坐标 ({x}, {y}, {z})"    
        else:
            connect_state = client.connect()
            if connect_state:
                client.publish(experiment_topic, f"a{x},{y},{z},{r}")
                client.publish(experiment_topic, "astart")
                return f"机械臂已精准移动至坐标 ({x}, {y}, {z})"
            else:
                return f"机械臂移动失败: {str(e)}"
    except Exception as e:
        return f"机械臂移动失败: {str(e)}"
