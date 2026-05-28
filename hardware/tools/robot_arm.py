"""
机械臂移动工具
"""
from typing import *
from ..mqtt import get_mqtt_client, EXPERIMENT_TOPIC
from .registry import register_tool


@register_tool(
    name="move_robot_arm",
    description="移动机械臂到指定坐标",
    params={
        "x": {"type": "float", "description": "X坐标", "required": True, "default": 220},
        "y": {"type": "float", "description": "Y坐标", "required": True, "default": -220},
        "z": {"type": "float", "description": "Z坐标", "required": True, "default": 200},
        "r": {"type": "float", "description": "R轴坐标", "required": False, "default": 0}
    }
)
def move_robot_arm(x: float, y: float, z: float, r: float) -> str:
    """
    底层同步函数：移动机械臂到指定坐标位置

    Args:
        x : X轴坐标
        y : Y轴坐标
        z : Z轴坐标
        r : R轴坐标

    Returns:
        str: 机械臂移动结果消息
    """
    #"ax,y,z,r,grip", grip=0 if loosen the gripper
    try:
        client = get_mqtt_client()
        if client.is_connected:
            client.publish(EXPERIMENT_TOPIC, f"a{x},{y},{z},{r},0")
            return f"机械臂已保存轨迹:移动至坐标 ({x}, {y}, {z}, {r})"
        else:
            connect_state = client.connect()
            if connect_state:
                client.publish(EXPERIMENT_TOPIC, f"a{x},{y},{z},{r},0")
                return f"机械臂已保存轨迹:移动至坐标 ({x}, {y}, {z}, {r})"
            else:
                return f"机械臂移动轨迹保存失败"
    except Exception as e:
        return f"机械臂移动轨迹保存失败: {str(e)}"
