"""
放滴头
"""
from typing import *
from ..mqtt import get_mqtt_client, EXPERIMENT_TOPIC
from .registry import register_tool


@register_tool(
    name="pop_tips",
    description="空气泵放枪头",
    params={
        "trash_pos": {"type": "Tuple[int,int,int]", "description": "滴头垃圾桶的位置,[x,y,z]", "required": True, "default": [0,0,0]},
        "tip": {"type": "int", "description": "空气泵编号(1或者2,1为左泵,2为右泵)", "required": True, "default": 1},
    }
)
def pop_tips(trash_pos: Tuple[int, int, int], tip: int) -> str:
    """
    底层同步函数：空气泵放枪头

    Args:
        "trash_pos": {"type": "Tuple[int,int,int]", "description": "滴头垃圾桶的位置,[x,y,z]", "required": True, "default": [0,0,0]},
        "tip": {"type": "int", "description": "空气泵编号(1或者2,1为左泵,2为右泵)", "required": True, "default": 1},

    Returns:
        str: 返回结果消息
    """
    # "d<action code>,tip,x,y,z,vol", action code 3: spit
    x, y, z = trash_pos[0], trash_pos[1], trash_pos[2]
    try:
        client = get_mqtt_client()
        if client.is_connected:
            client.publish(EXPERIMENT_TOPIC, f"d0,0,{x},{y},0,0")# move horizontal
            client.publish(EXPERIMENT_TOPIC, f"d1,{tip},0,0,{z},0,0")# move tip vertical
            client.publish(EXPERIMENT_TOPIC, f"d1,{tip},0,0,0,0,0")# move tip back
            client.listen_to_message("done")
            return f"滴液机已执行:{tip}号泵在x={trash_pos[0]},y={trash_pos[1]},z={trash_pos[2]}位置扔掉滴头"
        else:
            connect_state = client.connect()
            if connect_state:
                client.publish(EXPERIMENT_TOPIC, f"d0,0,{x},{y},0,0")
                client.publish(EXPERIMENT_TOPIC, f"d1,{tip},0,0,{z},0,0")
                client.publish(EXPERIMENT_TOPIC, f"d1,{tip},0,0,0,0,0")
                client.listen_to_message("done")
                return f"滴液机已执行:{tip}号泵在x={trash_pos[0]},y={trash_pos[1]},z={trash_pos[2]}位置扔掉滴头"
            else:
                return f"滴液机放滴头失败"
    except Exception as e:
        return f"滴液机放滴头失败: {str(e)}"
