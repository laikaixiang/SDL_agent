"""
基板转移
"""
from typing import *
from ..mqtt import get_mqtt_client, EXPERIMENT_TOPIC
from .registry import register_tool


@register_tool(
    name="move_substrate",
    description="基板转移",
    params={
        "start_pos": {"type": "Tuple[int, int, int, int]", "description": "基底在托盘中位置_起始,[x,y,z,r]", "required": True, "default": [220, -220, 200, 0]},
        "end_pos": {"type": "Tuple[int, int, int, int]", "description": "基底在托盘中位置_目标,[x,y,z,r]", "required": True, "default": [220, -220, 200, 0]}
    }
)
def move_substrate(start_pos: Tuple[int, int, int, int], end_pos: Tuple[int, int, int, int]) -> str:
    """
    底层同步函数：基板转移

    Args:
        "start_pos": {"type": "Tuple[int, int, int, int]", "description": "基底在托盘中位置_起始,[x,y,z,r]", "required": True, "default": [220, -220, 200, 0]},
        "end_pos": {"type": "Tuple[int, int, int, int]", "description": "基底在托盘中位置_目标,[x,y,z,r]", "required": True, "default": [220, -220, 200, 0]}

    Returns:
        str: 返回结果消息
    """
    x_init, y_init, z_init, r_init = start_pos[0], start_pos[1], start_pos[2], start_pos[3]
    x_tar, y_tar, z_tar, r_tar = end_pos[0], end_pos[1], end_pos[2], end_pos[3]
    try:
        client = get_mqtt_client()
        if client.is_connected:
            client.publish(EXPERIMENT_TOPIC, f"a{x_init},{y_init},200,{r_init},0")
            client.publish(EXPERIMENT_TOPIC, f"a{x_init},{y_init},{z_init},{r_init},1")
            client.publish(EXPERIMENT_TOPIC, f"a{x_init},{y_init},200,{r_init},1")
            client.publish(EXPERIMENT_TOPIC, f"a{x_tar},{y_tar},200,{r_tar},1")
            client.publish(EXPERIMENT_TOPIC, f"a{x_tar},{y_tar},{z_tar},{r_tar},0")
            client.publish(EXPERIMENT_TOPIC, f"a{x_tar},{y_tar},200,{r_tar},0")
            client.listen_to_message("done")
            return f"机械臂已执行运动:将基板从({x_init}, {y_init}, {z_init})转移至({x_tar}, {y_tar}, {z_tar})"
        else:
            connect_state = client.connect()
            if connect_state:
                client.publish(EXPERIMENT_TOPIC, f"a{x_init},{y_init},200,{r_init},0")
                client.publish(EXPERIMENT_TOPIC, f"a{x_init},{y_init},{z_init},{r_init},1")
                client.publish(EXPERIMENT_TOPIC, f"a{x_init},{y_init},200,{r_init},1")
                client.publish(EXPERIMENT_TOPIC, f"a{x_tar},{y_tar},200,{r_tar},1")
                client.publish(EXPERIMENT_TOPIC, f"a{x_tar},{y_tar},{z_tar},{r_tar},0")
                client.publish(EXPERIMENT_TOPIC, f"a{x_tar},{y_tar},200,{r_tar},0")
                client.listen_to_message("done")
                return f"机械臂已执行运动:将基板从({x_init}, {y_init}, {z_init})转移至({x_tar}, {y_tar}, {z_tar})"
            else:
                return f"机械臂基板转移失败"
    except Exception as e:
        return f"机械臂基板转移失败: {str(e)}"