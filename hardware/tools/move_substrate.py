"""
基板转移
"""
from ..mqtt import get_mqtt_client, EXPERIMENT_TOPIC
from .registry import register_tool
from hardware.pos_config.parse_pos import parse_arm

@register_tool(
    name="move_substrate",
    description="基板转移",
    params={
        "start_pos": {"type": "int", "description": "基底在托盘中起始位置编号", "required": True, "default": 0},
        "end_pos": {"type": "int", "description": "基底在托盘中目标位置编号", "required": True, "default": 0}
    }
)
def move_substrate(start_pos: int, end_pos: int) -> str:
    """
    底层同步函数：基板转移

    Args:
        "start_pos": {"type": "int", "description": "基底在托盘中起始位置编号", "required": True, "default": 0},
        "end_pos": {"type": "int", "description": "基底在托盘中目标位置编号", "required": True, "default": 0}

    Returns:
        str: 返回结果消息
    """
    start_pos = parse_arm(start_pos)
    end_pos = parse_arm(end_pos)
    x_init, y_init, z_init, r_init = start_pos[0], start_pos[1], start_pos[2], start_pos[3]
    x_tar, y_tar, z_tar, r_tar = end_pos[0], end_pos[1], end_pos[2], end_pos[3]
    try:
        client = get_mqtt_client()
        if client.is_connected:
            client.publish(EXPERIMENT_TOPIC, f"a{x_init},{y_init},200,{r_init},0")
            client.listen_to_message("done")
            client.publish(EXPERIMENT_TOPIC, f"a{x_init},{y_init},{z_init},{r_init},1")
            client.listen_to_message("done")
            client.publish(EXPERIMENT_TOPIC, f"a{x_init},{y_init},200,{r_init},1")
            client.listen_to_message("done")
            client.publish(EXPERIMENT_TOPIC, f"a{x_tar},{y_tar},200,{r_tar},1")
            client.listen_to_message("done")
            client.publish(EXPERIMENT_TOPIC, f"a{x_tar},{y_tar},{z_tar},{r_tar},0")
            client.listen_to_message("done")
            client.publish(EXPERIMENT_TOPIC, f"a{x_tar},{y_tar},200,{r_tar},0")
            client.listen_to_message("done")
            return f"机械臂已执行运动:将基板从({x_init}, {y_init}, {z_init})转移至({x_tar}, {y_tar}, {z_tar})"
        else:
            connect_state = client.connect()
            if connect_state:
                client.publish(EXPERIMENT_TOPIC, f"a{x_init},{y_init},200,{r_init},0")
                client.listen_to_message("done")
                client.publish(EXPERIMENT_TOPIC, f"a{x_init},{y_init},{z_init},{r_init},1")
                client.listen_to_message("done")
                client.publish(EXPERIMENT_TOPIC, f"a{x_init},{y_init},200,{r_init},1")
                client.listen_to_message("done")
                client.publish(EXPERIMENT_TOPIC, f"a{x_tar},{y_tar},200,{r_tar},1")
                client.listen_to_message("done")
                client.publish(EXPERIMENT_TOPIC, f"a{x_tar},{y_tar},{z_tar},{r_tar},0")
                client.listen_to_message("done")
                client.publish(EXPERIMENT_TOPIC, f"a{x_tar},{y_tar},200,{r_tar},0")
                client.listen_to_message("done")
                return f"机械臂已执行运动:将基板从({x_init}, {y_init}, {z_init})转移至({x_tar}, {y_tar}, {z_tar})"
            else:
                return f"机械臂基板转移失败"
    except Exception as e:
        return f"机械臂基板转移失败: {str(e)}"
