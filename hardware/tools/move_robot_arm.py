"""
机械臂移动工具
"""
from typing import *
from ..mqtt import get_mqtt_client, EXPERIMENT_TOPIC
# ============================================================
# 数字孪生离线规划器
#
# MQTT 真硬件优先,连不上或禁用则自动退到数字孪生。
# 孪生后端收到指令后在浏览器播 3D 动画,完成返回 "done"。
#
# TODO(后续): 合并到 config.json,不硬编码
# ============================================================
# MQTT 开关: True=先试真硬件,False=直接孪生
USE_MQTT = True
# 数字孪生地址(背后是一个 Three.js HTML 动画页面)
_TWIN_URL = "http://127.0.0.1:5001"

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
    底层同步函数:发指令 → 阻塞等"done" → 返回结果。
    优先 MQTT 真硬件; 失败/禁用时退到数字孪生播 HTML 动画。

    Args:
        x : X轴坐标
        y : Y轴坐标
        z : Z轴坐标
        r : R轴坐标

    Returns:
        str: 机械臂移动结果消息
    """
    payload = f"a{x},{y},{z},{r},0"

    # 安全验证 (两路径共用)
    from ..utils.collision import check_collision
    code, reason = check_collision({"x": x, "y": y, "z": z, "r": r})
    if code != 200:
        return f"机械臂移动拒绝 [400]: {reason}"

    # ① 优先 MQTT 真硬件
    if USE_MQTT:
        try:
            client = get_mqtt_client()
            if not client.is_connected:
                client.connect()
            if client.is_connected:
                client.publish(EXPERIMENT_TOPIC, payload)
                # TODO(V2): listen_to_message 应设超时,超时后跌落到孪生
                client.listen_to_message("done")
                return f"机械臂已移动至坐标 ({x}, {y}, {z}, {r}, 0) [真硬件]"
        except Exception as e:
            print(f"[MQTT] 降级到孪生: {e}")

    # ② 数字孪生 (兜底) — 背后是 Three.js HTML 页面播 3D 动画
    try:
        import requests
        print(f"[Twin] [中断] 等待孪生执行...")
        resp = requests.post(f"{_TWIN_URL}/api/twin/execute",
            json={"msg": payload}, timeout=60)
        body = resp.json()
        if resp.status_code == 200:
            return f"机械臂已移动至坐标 ({x}, {y}, {z}, {r}, 0) [孪生]"
        else:
            return f"机械臂移动拒绝 [400]: {body.get('reason', '未知')}"
    except Exception as e:
        return f"机械臂移动失败: {str(e)}"
