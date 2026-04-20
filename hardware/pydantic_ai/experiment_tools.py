"""
实验工具 - PydanticAI异步工具
"""

import uuid
from pydantic_ai import RunContext

from .deps import Deps
from ..mqtt import get_mqtt_client, EXPERIMENT_TOPIC
from ..utils.reagent import find_reagent


async def save_experiment_step(
    ctx: RunContext[Deps],
    spin_speed: int = 3000,
    spin_acc: int = 1000,
    spin_dur: int = 30000,
    reagent: str = "",
    volume: int = 10,
) -> str:
    """
    注册一步旋涂实验参数到自动化平台（需用户确认）

    此函数将实验参数格式化为MQTT消息发送给C#平台保存。
    一轮完整实验可能包含多个步骤（如先涂底层、再涂活性层），
    每个步骤需调用一次save_experiment_step()，
    所有步骤注册完毕后再调用start_experiment()启动执行。

    MQTT消息格式: "p{转速},{加速度},{时长},{试剂位置},{体积}"
    示例: "p3000,1000,30000,BP01,10"

    Args:
        ctx        : PydanticAI运行上下文
        spin_speed : 旋涂转速，单位rpm，最大6000rpm，默认3000rpm
        spin_acc   : 旋涂加速度，单位rpm/s，默认1000rpm/s
        spin_dur   : 旋涂持续时间，单位毫秒(ms)，默认30000ms（即30秒）
        reagent    : 使用的试剂名称（必须与reagent_layout.json中的名称一致）
        volume     : 试剂滴加体积，单位微升(µl)，默认10µl

    Returns:
        str: 成功时返回包含所有参数的确认消息（带✅前缀）
             试剂未找到时返回"Reagent is missing"
             MQTT连接失败时返回"Connect server failed"

    AI使用说明：
        - 先读取论文获取实验参数（转速、时间、试剂等）
        - 对每一步实验调用一次此函数
        - 所有步骤注册完毕后，调用start_experiment()启动
        - 如果收到"Reagent is missing"，可调用get_all_reagents()检查拼写
    """
    try:
        # 生成唯一请求ID
        request_id = str(uuid.uuid4())

        # 推送确认请求到前端
        await ctx.deps.send_event({
            "type": "experiment_confirm",
            "tool": "save_experiment_step",
            "request_id": request_id,
            "session_id": ctx.deps.session_id,
            "params": {
                "spin_speed": spin_speed,
                "spin_acc": spin_acc,
                "spin_dur": spin_dur,
                "reagent": reagent,
                "volume": volume,
            }
        })

        # 等待用户响应
        if ctx.deps.agent:
            response = await ctx.deps.agent.wait_for_response(request_id)

            if response["action"] == "skip":
                return "用户跳过此步骤"
            elif response["action"] == "cancel":
                return "用户取消操作"
            elif response["action"] == "timeout":
                return "等待用户确认超时"
            elif response["action"] == "confirm":
                # 使用修改后的参数（如果有）
                params = response.get("params", {})
                spin_speed = params.get("spin_speed", spin_speed)
                spin_acc = params.get("spin_acc", spin_acc)
                spin_dur = params.get("spin_dur", spin_dur)
                reagent = params.get("reagent", reagent)
                volume = params.get("volume", volume)

        # 通知前端：save_experiment_step工具被调用，附带参数详情
        await ctx.deps.send_event({
            "type": "tool_call",
            "name": "save_experiment_step",
            "args": {
                "spin_speed": spin_speed,
                "spin_acc": spin_acc,
                "spin_dur": spin_dur,
                "reagent": reagent,
                "volume": volume,
            },
        })

        # 在reagent_layout.json中查找试剂的物理位置（如"BP01"）
        reagent_pos = find_reagent(reagent)
        if reagent_pos[:2] != "BP":
            # 未找到试剂（返回值不以"BP"开头，说明是错误消息）
            return reagent_pos

        # 尝试通过MQTT发送实验参数
        client = get_mqtt_client()
        if client.is_connected:
            # 格式化并发送MQTT消息
            client.publish(EXPERIMENT_TOPIC, f"p{spin_speed},{spin_acc},{spin_dur},{reagent_pos},{volume}")
            msg = (
                f"✅ 实验步骤已注册: 试剂 {reagent} (位置 {reagent_pos}), "
                f"转速 {spin_speed} rpm, 加速度 {spin_acc} rpm/s, "
                f"持续 {spin_dur} ms, 体积 {volume} µl"
            )
            await ctx.deps.send_event({
                "type": "tool_result",
                "name": "save_experiment_step",
                "result": msg,
            })
            return msg
        else:
            # 当前未连接，尝试重新连接
            connect_state = client.connect()
            if connect_state:
                client.publish(EXPERIMENT_TOPIC, f"p{spin_speed},{spin_acc},{spin_dur},{reagent_pos},{volume}")
                msg = (
                    f"✅ 实验步骤已注册: 试剂 {reagent} (位置 {reagent_pos}), "
                    f"转速 {spin_speed} rpm, 加速度 {spin_acc} rpm/s, "
                    f"持续 {spin_dur} ms, 体积 {volume} µl"
                )
                await ctx.deps.send_event({
                    "type": "tool_result",
                    "name": "save_experiment_step",
                    "result": msg,
                })
                return msg
            else:
                return "Connect server failed"
    except Exception as e:
        err = f"Error occurred: {str(e)}"
        return err


async def start_experiment(
    ctx: RunContext[Deps],
) -> bool:
    """
    启动已注册的多步实验序列（需用户确认）

    向C#自动化平台发送"pstart"命令，平台会按照之前通过
    save_experiment_step()注册的步骤顺序，依次执行所有实验操作。

    调用前提：
        必须先通过save_experiment_step()注册至少一个实验步骤。
        如果没有注册任何步骤就调用此函数，平台不会执行任何操作。

    Args:
        ctx : PydanticAI运行上下文

    Returns:
        bool: True表示启动指令发送成功，False表示MQTT连接失败

    AI使用说明：
        - 确保所有实验步骤都已通过save_experiment_step()注册
        - 检查步骤数量是否与论文描述一致
        - 确认后再调用此函数启动实验
    """
    try:
        # 生成唯一请求ID
        request_id = str(uuid.uuid4())

        # 推送确认请求到前端
        await ctx.deps.send_event({
            "type": "experiment_confirm",
            "tool": "start_experiment",
            "request_id": request_id,
            "session_id": ctx.deps.session_id,
            "params": {}
        })

        # 等待用户响应
        if ctx.deps.agent:
            response = await ctx.deps.agent.wait_for_response(request_id)

            if response["action"] == "skip":
                return False
            elif response["action"] == "cancel":
                return False
            elif response["action"] == "timeout":
                return False

        # 通知前端：start_experiment工具被调用
        await ctx.deps.send_event({
            "type": "tool_call",
            "name": "start_experiment",
            "args": {},
        })

        client = get_mqtt_client()
        if client.is_connected:
            # 发送启动命令"pstart"
            client.publish(EXPERIMENT_TOPIC, "pstart")
            msg = "✅ 实验序列已启动"
            await ctx.deps.send_event({
                "type": "tool_result",
                "name": "start_experiment",
                "result": msg,
            })
            return True
        else:
            # 尝试重连
            connect_state = client.connect()
            if connect_state:
                client.publish(EXPERIMENT_TOPIC, "pstart")
                msg = "✅ 实验序列已启动"
                await ctx.deps.send_event({
                    "type": "tool_result",
                    "name": "start_experiment",
                    "result": msg,
                })
                return True
            else:
                return False

    except Exception:
        return False


async def do_experiment(
    ctx: RunContext[Deps],
    spin_speed: int = 3000,
    spin_acc: int = 1000,
    spin_dur: int = 30000,
    reagent: str = "",
    volume: int = 10,
) -> str:
    """
    执行单次旋涂实验（旧接口，保留向后兼容）

    与save_experiment_step + start_experiment的区别：
    - do_experiment是单次执行，适合只有一步的简单实验
    - save_experiment_step + start_experiment支持多步实验序列

    新代码建议使用save_experiment_step + start_experiment组合。

    Args:
        ctx        : PydanticAI运行上下文
        spin_speed : 旋涂转速(rpm)，最大6000，默认3000
        spin_acc   : 加速度(rpm/s)，默认1000
        spin_dur   : 持续时间(ms)，默认30000
        reagent    : 试剂名称
        volume     : 体积(µl)，默认10

    Returns:
        str: 成功时返回确认消息，失败时返回错误描述
    """
    try:
        await ctx.deps.send_event({
            "type": "tool_call",
            "name": "do_experiment",
            "args": {
                "spin_speed": spin_speed,
                "spin_acc": spin_acc,
                "spin_dur": spin_dur,
                "reagent": reagent,
                "volume": volume,
            },
        })

        # 查找试剂位置
        reagent_pos = find_reagent(reagent)
        if reagent_pos[:2] != "BP":
            return reagent_pos

        client = get_mqtt_client()
        if client.is_connected:
            client.publish(EXPERIMENT_TOPIC, f"p{spin_speed},{spin_acc},{spin_dur},{reagent_pos},{volume}")
            msg = (
                f"✅ Experiment started: seeking {reagent} at {reagent_pos}, "
                f"{spin_speed} rpm, acc {spin_acc} rpm/s, "
                f"duration {spin_dur} ms, volume {volume} µl."
            )
            await ctx.deps.send_event({"type": "tool_result", "name": "do_experiment", "result": msg})
            return msg
        else:
            connect_state = client.connect()
            if connect_state:
                client.publish(EXPERIMENT_TOPIC, f"p{spin_speed},{spin_acc},{spin_dur},{reagent_pos},{volume}")
                msg = (
                    f"✅ Experiment started: seeking {reagent} at {reagent_pos}, "
                    f"{spin_speed} rpm, acc {spin_acc} rpm/s, "
                    f"duration {spin_dur} ms, volume {volume} µl."
                )
                await ctx.deps.send_event({"type": "tool_result", "name": "do_experiment", "result": msg})
                return msg
            else:
                return "Connect server failed"
    except Exception as e:
        err = f"Error occurred: {str(e)}"
        return err
