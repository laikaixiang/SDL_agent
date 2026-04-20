"""
试剂工具 - PydanticAI异步工具
"""

from pydantic_ai import RunContext

from .deps import Deps
from ..mqtt.config import REAGENT_LAYOUT_PATH


async def get_all_reagents(
    ctx: RunContext[Deps],
    path: str = REAGENT_LAYOUT_PATH,
) -> str:
    """
    扫描reagent_layout.json，列出平台上所有已配置的试剂名称

    当AI调用save_experiment_step时收到"Reagent is missing"错误，
    可以调用此工具检查是否是试剂名拼写错误，或确认试剂是否已装载到平台上。

    Args:
        ctx  : PydanticAI运行上下文
        path : reagent_layout.json文件路径

    Returns:
        str: 逗号分隔的试剂名称列表，如"Perovskite, DMF, DMSO, "
             如果发生错误，返回错误描述字符串

    AI使用说明：
        - 当收到"Reagent is missing"时，调用此工具查看有哪些试剂可用
        - 检查试剂名是否拼写正确
    """
    try:
        # 通知前端：get_all_reagents工具被调用
        await ctx.deps.send_event({
            "type": "tool_call",
            "name": "get_all_reagents",
            "args": {},
        })

        import json
        with open(path, "r", encoding="utf-8") as f:
            available_reagents = ""   # 汇总字符串
            idx = 0                   # 已找到的试剂计数
            data = json.load(f)
            points = data.get("Points", {})
            for point_id, info in points.items():
                if info.get("name") != "":
                    # 将非空试剂名追加到结果字符串
                    available_reagents += f"{info.get('name')}, "
                    idx += 1

            msg = f"扫描完成，找到 {idx} 种可用试剂"
            await ctx.deps.send_event({
                "type": "tool_result",
                "name": "get_all_reagents",
                "result": msg,
            })
            return available_reagents

    except Exception as e:
        err = str(e)
        return err
