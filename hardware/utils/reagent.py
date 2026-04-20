"""
试剂查找辅助函数
"""

import json
from ..mqtt.config import REAGENT_LAYOUT_PATH


def find_reagent(name: str, path: str = REAGENT_LAYOUT_PATH) -> str:
    """
    根据试剂名称在reagent_layout.json中查找其物理位置

    reagent_layout.json文件结构示例::

        {
            "Points": {
                "BP01": {"name": "Perovskite", "x": 100, "y": 200},
                "BP02": {"name": "DMF", "x": 150, "y": 250},
                ...
            }
        }

    Args:
        name : 要查找的试剂名称（必须完全匹配，区分大小写）
        path : reagent_layout.json文件路径，默认使用REAGENT_LAYOUT_PATH

    Returns:
        str: 如果找到，返回位置编号字符串（如"BP01"）
             如果未找到，返回"Reagent is missing"
             如果读取文件出错，返回错误描述字符串
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # "Points"字典中每个key是位置编号（如"BP01"），value包含试剂信息
            points = data.get("Points", {})
            for point_id, info in points.items():
                reagent_name = info.get("name", "")
                if reagent_name == name:
                    return point_id  # 找到匹配的试剂，返回位置编号
            return "Reagent is missing"  # 遍历完毕未找到
    except Exception as e:
        err = str(e)
        return err


# 兼容旧代码：get_reagent是find_reagent的别名
get_reagent = find_reagent
