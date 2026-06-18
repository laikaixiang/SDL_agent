"""
数据统计分析算法 (software/algorithms/default/data_statistics.py)
================================================================

对数值数据进行基础统计分析，包括：
    - 描述性统计：均值、中位数、标准差、方差、最值、分位数
    - 多列相关性矩阵

输入格式：
    dict  → {"col_a": [1, 2, 3], "col_b": [4, 5, 6]}   （多列）
    list  → [1, 2, 3, 4, 5]                              （单列）

输出格式（result 字段）：
    {
        "statistics": {
            "col_a": {"count": 3, "mean": 2.0, "median": 2.0, ...},
            ...
        },
        "correlation": {           # 仅多列且 include_correlation=True 时出现
            "col_a": {"col_b": 1.0, ...},
            ...
        }
    }
"""

import numpy as np
from typing import Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from software.algorithms.base import BaseAlgorithm


class DataStatistics(BaseAlgorithm):
    """基础描述性统计算法"""

    name = "data_statistics"
    chinese_name = "数据统计分析"
    description = "对数值数据进行描述性统计分析（均值、中位数、标准差、最值、分位数、相关性矩阵）"
    params_schema = {
        "columns": {
            "type": "list",
            "description": "要分析的列名列表（数据为 dict 时有效），留空则分析所有列",
            "default": None,
            "required": False,
        },
        "include_correlation": {
            "type": "bool",
            "description": "是否计算列间相关性矩阵（多列时有效）",
            "default": True,
            "required": False,
        },
    }
    result_schema = {
        "type": "table",
        "sections": [
            {
                "title": "描述性统计",
                "columns": [
                    {"key": "count",   "label": "数量"},
                    {"key": "mean",    "label": "均值",    "format": "decimal:3"},
                    {"key": "median",  "label": "中位数",  "format": "decimal:3"},
                    {"key": "std",     "label": "标准差",  "format": "decimal:3"},
                    {"key": "variance","label": "方差",    "format": "decimal:3"},
                    {"key": "min",     "label": "最小值",  "format": "decimal:3"},
                    {"key": "max",     "label": "最大值",  "format": "decimal:3"},
                    {"key": "q25",     "label": "Q1",      "format": "decimal:3"},
                    {"key": "q75",     "label": "Q3",      "format": "decimal:3"},
                ],
                "rows_from": "result.statistics",
            },
            {
                "title": "相关性矩阵",
                "type": "matrix",
                "rows_from": "result.correlation",
                "value_format": "decimal:3",
            },
        ],
    }

    def run(self, data: Any, params: dict = None) -> dict:
        """
        执行统计分析

        Args:
            data  : dict（多列）或 list（单列）数值数据
            params: {
                "columns"            : list[str] | None,
                "include_correlation": bool
            }

        Returns:
            统一格式 dict，result 包含 statistics 与可选 correlation
        """
        params = params or {}

        try:
            arrays = self._parse_data(data, params.get("columns"))
        except ValueError as e:
            return self._build_error(str(e))

        if not arrays:
            return self._build_error("数据为空，无法进行统计分析")

        # 计算每列的描述性统计
        stats = {}
        for col, arr in arrays.items():
            arr_clean = arr[~np.isnan(arr)]
            if len(arr_clean) == 0:
                stats[col] = {"count": 0, "message": "全为 NaN，跳过"}
                continue
            stats[col] = {
                "count"   : int(len(arr_clean)),
                "mean"    : float(np.mean(arr_clean)),
                "median"  : float(np.median(arr_clean)),
                "std"     : float(np.std(arr_clean, ddof=1)) if len(arr_clean) > 1 else 0.0,
                "variance": float(np.var(arr_clean, ddof=1)) if len(arr_clean) > 1 else 0.0,
                "min"     : float(np.min(arr_clean)),
                "max"     : float(np.max(arr_clean)),
                "q25"     : float(np.percentile(arr_clean, 25)),
                "q75"     : float(np.percentile(arr_clean, 75)),
            }

        result = {"statistics": stats}

        # 多列时计算相关性矩阵
        if len(arrays) > 1 and params.get("include_correlation", True):
            col_names = list(arrays.keys())
            matrix_data = np.array([arrays[c] for c in col_names])
            corr = np.corrcoef(matrix_data)
            result["correlation"] = {
                col_names[i]: {
                    col_names[j]: round(float(corr[i][j]), 6)
                    for j in range(len(col_names))
                }
                for i in range(len(col_names))
            }

        return self._build_success(
            result,
            f"已完成 {len(arrays)} 列数据的统计分析，共 {sum(s.get('count', 0) for s in stats.values())} 个有效数据点"
        )

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _parse_data(self, data: Any, columns=None) -> dict:
        """将不同格式的输入统一转换为 {列名: numpy数组} 字典"""
        if isinstance(data, dict):
            selected = columns if columns else list(data.keys())
            result = {}
            for col in selected:
                if col not in data:
                    continue
                try:
                    result[col] = np.array(data[col], dtype=float)
                except (TypeError, ValueError):
                    pass  # 跳过无法转换的列
            return result

        elif isinstance(data, list):
            return {"data": np.array(data, dtype=float)}

        else:
            raise ValueError(f"不支持的数据类型: {type(data).__name__}，请传入 dict 或 list")


# ==============================================================================
# 测试接口
# ==============================================================================

if __name__ == "__main__":
    algo = DataStatistics()
    print(f"算法信息: {algo.get_info()}\n")

    # 示例1：单列 list 输入
    result1 = algo.run([10, 20, 30, 40, 50, 60])
    print("示例1 - 单列列表:")
    import json
    print(json.dumps(result1, indent=2, ensure_ascii=False))

    # 示例2：多列 dict 输入（含相关性矩阵）
    result2 = algo.run(
        data={
            "temperature": [100, 150, 200, 250, 300],
            "efficiency" : [0.10, 0.15, 0.18, 0.20, 0.17],
            "thickness"  : [50, 80, 120, 160, 100],
        },
        params={"include_correlation": True}
    )
    print("\n示例2 - 多列字典（含相关性矩阵）:")
    print(json.dumps(result2, indent=2, ensure_ascii=False))

    # 示例3：仅分析部分列
    result3 = algo.run(
        data={"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]},
        params={"columns": ["A", "C"], "include_correlation": False}
    )
    print("\n示例3 - 指定列分析（不含相关性）:")
    print(json.dumps(result3, indent=2, ensure_ascii=False))
