"""
数据归一化/标准化算法 (software/algorithms/default/data_normalization.py)
=========================================================================

支持的方法：
    - minmax   : Min-Max 归一化，将数据映射到 [0, 1]
    - zscore   : Z-Score 标准化，均值为 0，标准差为 1
    - robust   : 鲁棒标准化，使用中位数和 IQR，对异常值不敏感

输入格式：
    dict  → {"col_a": [1, 2, 3], "col_b": [4, 5, 6]}   （多列）
    list  → [1, 2, 3, 4, 5]                              （单列）

输出格式（result 字段）：
    {
        "normalized": {
            "col_a": [0.0, 0.5, 1.0],
            ...
        },
        "transform_params": {           # 逆变换所需参数
            "col_a": {"method": "minmax", "min": 1.0, "max": 3.0},
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


class DataNormalization(BaseAlgorithm):
    """数据归一化与标准化算法"""

    name = "data_normalization"
    chinese_name = "数据归一化"
    description = "对数值数据进行归一化或标准化处理，支持 Min-Max、Z-Score、Robust 三种方法"
    params_schema = {
        "method": {
            "type": "str",
            "description": "归一化方法：'minmax'（默认）/ 'zscore' / 'robust'",
            "default": "minmax",
            "required": False,
        },
        "columns": {
            "type": "list",
            "description": "要处理的列名列表（dict 输入时有效），留空则处理所有列",
            "default": None,
            "required": False,
        },
        "feature_range": {
            "type": "list",
            "description": "minmax 方法的目标区间 [min, max]，默认 [0, 1]",
            "default": [0, 1],
            "required": False,
        },
    }
    result_schema = {
        "type": "kv",
        "items": [
            {"key": "method",        "label": "归一化方法"},
            {"key": "original_min",  "label": "原始最小值",  "format": "decimal:4"},
            {"key": "original_max",  "label": "原始最大值",  "format": "decimal:4"},
            {"key": "mean",          "label": "均值",        "format": "decimal:4"},
            {"key": "std",           "label": "标准差",      "format": "decimal:4"},
            {"key": "median",        "label": "中位数",      "format": "decimal:4"},
            {"key": "iqr",           "label": "IQR",         "format": "decimal:4"},
        ],
    }

    _SUPPORTED_METHODS = ("minmax", "zscore", "robust")

    def run(self, data: Any, params: dict = None) -> dict:
        """
        执行数据归一化

        Args:
            data  : dict（多列）或 list（单列）数值数据
            params: {
                "method"       : str,
                "columns"      : list[str] | None,
                "feature_range": [float, float]   # 仅 minmax 有效
            }

        Returns:
            统一格式 dict，result 包含 normalized 数据和 transform_params
        """
        params = params or {}
        method = params.get("method", "minmax").lower()

        if method not in self._SUPPORTED_METHODS:
            return self._build_error(
                f"不支持的方法 '{method}'，可选: {', '.join(self._SUPPORTED_METHODS)}"
            )

        try:
            arrays = self._parse_data(data, params.get("columns"))
        except ValueError as e:
            return self._build_error(str(e))

        if not arrays:
            return self._build_error("数据为空，无法归一化")

        normalized = {}
        transform_params = {}

        for col, arr in arrays.items():
            norm_arr, tp = self._normalize(arr, method, params)
            normalized[col] = norm_arr.tolist()
            transform_params[col] = tp

        return self._build_success(
            {"normalized": normalized, "transform_params": transform_params},
            f"已使用 {method} 方法完成 {len(arrays)} 列数据的归一化"
        )

    # ------------------------------------------------------------------
    # 归一化方法实现
    # ------------------------------------------------------------------

    def _normalize(self, arr: np.ndarray, method: str, params: dict):
        """单列归一化，返回 (归一化后数组, 变换参数)"""
        if method == "minmax":
            return self._minmax(arr, params.get("feature_range", [0, 1]))
        elif method == "zscore":
            return self._zscore(arr)
        elif method == "robust":
            return self._robust(arr)

    def _minmax(self, arr: np.ndarray, feature_range):
        """Min-Max 归一化"""
        a, b = float(feature_range[0]), float(feature_range[1])
        x_min, x_max = float(np.nanmin(arr)), float(np.nanmax(arr))
        denom = x_max - x_min if x_max != x_min else 1.0
        norm = (arr - x_min) / denom * (b - a) + a
        return norm, {
            "method": "minmax",
            "original_min": x_min,
            "original_max": x_max,
            "feature_range": [a, b],
        }

    def _zscore(self, arr: np.ndarray):
        """Z-Score 标准化"""
        mean = float(np.nanmean(arr))
        std  = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 1.0
        denom = std if std != 0 else 1.0
        norm = (arr - mean) / denom
        return norm, {
            "method": "zscore",
            "mean": mean,
            "std": std,
        }

    def _robust(self, arr: np.ndarray):
        """Robust 标准化（中位数 + IQR）"""
        median = float(np.nanmedian(arr))
        q25    = float(np.nanpercentile(arr, 25))
        q75    = float(np.nanpercentile(arr, 75))
        iqr    = q75 - q25 if q75 != q25 else 1.0
        norm = (arr - median) / iqr
        return norm, {
            "method": "robust",
            "median": median,
            "q25": q25,
            "q75": q75,
            "iqr": iqr,
        }

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _parse_data(self, data: Any, columns=None) -> dict:
        if isinstance(data, dict):
            selected = columns if columns else list(data.keys())
            result = {}
            for col in selected:
                if col in data:
                    try:
                        result[col] = np.array(data[col], dtype=float)
                    except (TypeError, ValueError):
                        pass
            return result
        elif isinstance(data, list):
            return {"data": np.array(data, dtype=float)}
        else:
            raise ValueError(f"不支持的数据类型: {type(data).__name__}，请传入 dict 或 list")


# ==============================================================================
# 测试接口
# ==============================================================================

if __name__ == "__main__":
    import json

    algo = DataNormalization()
    print(f"算法信息: {algo.get_info()}\n")

    raw = [10, 20, 30, 40, 50]

    # 示例1：Min-Max 归一化
    r1 = algo.run(raw, {"method": "minmax"})
    print("示例1 - Min-Max 归一化:")
    print(json.dumps(r1, indent=2, ensure_ascii=False))

    # 示例2：Z-Score 标准化
    r2 = algo.run(raw, {"method": "zscore"})
    print("\n示例2 - Z-Score 标准化:")
    print(json.dumps(r2, indent=2, ensure_ascii=False))

    # 示例3：Robust 标准化（多列）
    r3 = algo.run(
        data={
            "efficiency": [0.10, 0.15, 0.18, 0.20, 0.17],
            "thickness" : [50, 80, 120, 160, 100],
        },
        params={"method": "robust"}
    )
    print("\n示例3 - Robust 标准化（多列）:")
    print(json.dumps(r3, indent=2, ensure_ascii=False))

    # 示例4：自定义 feature_range
    r4 = algo.run(raw, {"method": "minmax", "feature_range": [-1, 1]})
    print("\n示例4 - Min-Max 映射到 [-1, 1]:")
    print(json.dumps(r4, indent=2, ensure_ascii=False))
