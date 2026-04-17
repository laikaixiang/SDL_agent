"""
光谱数据分析算法 (software/algorithms/default/spectrum_analysis.py)
=================================================================

专为光谱仪数据设计的分析算法，输入格式与导出 Excel 结构一致：
    第一行：波长序列（wavelength）
    后续行：各次测量的强度序列（intensity，可有多条）

输入格式（dict）：
    {
        "wavelength": [400, 401, ..., 700],        # 波长列表
        "intensity" : [2, 1, ..., 234]             # 单条强度（list）
    }
    或多条：
    {
        "wavelength": [400, 401, ..., 700],
        "intensity" : [
            [2, 1, ..., 234],                      # 第1次测量
            [3, 2, ..., 198],                      # 第2次测量
            ...
        ]
    }

输出（result 字段），每条光谱返回：
    {
        "peak_wavelength" : 532.0,      # 最高峰波长（nm）
        "peak_intensity"  : 0.95,       # 最高峰强度
        "fwhm"            : 15.3,       # 半高宽 FWHM（nm）
        "peak_area"       : 123.4,      # 峰面积（梯形积分）
        "baseline"        : 0.05        # 估算基线（首尾均值）
    }
"""

import numpy as np
from typing import Any, List, Union

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from software.algorithms.base import BaseAlgorithm


class SpectrumAnalysis(BaseAlgorithm):
    """光谱数据分析：最高峰、半高宽（FWHM）、峰面积"""

    name = "spectrum_analysis"
    chinese_name = "光谱分析"
    description = "光谱数据分析：检测最高峰波长/强度、计算半高宽（FWHM）和峰面积（梯形积分）"
    params_schema = {
        "subtract_baseline": {
            "type": "bool",
            "description": "是否在计算前扣除基线（取首尾强度均值作为基线），默认 True",
            "default": True,
            "required": False,
        },
        "integration_range": {
            "type": "list",
            "description": "计算峰面积的波长范围 [start_nm, end_nm]，留空则对全谱积分",
            "default": None,
            "required": False,
        },
    }

    def run(self, data: Any, params: dict = None) -> dict:
        """
        执行光谱分析

        Args:
            data  : 含 'wavelength' 和 'intensity' 的 dict
            params: {
                "subtract_baseline": bool,
                "integration_range": [float, float] | None
            }

        Returns:
            统一格式 dict
        """
        params = params or {}

        if not isinstance(data, dict):
            return self._build_error("数据必须为 dict，包含 'wavelength' 和 'intensity' 键")

        if "wavelength" not in data or "intensity" not in data:
            return self._build_error("数据缺少 'wavelength' 或 'intensity' 键")

        wavelength = np.array(data["wavelength"], dtype=float)
        raw_intensity = data["intensity"]

        # 支持单条（list of numbers）或多条（list of lists）
        if isinstance(raw_intensity[0], (list, tuple)):
            intensity_rows = [np.array(row, dtype=float) for row in raw_intensity]
        else:
            intensity_rows = [np.array(raw_intensity, dtype=float)]

        results = []
        for i, intensity in enumerate(intensity_rows):
            if len(intensity) != len(wavelength):
                results.append({"index": i, "error": f"强度列长度 {len(intensity)} 与波长列 {len(wavelength)} 不匹配"})
                continue
            r = self._analyze(wavelength, intensity, params)
            r["index"] = i
            results.append(r)

        if len(results) == 1:
            return self._build_success(results[0], "光谱分析完成")
        else:
            return self._build_success(
                {"spectra": results, "count": len(results)},
                f"已分析 {len(results)} 条光谱"
            )

    # ------------------------------------------------------------------
    # 单条光谱分析
    # ------------------------------------------------------------------

    def _analyze(self, wavelength: np.ndarray, intensity: np.ndarray, params: dict) -> dict:
        subtract = params.get("subtract_baseline", True)
        integration_range = params.get("integration_range")

        # 估算并可选扣除基线（首尾均值）
        baseline = float(np.mean([intensity[0], intensity[-1]]))
        if subtract:
            signal = np.maximum(intensity - baseline, 0.0)
        else:
            signal = intensity.copy()
            baseline = 0.0

        # 最高峰
        peak_idx = int(np.argmax(signal))
        peak_wavelength = float(wavelength[peak_idx])
        peak_intensity  = float(signal[peak_idx])

        # 半高宽（FWHM）
        fwhm = self._calc_fwhm(wavelength, signal, peak_idx)

        # 峰面积（梯形积分）
        area = self._integrate(wavelength, signal, integration_range)

        return {
            "peak_wavelength": round(peak_wavelength, 4),
            "peak_intensity" : round(peak_intensity,  6),
            "fwhm"           : round(fwhm,            4),
            "peak_area"      : round(area,             4),
            "baseline"       : round(baseline,         6),
        }

    def _calc_fwhm(self, wavelength: np.ndarray, signal: np.ndarray, peak_idx: int) -> float:
        """计算半高宽（插值精度）"""
        half_max = signal[peak_idx] / 2.0
        n = len(signal)

        # 左侧：从峰值向左找第一个低于半高的点，线性插值
        left_wl = float(wavelength[0])
        for i in range(peak_idx, 0, -1):
            if signal[i - 1] <= half_max <= signal[i]:
                # 线性插值
                frac = (half_max - signal[i - 1]) / (signal[i] - signal[i - 1] + 1e-12)
                left_wl = float(wavelength[i - 1] + frac * (wavelength[i] - wavelength[i - 1]))
                break

        # 右侧：从峰值向右找第一个低于半高的点，线性插值
        right_wl = float(wavelength[-1])
        for i in range(peak_idx, n - 1):
            if signal[i] >= half_max >= signal[i + 1]:
                frac = (signal[i] - half_max) / (signal[i] - signal[i + 1] + 1e-12)
                right_wl = float(wavelength[i] + frac * (wavelength[i + 1] - wavelength[i]))
                break

        return abs(right_wl - left_wl)

    def _integrate(self, wavelength: np.ndarray, signal: np.ndarray,
                   integration_range=None) -> float:
        """梯形积分"""
        if integration_range:
            lo, hi = float(integration_range[0]), float(integration_range[1])
            mask = (wavelength >= lo) & (wavelength <= hi)
            x, y = wavelength[mask], signal[mask]
        else:
            x, y = wavelength, signal
        return float(np.trapz(y, x)) if len(x) >= 2 else 0.0


# ==============================================================================
# 测试接口
# ==============================================================================

if __name__ == "__main__":
    import json

    algo = SpectrumAnalysis()
    print(f"算法信息: {algo.get_info()}\n")

    # 构造模拟光谱（高斯峰 @ 532 nm + 基线 0.05）
    wl = list(range(400, 701))                          # 400~700 nm，步长 1
    import math
    intensity = [
        0.05 + 0.9 * math.exp(-0.5 * ((w - 532) / 15) ** 2)
        for w in wl
    ]

    # 示例1：单条光谱
    r1 = algo.run(
        data={"wavelength": wl, "intensity": intensity},
        params={"subtract_baseline": True}
    )
    print("示例1 - 单条光谱分析:")
    print(json.dumps(r1, indent=2, ensure_ascii=False))

    # 示例2：多条光谱
    intensity2 = [0.03 + 0.6 * math.exp(-0.5 * ((w - 650) / 20) ** 2) for w in wl]
    r2 = algo.run(
        data={"wavelength": wl, "intensity": [intensity, intensity2]},
        params={"subtract_baseline": True, "integration_range": [500, 600]}
    )
    print("\n示例2 - 多条光谱 + 区间积分:")
    print(json.dumps(r2, indent=2, ensure_ascii=False))
