"""
软件功能管理器 (core/software_manager.py)
=========================================

桥接 app.py 与 software 模块，提供以下功能：
    - 算法注册表查询（list_algorithms）
    - 算法执行（run_algorithm）
    - 从 temporal/extraction.csv 读取数据并执行算法（run_on_csv）
    - 调用 LLM 自动生成新算法（generate_algorithm）

设计原则：
    - app.py 只与 SoftwareManager 交互，不直接导入 software 子模块
    - SoftwareManager 在首次使用时懒加载 SoftwareController（避免启动时导入错误）
"""

import os
import csv
from typing import Any, Optional


class SoftwareManager:
    """
    软件模块管理器

    职责：
        - 封装 SoftwareController 的初始化与复用
        - 提供 CSV 数据读取与算法结合的快捷方法
        - 提供 LLM 算法生成功能

    使用示例::

        mgr = SoftwareManager()

        # 查询可用算法
        algos = mgr.list_algorithms()

        # 运行算法
        result = mgr.run_algorithm("data_statistics", data=[1, 2, 3])

        # 对 temporal/extraction.csv 的数值列做统计
        result = mgr.run_on_csv("data_statistics")

        # 用 LLM 自动生成新算法
        result = mgr.generate_algorithm("我需要一个高斯平滑算法...")
    """

    def __init__(self, temporal_dir: str = "temporal"):
        self._temporal_dir = temporal_dir
        self._controller = None  # 懒加载

    # ------------------------------------------------------------------
    # 对外接口：算法管理
    # ------------------------------------------------------------------

    def list_algorithms(self) -> list:
        """返回所有已注册算法的元数据列表"""
        return self._get_controller().list_algorithms()

    def get_algorithm_info(self, name: str) -> Optional[dict]:
        """获取单个算法的元数据，未找到时返回 None"""
        return self._get_controller().get_algorithm_info(name)

    def run_algorithm(self, name: str, data: Any, params: dict = None) -> dict:
        """
        运行指定算法

        Args:
            name  : 算法标识（如 'data_statistics'）
            data  : 输入数据（dict / list）
            params: 算法参数（可选）

        Returns:
            {"success": bool, "algorithm": str, "result": Any, "message": str}
        """
        return self._get_controller().run_algorithm(name, data, params)

    def run_on_csv(self, algorithm_name: str, params: dict = None) -> dict:
        """
        读取 temporal/extraction.csv 的数值列，作为数据运行算法

        Args:
            algorithm_name: 算法标识
            params        : 算法参数（可选）

        Returns:
            算法标准返回格式
        """
        csv_path = os.path.join(self._temporal_dir, "extraction.csv")
        if not os.path.exists(csv_path):
            return {
                "success"  : False,
                "algorithm": algorithm_name,
                "result"   : None,
                "message"  : f"CSV 文件不存在: {csv_path}，请先执行文献提取任务",
            }

        try:
            data = self._read_csv_as_columns(csv_path)
        except Exception as e:
            return {
                "success"  : False,
                "algorithm": algorithm_name,
                "result"   : None,
                "message"  : f"读取 CSV 文件失败: {str(e)}",
            }

        if not data:
            return {
                "success"  : False,
                "algorithm": algorithm_name,
                "result"   : None,
                "message"  : "CSV 文件中没有可解析的数值列",
            }

        return self._get_controller().run_algorithm(algorithm_name, data, params)

    # ------------------------------------------------------------------
    # 对外接口：算法生成
    # ------------------------------------------------------------------

    def generate_algorithm(self, user_description: str) -> dict:
        """
        使用 LLM 根据用户自然语言描述自动生成新算法

        生成完成后，调用 reload_algorithms() 即可立即使用新算法。

        Args:
            user_description: 用户对算法功能的自然语言描述

        Returns:
            {"success": bool, "name": str, "filepath": str, "spec": dict, "message": str}
        """
        try:
            from software.algorithms.extra_algorithms_fromProjects.prompt_template import (
                generate_algorithm as _generate,
            )
            return _generate(user_description, verbose=False)
        except Exception as e:
            return {
                "success" : False,
                "name"    : "",
                "filepath": "",
                "spec"    : {},
                "message" : f"算法生成失败: {str(e)}",
            }

    def reload_algorithms(self) -> list:
        """
        重新扫描并注册算法（生成新算法后调用）

        Returns:
            更新后的算法列表
        """
        from software.software_controller import SoftwareController
        self._controller = SoftwareController()
        return self._controller.list_algorithms()

    def get_load_errors(self) -> list:
        """返回算法加载时的错误信息（调试用）"""
        return self._get_controller().get_load_errors()

    def auto_analyze(self, csv_path: str, task_manager) -> None:
        """
        运行自动分析流水线，通过 task_manager 推送 SSE 进度

        流程：读取 CSV 列名 → LLM 选算法 → 读数据 → 执行算法 → 保存结果 → 推送 complete

        Args:
            csv_path    : CSV 文件路径（如 "temporal/extraction.csv"）
            task_manager: TaskManager 实例，用于 put_task_message
        """
        from software.auto_analyze import run_pipeline

        def send_msg(msg_type, data):
            task_manager.put_task_message(msg_type, data)

        try:
            run_pipeline(
                csv_path   = csv_path,
                send_msg   = send_msg,
                algorithms = self.list_algorithms(),
                run_fn     = self.run_algorithm,
            )
        except Exception as e:
            task_manager.put_task_message("complete", {"error": f"流水线异常: {str(e)}"})
        finally:
            task_manager.task_running = False

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _get_controller(self):
        """懒加载 SoftwareController（首次使用时初始化）"""
        if self._controller is None:
            from software.software_controller import SoftwareController
            self._controller = SoftwareController()
        return self._controller

    def _read_csv_as_columns(self, csv_path: str) -> dict:
        """
        读取 CSV 文件，返回各列的数值列表（跳过非数值列）

        Returns:
            {列名: [float, float, ...], ...}
        """
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            raw_columns: dict = {}
            for row in reader:
                for key, val in row.items():
                    raw_columns.setdefault(key, []).append(val)

        numeric_columns = {}
        for col, values in raw_columns.items():
            parsed = []
            for v in values:
                try:
                    parsed.append(float(v))
                except (ValueError, TypeError):
                    break
            else:
                if parsed:
                    numeric_columns[col] = parsed

        return numeric_columns


# ==============================================================================
# 测试接口
# ==============================================================================

if __name__ == "__main__":
    import json

    mgr = SoftwareManager()

    print("=" * 50)
    print("可用算法列表:")
    print("=" * 50)
    for info in mgr.list_algorithms():
        print(f"  [{info['name']}]  {info['description']}")

    print("\n" + "=" * 50)
    print("运行 data_normalization (zscore):")
    print("=" * 50)
    result = mgr.run_algorithm(
        "data_normalization",
        data=[10.5, 15.2, 18.7, 22.3, 16.8],
        params={"method": "zscore"},
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 50)
    print("运行 data_statistics (多列):")
    print("=" * 50)
    result = mgr.run_algorithm(
        "data_statistics",
        data={
            "PCE"      : [15.2, 16.1, 17.3, 18.0, 16.8],
            "thickness": [100, 120, 150, 180, 130],
        },
        params={"include_correlation": True},
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 50)
    print("运行 spectrum_analysis:")
    print("=" * 50)
    import math
    wl = list(range(400, 701))
    intensity = [0.05 + 0.9 * math.exp(-0.5 * ((w - 532) / 15) ** 2) for w in wl]
    result = mgr.run_algorithm(
        "spectrum_analysis",
        data={"wavelength": wl, "intensity": intensity},
        params={"subtract_baseline": True},
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
