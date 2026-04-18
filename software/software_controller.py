"""
软件功能控制器 (software/software_controller.py)
================================================

动态加载并管理 software/algorithms/ 下所有算法，提供统一的调用入口。

设计特点：
    - 自动扫描 default/ 和 extra_algorithms_fromProjects/ 目录
    - 凡是继承了 BaseAlgorithm 且设置了 name 的类，均会被自动注册
    - 新增算法只需将文件放入上述目录，无需修改此文件

对外接口：
    controller = SoftwareController()
    controller.list_algorithms()                         # 列出所有可用算法
    controller.run_algorithm("data_statistics", data)    # 运行指定算法
    controller.get_algorithm_info("spectrum_analysis")   # 获取算法元数据
"""

import importlib
import pkgutil
import traceback
from pathlib import Path
from typing import Any, Optional

from software.algorithms.base import BaseAlgorithm

# 扫描的算法子包目录名（相对于 software/algorithms/）
_ALGORITHM_PACKAGES = [
    "default",
    "extra_algorithms_fromProjects",
]


class SoftwareController:
    """
    软件算法控制器

    职责：
        - 启动时自动发现并注册所有 BaseAlgorithm 子类
        - 提供算法列表查询
        - 提供算法调用入口
        - 记录加载失败的模块（供调试）

    使用示例::

        ctrl = SoftwareController()

        # 查询可用算法
        print(ctrl.list_algorithms())

        # 运行算法
        result = ctrl.run_algorithm(
            name="data_statistics",
            data={"temperature": [100, 150, 200]},
            params={"include_correlation": False}
        )
        print(result)
    """

    def __init__(self):
        self._registry: dict[str, BaseAlgorithm] = {}
        self._load_errors: list[str] = []
        self._discover_algorithms()

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def list_algorithms(self) -> list[dict]:
        """
        返回所有已注册算法的元数据列表

        Returns:
            [{"name": ..., "description": ..., "params_schema": ...}, ...]
        """
        return [algo.get_info() for algo in self._registry.values()]

    def run_algorithm(self, name: str, data: Any, params: dict = None) -> dict:
        """
        运行指定算法

        Args:
            name  : 算法唯一标识（与算法类的 name 属性对应）
            data  : 输入数据（格式由各算法定义）
            params: 算法参数字典（可选）

        Returns:
            算法标准返回格式：
            {"success": bool, "algorithm": str, "result": Any, "message": str}
        """
        if name not in self._registry:
            available = ", ".join(self._registry.keys()) or "（无）"
            return {
                "success"  : False,
                "algorithm": name,
                "result"   : None,
                "message"  : f"未找到算法 '{name}'，可用算法：{available}",
            }

        try:
            return self._registry[name].run(data, params)
        except Exception as e:
            return {
                "success"  : False,
                "algorithm": name,
                "result"   : None,
                "message"  : f"算法 '{name}' 运行时异常: {str(e)}",
            }

    def get_algorithm_info(self, name: str) -> Optional[dict]:
        """
        获取单个算法的元数据

        Args:
            name: 算法标识

        Returns:
            元数据 dict 或 None（未找到时）
        """
        algo = self._registry.get(name)
        return algo.get_info() if algo else None

    def get_load_errors(self) -> list[str]:
        """返回算法加载时的错误信息（调试用）"""
        return list(self._load_errors)

    def generate_algorithm(self, user_description: str) -> dict:
        """
        使用 LLM 根据用户自然语言描述自动生成新算法

        生成完成后会自动重新加载算法注册表。

        Args:
            user_description: 用户对算法功能的自然语言描述

        Returns:
            {"success": bool, "name": str, "filepath": str, "spec": dict, "message": str}
        """
        try:
            from software.algorithms.extra_algorithms_fromProjects.prompt_template import (
                generate_algorithm as _generate,
            )
            result = _generate(user_description, verbose=False)

            # 如果生成成功，重新加载算法
            if result.get("success"):
                self._registry.clear()
                self._load_errors.clear()
                self._discover_algorithms()
                result["message"] += f"\n算法已自动注册，当前共有 {len(self._registry)} 个算法可用。"

            return result
        except Exception as e:
            return {
                "success" : False,
                "name"    : "",
                "filepath": "",
                "spec"    : {},
                "message" : f"算法生成失败: {str(e)}",
            }

    # ------------------------------------------------------------------
    # 内部：自动发现与注册算法
    # ------------------------------------------------------------------

    def _discover_algorithms(self):
        """
        扫描 software/algorithms/{package}/ 下的所有 .py 文件，
        自动导入并注册所有 BaseAlgorithm 子类。
        """
        algo_root = Path(__file__).parent / "algorithms"

        for pkg_name in _ALGORITHM_PACKAGES:
            pkg_dir = algo_root / pkg_name
            if not pkg_dir.is_dir():
                continue

            for finder, module_name, is_pkg in pkgutil.iter_modules([str(pkg_dir)]):
                full_module_path = f"software.algorithms.{pkg_name}.{module_name}"
                self._try_load_module(full_module_path)

    def _try_load_module(self, module_path: str):
        """尝试导入一个模块并注册其中的算法类"""
        try:
            module = importlib.import_module(module_path)
        except Exception as e:
            msg = f"Failed to load module {module_path}: {e}\n{traceback.format_exc()}"
            self._load_errors.append(msg)
            print(f"[SoftwareController] [WARN] {msg}")
            return

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if not isinstance(attr, type):
                continue
            if not issubclass(attr, BaseAlgorithm):
                continue
            if attr is BaseAlgorithm:
                continue

            instance = attr()
            if not instance.name:
                # 跳过 name 为空的抽象/模板类（如 MyAlgorithm 模板）
                continue

            if instance.name in self._registry:
                print(f"[SoftwareController] [WARN] Algorithm name conflict: '{instance.name}' "
                      f"already registered, skipping {module_path}.{attr_name}")
                continue

            self._registry[instance.name] = instance
            print(f"[SoftwareController] [OK] Registered algorithm: {instance.name} ({module_path})")


# ==============================================================================
# 测试接口
# ==============================================================================

if __name__ == "__main__":
    import json

    ctrl = SoftwareController()

    print("\n" + "=" * 50)
    print("已注册算法列表:")
    print("=" * 50)
    for info in ctrl.list_algorithms():
        print(f"  [{info['name']}] {info['description']}")

    if ctrl.get_load_errors():
        print("\n加载错误:")
        for err in ctrl.get_load_errors():
            print(err)

    # 测试 data_statistics
    print("\n" + "=" * 50)
    print("测试 data_statistics:")
    result = ctrl.run_algorithm(
        "data_statistics",
        data={"PCE": [15.2, 16.1, 17.3, 18.0, 16.8], "thickness": [100, 120, 150, 180, 130]},
        params={"include_correlation": True},
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 测试 data_normalization
    print("\n" + "=" * 50)
    print("测试 data_normalization:")
    result = ctrl.run_algorithm(
        "data_normalization",
        data=[15.2, 16.1, 17.3, 18.0, 16.8],
        params={"method": "minmax"},
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 测试 spectrum_analysis
    print("\n" + "=" * 50)
    print("测试 spectrum_analysis:")
    import math
    wl = list(range(400, 701))
    intensity = [0.05 + 0.9 * math.exp(-0.5 * ((w - 532) / 15) ** 2) for w in wl]
    result = ctrl.run_algorithm(
        "spectrum_analysis",
        data={"wavelength": wl, "intensity": intensity},
        params={"subtract_baseline": True},
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 测试不存在的算法
    print("\n" + "=" * 50)
    print("测试不存在的算法:")
    result = ctrl.run_algorithm("nonexistent_algo", data=[])
    print(json.dumps(result, indent=2, ensure_ascii=False))
