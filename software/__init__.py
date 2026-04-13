"""
软件功能模块包 (software/)
==========================

提供与硬件控制无关的纯算法和数据处理能力。

核心组件：
    SoftwareController  - 算法注册表与统一调用入口（自动发现所有算法）

算法目录：
    algorithms/default/                      - 内置算法（开箱即用）
        data_statistics.py                   - 描述性统计分析
        data_normalization.py                - 数据归一化/标准化
        spectrum_analysis.py                 - 光谱峰值/FWHM/峰面积分析

    algorithms/extra_algorithms_fromProjects/ - 项目扩展算法（按需添加）
        prompt_template.py                   - 算法自动生成器（用 LLM 写算法）

================================================================
如何添加新算法（手动方式）：
================================================================

1. 在 algorithms/default/ 或 algorithms/extra_algorithms_fromProjects/ 中新建 .py 文件

2. 文件内容模板：

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from software.algorithms.base import BaseAlgorithm

    class MyNewAlgorithm(BaseAlgorithm):
        name        = "my_new_algorithm"        # 唯一标识，用于 API 路由
        description = "算法功能说明"
        params_schema = {
            "param": {"type": "float", "description": "参数说明", "default": 1.0}
        }

        def run(self, data, params=None):
            params = params or {}
            try:
                result = {"output": ...}
                return self._build_success(result, "完成")
            except Exception as e:
                return self._build_error(str(e))

    if __name__ == "__main__":
        import json
        r = MyNewAlgorithm().run(data=[1,2,3])
        print(json.dumps(r, indent=2, ensure_ascii=False))

3. 保存文件后，下次实例化 SoftwareController() 时自动注册，无需其他修改。

================================================================
如何使用 LLM 自动生成算法（推荐方式）：
================================================================

通过 Web 接口（描述需求，系统自动生成并保存代码）：
    POST /api/software/generate_algorithm
    Body: {"description": "我需要一个对光谱数据做高斯平滑的算法..."}

或在 Python 中直接调用：
    from software.algorithms.extra_algorithms_fromProjects.prompt_template import generate_algorithm
    result = generate_algorithm("我需要一个移动平均算法，输入是数值列表，窗口大小可配置")
"""

from .software_controller import SoftwareController

__all__ = ["SoftwareController"]
