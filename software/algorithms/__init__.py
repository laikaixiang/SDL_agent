"""
算法包 (software/algorithms/)
=============================

包含两类算法：
    default/                      - 内置算法，开箱即用
    extra_algorithms_fromProjects/ - 项目扩展算法，可使用 prompt_template.py 自动生成

所有算法继承 BaseAlgorithm（algorithms/base.py），共享统一调用接口：
    run(data, params=None) -> {"success": bool, "algorithm": str, "result": Any, "message": str}
"""
