"""
实验模块 - 实验设计、执行、编译

子模块：
- executor: 实验执行器（硬件调用、算法执行）
- compiler: 实验编译器（JSON → Python代码）
- format: 格式转换器（JSON ↔ Visual）
- agent: 实验代理（AI生成实验方案）
"""

from .executor import ExperimentExecutor
from .compiler import ExperimentCompiler
from .format import ExperimentFormatConverter

__all__ = [
    'ExperimentExecutor',
    'ExperimentCompiler',
    'ExperimentFormatConverter',
]
