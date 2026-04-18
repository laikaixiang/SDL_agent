# 实验模块

实验设计、执行和编译的核心模块。

## 子模块

### executor.py - 实验执行器
- 执行JSON格式的实验方案
- 调用硬件工具（旋涂、温控、机械臂等）
- 调用数据分析算法
- 实验方案验证
- 实时进度反馈

### compiler.py - 实验编译器
- 将实验JSON编译为Python代码
- 支持控制流（LOOP/CONDITION/WAIT等）
- 编译并执行代码

### format.py - 格式转换器
- JSON ↔ Visual双向转换
- 拓扑排序确定执行顺序

### agent.py - 实验代理
- AI生成实验方案
- 基于PydanticAI（legacy模式）

## 使用示例

```python
from experiment import ExperimentExecutor, ExperimentCompiler, ExperimentFormatConverter

# 执行实验
executor = ExperimentExecutor()
result = executor.execute_plan(experiment_json)

# 编译为Python代码
compiler = ExperimentCompiler()
code = compiler.compile_to_python(experiment_json)

# 格式转换
converter = ExperimentFormatConverter()
visual = converter.json_to_visual(experiment_json)
```

## 架构说明

原 `core/experiment_manager.py` 已拆分为三个独立模块：
- `experiment/executor.py` - 执行和验证
- `experiment/compiler.py` - 编译
- `experiment/format.py` - 格式转换

这样做的好处：
- 单一职责：每个模块只做一件事
- 独立测试：可以单独测试每个模块
- 灵活组合：按需导入所需模块
