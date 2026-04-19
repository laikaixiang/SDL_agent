# 实验模块

实验设计、执行和编译的核心模块。

## 子模块

### agent.py - 实验设计代理（已弃用）
- **状态**: 已弃用，保留用于参考
- **实现**: 基于PydanticAI Function Calling（方案1）
- **限制**: 需要模型支持OpenAI格式的Function Calling
- **当前使用**: 请使用 `core/field_inference.py:ExperimentDesignAgent`（方案2）

**方案对比**:
- **方案1（本文件）**: PydanticAI + Function Calling，支持交互式设计（读PDF、多轮对话）
- **方案2（推荐）**: JSON + 提示词，无需Function Calling，支持任何LLM

### executor.py - 实验执行器
- 执行JSON格式的实验方案
- 调用硬件工具（旋涂、温控、机械臂等）
- 调用数据分析算法
- 实验方案验证
- 实时进度反馈
- 支持三种步骤类型：
  - `type: "tool"` - 硬件操作
  - `type: "software"` - 数据分析算法
  - `type: "helper"` - 辅助操作（WAIT/LOOP/GROUP/CONDITION/END/USER_INPUT）

### compiler.py - 实验编译器
- 将实验JSON编译为Python代码
- 支持控制流（LOOP/CONDITION/WAIT等）
- 编译并执行代码
- 详细文档见 `COMPILER.md`

### format.py - 格式转换器
- JSON ↔ Visual双向转换
- 拓扑排序确定执行顺序
- 支持前端可视化编辑

## 使用示例

### 实验设计（方案2 - 推荐）

```python
from core.field_inference import ExperimentDesignAgent

# 创建实验设计代理
agent = ExperimentDesignAgent()

# 生成实验方案
success, experiment_json = agent.parse_experiment_design("设计一个旋涂实验")

if success:
    print(f"实验名称: {experiment_json['experiment_name']}")
    print(f"步骤数量: {len(experiment_json['steps'])}")
```

### 实验执行

```python
from experiment import ExperimentExecutor

# 执行实验
executor = ExperimentExecutor()
result = executor.execute_plan(experiment_json)

if result['success']:
    print("实验执行成功")
```

### 实验编译

```python
from experiment import ExperimentCompiler

# 编译为Python代码
compiler = ExperimentCompiler()
code = compiler.compile_to_python(experiment_json)
print(code)

# 编译并执行
output, errors = compiler.compile_and_run(experiment_json)
```

### 格式转换

```python
from experiment import ExperimentFormatConverter

# JSON → Visual（前端可视化）
converter = ExperimentFormatConverter()
visual = converter.json_to_visual(experiment_json)

# Visual → JSON
experiment_json = converter.visual_to_json(visual_data)
```

## 架构说明

### 模块拆分

原 `core/experiment_manager.py` 已拆分为三个独立模块：
- `experiment/executor.py` - 执行和验证
- `experiment/compiler.py` - 编译
- `experiment/format.py` - 格式转换

这样做的好处：
- 单一职责：每个模块只做一件事
- 独立测试：可以单独测试每个模块
- 灵活组合：按需导入所需模块

### 实验设计方案演进

**方案1（已弃用）** - `experiment/agent.py`:
- 基于PydanticAI的Function Calling实现
- 需要模型支持OpenAI格式的Function Calling
- 支持交互式设计（读PDF、多轮对话）
- 代码保留用于参考

**方案2（当前使用）** - `core/field_inference.py:ExperimentDesignAgent`:
- JSON + 提示词方式
- 提示词从注册表动态生成
- 无需Function Calling支持，任何LLM都可使用
- 注册表驱动：
  - 硬件工具: `hardware/tools/REGISTRY.json`
  - 软件算法: 通过 `SoftwareController` 动态加载
  - 辅助操作: 内置在 `ExperimentDesignAgent` 中

## 统一JSON格式

```json
{
  "experiment_name": "实验名称",
  "description": "实验描述",
  "steps": [
    {
      "type": "tool",
      "name": "spin_coating",
      "params": {
        "spin_speed": 3000,
        "spin_acc": 1000,
        "spin_dur": 30000,
        "reagent": "Perovskite",
        "volume": 10.0
      },
      "description": "旋涂钙钛矿溶液"
    },
    {
      "type": "helper",
      "name": "WAIT",
      "params": {"duration": 5000},
      "description": "等待5秒"
    },
    {
      "type": "software",
      "name": "spectrum_analysis",
      "params": {"subtract_baseline": true},
      "input_file": "data.csv",
      "output_file": "result.json",
      "description": "光谱分析"
    }
  ],
  "notes": "注意事项"
}
```

## 测试

- 执行器测试: 见 `test/compile_test/`
- 编译器测试: `python experiment/compiler.py`
- 方案2测试: 见 `test/experiment_design_v2/`

## 相关文档

- `CLAUDE.md` - 完整项目文档
- `experiment/COMPILER.md` - 编译器详细文档
- `test/experiment_design_v2/README.md` - 方案2测试说明
- `logs/experiment_design_refactor_20260420_lkx.md` - 重构日志

