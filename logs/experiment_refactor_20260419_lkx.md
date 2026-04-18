# 实验模块重构日志

**日期**: 2026-04-19  
**作者**: laikaixiang  
**版本**: v1.0

## 重构目标

将 `core/experiment_manager.py` 按职责拆分为独立模块，提高代码可维护性和可测试性。

## 变更内容

### 1. 新增模块

#### experiment/ 目录（新建）
- `experiment/executor.py` - 实验执行器（273-677行）
  - 执行JSON格式的实验方案
  - 调用硬件工具和数据分析算法
  - 实验方案验证
  - 实时进度反馈

- `experiment/compiler.py` - 实验编译器（679-855行）
  - 将实验JSON编译为Python代码
  - 支持控制流（LOOP/CONDITION/WAIT等）
  - 编译并执行代码

- `experiment/format.py` - 格式转换器（61-271行）
  - JSON ↔ Visual双向转换
  - 拓扑排序确定执行顺序

- `experiment/agent.py` - 实验代理（从core移动）
  - AI生成实验方案
  - 基于PydanticAI

- `experiment/__init__.py` - 模块导出
- `experiment/README.md` - 模块文档
- `experiment/COMPILER.md` - 编译器原理文档（从core移动）

#### utils/ 目录（新建）
- `utils/csv_writer.py` - CSV写入工具（从core移动）
- `utils/pdf_to_markdown.py` - PDF转Markdown工具（从core移动）
- `utils/__init__.py` - 工具模块导出

### 2. 修改文件

#### app.py
**导入变更**:
```python
# 修改前
from core import CSVWriter, ExperimentManager
from core.experiment_manager import ExperimentManager

# 修改后
from utils import CSVWriter
from experiment.format import ExperimentFormatConverter
from experiment.executor import ExperimentExecutor
from experiment.compiler import ExperimentCompiler
```

**使用变更**:
```python
# /api/experiment_chat - 格式转换
converter = ExperimentFormatConverter()
visual_data = converter.json_to_visual(result)

# /api/execute_experiment_design - 实验执行
executor = ExperimentExecutor(software_manager=software_manager)
result = executor.execute_plan(data, progress_callback=on_progress)

# /api/compile_experiment - 编译
compiler = ExperimentCompiler()
python_code = compiler.compile_to_python(experiment_json)

# /api/compile_and_run_experiment - 编译并运行
compiler = ExperimentCompiler()
result = compiler.compile_and_run(experiment_json)
```

#### core/__init__.py
**移除导出**:
- `CSVWriter` - 移至 utils
- `ExperimentManager` - 已拆分，不再导出

### 3. 保留文件

#### core/ 目录（保留）
- `core/field_inference.py` - 字段推断（公用模块）
- `core/experiment_agent.py` - 保留副本（向后兼容）
- `core/experiment_manager.py` - 保留原文件（向后兼容）
- 其他核心模块不变

## 目录结构对比

### 重构前
```
SDL_agent/
├── core/
│   ├── experiment_manager.py  # 901行，职责混杂
│   ├── experiment_agent.py
│   ├── csv_writer.py
│   ├── pdf_to_markdown.py
│   └── ...
├── hardware/
├── software/
└── templates/
```

### 重构后
```
SDL_agent/
├── core/                      # 核心基础设施
│   ├── config.py
│   ├── llm_client.py
│   ├── field_inference.py    # 公用模块
│   ├── experiment_agent.py   # 保留副本
│   ├── experiment_manager.py # 保留原文件
│   └── ...
│
├── experiment/               # 实验模块（新建）
│   ├── __init__.py
│   ├── executor.py          # 执行器
│   ├── compiler.py          # 编译器
│   ├── format.py            # 格式转换器
│   ├── agent.py             # 实验代理
│   ├── README.md
│   └── COMPILER.md
│
├── utils/                    # 工具模块（新建）
│   ├── __init__.py
│   ├── csv_writer.py
│   └── pdf_to_markdown.py
│
├── hardware/
├── software/
└── templates/
```

## 优势

### 1. 单一职责
- 每个模块只做一件事，易于理解和维护
- `executor.py` - 执行
- `compiler.py` - 编译
- `format.py` - 格式转换

### 2. 独立测试
```python
# 测试格式转换（无需硬件）
converter = ExperimentFormatConverter()
visual = converter.json_to_visual(json_data)

# 测试编译器（无需硬件）
compiler = ExperimentCompiler()
code = compiler.compile_to_python(json_data)

# 测试执行器（需要mock硬件）
executor = ExperimentExecutor()
result = executor.execute_plan(json_data)
```

### 3. 灵活组合
```python
# 只需要格式转换
from experiment.format import ExperimentFormatConverter

# 只需要编译
from experiment.compiler import ExperimentCompiler

# 需要执行+数据分析
from experiment.executor import ExperimentExecutor
from software.software_manager import SoftwareManager
```

### 4. 易于扩展
- 新增编译目标（如C++、MATLAB）：只修改 `compiler.py`
- 新增硬件工具：只修改 `executor.py`
- 新增格式（如XML）：只修改 `format.py`

## 向后兼容

保留了以下文件以确保向后兼容：
- `core/experiment_manager.py` - 原文件保留
- `core/experiment_agent.py` - 原文件保留

如需使用旧接口，仍可导入：
```python
from core.experiment_manager import ExperimentManager
```

## 迁移指南

### 对于新代码
推荐使用新模块：
```python
from experiment import ExperimentExecutor, ExperimentCompiler, ExperimentFormatConverter
```

### 对于旧代码
可以继续使用旧接口，或逐步迁移：
```python
# 旧代码（仍可用）
from core.experiment_manager import ExperimentManager
manager = ExperimentManager()

# 新代码（推荐）
from experiment.executor import ExperimentExecutor
executor = ExperimentExecutor()
```

## 测试验证

### 文件结构验证
```bash
ls -la experiment/  # 检查experiment目录
ls -la utils/       # 检查utils目录
```

### 导入测试
```python
# 测试experiment模块
from experiment import ExperimentExecutor, ExperimentCompiler, ExperimentFormatConverter

# 测试utils模块
from utils import CSVWriter
```

## 相关文件

- `experiment/README.md` - 实验模块使用文档
- `experiment/COMPILER.md` - 编译器原理详解
- `CLAUDE.md` - 项目总体文档（需更新）

## 后续工作

1. 更新 `CLAUDE.md` 中的架构说明
2. 更新测试脚本中的导入路径
3. 考虑为 `experiment/` 模块添加单元测试
4. 考虑将 `hardware_controller.py` 移至 `hardware/` 目录
5. 考虑将 `software_manager.py` 移至 `software/` 目录

## 总结

本次重构成功将臃肿的 `experiment_manager.py`（901行）拆分为三个职责清晰的模块：
- `executor.py` - 执行和验证
- `compiler.py` - 编译
- `format.py` - 格式转换

同时创建了 `experiment/` 和 `utils/` 两个新的顶级模块，使项目结构更加清晰，符合领域驱动设计原则。
