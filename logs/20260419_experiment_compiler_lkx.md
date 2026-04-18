# 实验编译器功能开发日志

**日期**: 2026-04-19  
**修改人**: lkx  
**版本**: v1.0  

---

## 修改文件清单

| 文件路径 | 修改类型 | 说明 |
|---------|---------|------|
| `templates/static/js/analysis/algorithm_panel.js` | 修改 | 算法添加到实验设计时类型改为 `software` |
| `templates/index.html` | 新增 | 添加 END 和 USER_INPUT 辅助函数按钮 |
| `templates/static/js/experiment/experiment_design.js` | 新增 | 添加 END/USER_INPUT 模板，编译器前端函数 |
| `core/experiment_manager.py` | 新增 | 编译器核心功能、测试代码 |
| `app.py` | 新增 | 编译器API路由 |
| `test/compile_test/test_experiment_compiler.py` | 新增 | 模块化测试脚本 |
| `test/compile_test/sample_experiment.json` | 新增 | 完整示例实验JSON |
| `test/compile_test/simple_experiment.json` | 新增 | 简单示例实验JSON |
| `test/compile_test/README_COMPILER.md` | 新增 | 编译器使用文档 |

---

## 功能概述

### 1. 算法类型修正

**问题**: 算法面板添加算法到实验设计时，使用 `type: 'tool'`，导致无法显示算法特有字段（输入/输出文件）。

**解决**: 修改 `algorithm_panel.js:165` 行，将类型改为 `type: 'software'`。

**影响**: 算法在实验设计画布中正确显示为软件类型，支持输入/输出文件配置。

---

### 2. 新增辅助函数

#### END（结束点）🏁
- **功能**: 标志最近的 LOOP/GROUP/CONDITION 结束
- **参数**: 无
- **编译结果**: 自动闭合代码块

#### USER_INPUT（用户输入）✋
- **功能**: 运行时弹窗询问用户输入
- **参数**: 
  - `prompt`: 提示文字（默认："请输入参数"）
  - `variable_name`: 变量名（默认："user_value"）
- **编译结果**: `user_vars['var'] = input('prompt: ')`

**实现位置**:
- 前端: `templates/index.html` (按钮), `experiment_design.js` (模板)
- 后端: `core/experiment_manager.py` (执行函数)

---

### 3. 实验JSON编译器

#### 核心功能 (`core/experiment_manager.py`)

**`compile_to_python(experiment_json)`**
- 将实验JSON编译为可执行的Python代码
- 支持控制结构: LOOP, GROUP, CONDITION, WAIT, USER_INPUT, END
- 自动处理嵌套和缩进
- 使用栈结构跟踪代码块闭合

**`compile_and_run(experiment_json)`**
- 编译并执行实验
- 创建临时文件运行
- 捕获输出和错误
- 5分钟超时保护

#### 编译规则

| 步骤类型 | 编译结果 |
|---------|---------|
| LOOP | `for _loop_iter in range(n):` |
| GROUP | `for _group_iter in range(1):` |
| CONDITION | `if condition:` |
| WAIT | `time.sleep(seconds)` |
| USER_INPUT | `user_vars['var'] = input('prompt: ')` |
| END | 自动闭合最近的代码块 |
| tool | `print('执行硬件操作: ...')` + TODO注释 |
| software | `print('执行算法: ...')` + TODO注释 |

#### API路由 (`app.py`)

- **`POST /api/compile_experiment`** - 编译为Python代码
  - 请求: `{"experiment_json": {...}}`
  - 响应: `{"success": true, "code": "..."}`

- **`POST /api/compile_and_run_experiment`** - 编译并运行
  - 请求: `{"experiment_json": {...}}`
  - 响应: `{"success": true, "code": "...", "output": "...", "error": ""}`

#### 前端集成 (`experiment_design.js`)

新增工具栏按钮:
- 🔧 **编译代码** - `compileExperiment()` - 生成并显示Python代码
- ⚡ **编译并运行** - `compileAndRunExperiment()` - 生成代码并执行

---

### 4. 测试脚本

#### 文件结构
```
test/compile_test/
├── test_experiment_compiler.py  # 主测试脚本
├── sample_experiment.json       # 完整示例（循环/条件/输入）
├── simple_experiment.json       # 简单示例（循环/等待）
└── README_COMPILER.md           # 使用文档
```

#### 测试脚本功能

**`compile_experiment(json_file, output_py_file=None)`**
- 编译实验JSON为Python代码
- 自动生成输出文件名（如未指定）
- 返回: `(success, python_code, output_file)`

**`run_experiment(json_file)`**
- 编译并运行实验
- 返回执行结果字典

**测试代码** (`if __name__ == "__main__":`)
- 测试1: 编译简单实验，指定输出文件名
- 测试2: 编译复杂实验，自动生成输出文件名
- 测试3: 编译并运行简单实验

#### 使用方法

```bash
# 直接运行测试
cd test/compile_test
python test_experiment_compiler.py

# 在代码中使用
from test.compile_test.test_experiment_compiler import compile_experiment
success, code, output = compile_experiment("exp.json", "output.py")
```

---

## 示例输出

### 输入JSON
```json
{
  "experiment_name": "简单测试",
  "steps": [
    {"type": "helper", "name": "LOOP", "params": {"iterations": 2}},
    {"type": "helper", "name": "WAIT", "params": {"duration": 500}},
    {"type": "helper", "name": "END", "params": {}}
  ]
}
```

### 生成的Python代码
```python
# 自动生成的实验执行代码
import time

# 用户输入变量存储
user_vars = {}

def execute_experiment():
    # 循环2次
    for _loop_iter in range(2):
        # 等待0.5秒
        time.sleep(0.5)  # 等待 0.5 秒

if __name__ == '__main__':
    execute_experiment()
```

---

## 技术要点

### 1. 栈结构处理嵌套
使用栈 `stack = []` 跟踪嵌套结构，遇到 END 时自动弹出并减少缩进层级。

### 2. 用户变量存储
所有 USER_INPUT 的输入存储在 `user_vars` 字典中，供后续步骤使用。

### 3. 临时文件执行
使用 `tempfile.NamedTemporaryFile` 创建临时Python文件，执行后自动清理。

### 4. 超时保护
`subprocess.run(timeout=300)` 设置5分钟超时，防止无限循环。

---

## 注意事项

1. **END标记必须配对**: 每个 LOOP/GROUP/CONDITION 必须有对应的 END
2. **条件表达式**: CONDITION 的 `condition` 参数必须是有效的Python表达式
3. **硬件/算法调用**: 当前生成的是占位符代码，需要手动实现具体调用逻辑
4. **用户输入变量**: 通过 `user_vars['variable_name']` 访问用户输入的值
5. **GROUP实现**: 使用 `for i in range(1)` 实现单次循环，保持代码块结构

---

## 后续优化建议

1. **完整的硬件调用**: 将 TODO 注释替换为实际的硬件函数调用
2. **算法集成**: 实现 software 类型步骤的实际算法调用
3. **错误处理**: 添加更详细的错误处理和异常捕获
4. **变量作用域**: 支持更复杂的变量作用域和数据流
5. **可视化调试**: 添加步骤执行的可视化反馈
6. **代码优化**: 生成更优化的Python代码（如合并连续的等待）

---

## 测试验证

### 测试环境
- Python 3.10
- Windows 11
- Flask 开发服务器

### 测试结果
- ✅ 算法类型修正 - 通过
- ✅ END 辅助函数 - 通过
- ✅ USER_INPUT 辅助函数 - 通过
- ✅ 编译简单实验 - 通过
- ✅ 编译复杂实验 - 通过
- ✅ 编译并运行 - 通过
- ✅ 前端集成 - 通过

---

## 相关文档

- `test/compile_test/README_COMPILER.md` - 编译器详细使用文档
- `CLAUDE.md` - 项目整体文档
- `dialogue data/README.md` - 数据存储说明

---

**日志结束**
