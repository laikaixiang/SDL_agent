# 实验编译器测试

本目录包含实验JSON编译器的测试脚本和示例文件。

## 文件说明

- `test_experiment_compiler.py` - 主测试脚本
- `sample_experiment.json` - 完整示例实验（包含循环、条件、用户输入）
- `simple_experiment.json` - 简单示例实验（仅循环和等待）

## 使用方法

### 1. 直接运行 experiment_manager.py

```bash
cd D:/PycharmProjects/SDL_agent
python core/experiment_manager.py
```

这会运行内置的测试代码，展示编译器功能。

### 2. 使用测试脚本编译JSON文件

```bash
# 基本用法：编译并显示代码
python test/test_experiment_compiler.py test/simple_experiment.json

# 编译并保存到文件
python test/test_experiment_compiler.py test/sample_experiment.json output.py

# 交互式选择操作
python test/test_experiment_compiler.py test/sample_experiment.json
# 然后选择：
#   1. 仅显示代码
#   2. 保存代码到文件
#   3. 编译并运行代码
#   4. 退出
```

### 3. 创建自定义实验JSON

实验JSON格式：

```json
{
  "experiment_name": "实验名称",
  "steps": [
    {
      "type": "helper",
      "name": "LOOP",
      "params": {"iterations": 3},
      "description": "循环3次"
    },
    {
      "type": "helper",
      "name": "WAIT",
      "params": {"duration": 1000},
      "description": "等待1秒"
    },
    {
      "type": "helper",
      "name": "END",
      "params": {},
      "description": "结束循环"
    }
  ]
}
```

## 支持的步骤类型

### 辅助函数 (type: "helper")

| 名称 | 参数 | 说明 | 编译结果 |
|------|------|------|----------|
| LOOP | `iterations`: 循环次数 | 循环执行 | `for _loop_iter in range(n):` |
| GROUP | `name`: 组名 | 步骤组（单次循环） | `for _group_iter in range(1):` |
| CONDITION | `condition`: 条件表达式 | 条件判断 | `if condition:` |
| WAIT | `duration`: 毫秒数 | 等待 | `time.sleep(seconds)` |
| USER_INPUT | `prompt`: 提示文字<br>`variable_name`: 变量名 | 用户输入 | `user_vars['var'] = input('prompt: ')` |
| END | 无 | 结束最近的循环/条件/组 | 自动闭合代码块 |

### 硬件工具 (type: "tool")

```json
{
  "type": "tool",
  "name": "set_temperature",
  "params": {"temperature": 150},
  "description": "设置温度"
}
```

编译为：
```python
print('执行硬件操作: set_temperature')
# TODO: 调用硬件函数 set_temperature({'temperature': 150})
```

### 软件算法 (type: "software")

```json
{
  "type": "software",
  "name": "data_normalization",
  "params": {},
  "input_file": "data.csv",
  "output_file": "result.csv",
  "description": "数据归一化"
}
```

编译为：
```python
print('执行算法: data_normalization')
# TODO: 调用算法 data_normalization
# 输入文件: data.csv
# 输出文件: result.csv
```

## 示例输出

运行 `python test/test_experiment_compiler.py test/simple_experiment.json` 会生成：

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

## 注意事项

1. **END标记**: 必须与对应的 LOOP/GROUP/CONDITION 配对使用
2. **嵌套结构**: 支持任意层级的嵌套，编译器会自动处理缩进
3. **用户输入**: USER_INPUT 生成的变量存储在 `user_vars` 字典中
4. **条件表达式**: CONDITION 的 `condition` 参数应为有效的Python表达式
5. **硬件/算法调用**: 当前生成的是占位符代码，需要手动实现具体调用逻辑

## 故障排除

### 问题：JSON格式错误

```
❌ 错误: JSON格式错误 - Expecting ',' delimiter: line 5 column 3
```

**解决**: 检查JSON文件格式，确保所有字段正确闭合，逗号使用正确。

### 问题：文件不存在

```
❌ 错误: 文件不存在 - test/my_experiment.json
```

**解决**: 检查文件路径是否正确，使用相对路径或绝对路径。

### 问题：执行超时

```
❌ 执行失败
错误: 执行超时（超过5分钟）
```

**解决**: 检查实验中是否有过长的等待时间或无限循环。

## 扩展开发

如需添加新的步骤类型，修改 `core/experiment_manager.py` 中的 `compile_to_python()` 方法：

```python
elif step_name == "MY_NEW_STEP":
    # 添加自定义编译逻辑
    code_lines.append(f"{indent}# 自定义步骤")
```
