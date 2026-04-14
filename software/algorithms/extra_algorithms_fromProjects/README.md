# 算法自动生成器使用指南

本目录用于存放用户自定义生成的算法文件。系统提供了基于大语言模型的算法自动生成功能，可以根据自然语言描述自动生成符合接口规范的算法代码。

## 📋 目录说明

- `prompt_template.py` - 算法生成核心模块，包含完整的生成流程
- `README.md` - 本文档
- `*.py` - 用户生成的自定义算法文件（自动保存到此目录）

## 🚀 快速开始

### 方式一：通过 Web 界面生成（推荐）

1. 在 Web 界面的聊天框中输入：
   ```
   生成算法：<你的算法描述>
   ```

2. 示例：
   ```
   生成算法：我需要一个对数值列表做移动平均的算法，窗口大小可配置，默认5，输出平滑后的序列和残差
   ```

3. 系统会自动：
   - 提取算法规格（名称、输入格式、输出字段、参数）
   - 生成完整的 Python 代码
   - 保存到当前目录
   - 更新算法注册表
   - 重新加载算法列表

4. 生成成功后，你可以立即在"数据分析"模式中使用该算法

### 方式二：直接调用 Python 模块

```python
from software.algorithms.extra_algorithms_fromProjects.prompt_template import generate_algorithm

# 描述你的算法需求
description = """
我需要一个对光谱数据做高斯平滑的算法，
输入是 dict 含有 wavelength 和 intensity 两个列表，
参数是窗口大小 window_size（整数，默认 5），
输出平滑后的强度序列 smoothed_intensity
"""

# 生成算法
result = generate_algorithm(description, verbose=True)

if result["success"]:
    print(f"✅ 算法已生成: {result['name']}")
    print(f"📁 文件路径: {result['filepath']}")
else:
    print(f"❌ 生成失败: {result['message']}")
```

### 方式三：交互式命令行测试

```bash
cd software/algorithms/extra_algorithms_fromProjects
python prompt_template.py
```

按照提示输入算法需求描述即可。

## 📝 如何描述算法需求

为了让 AI 准确理解你的需求，建议从以下几个方面进行描述：

### 1. 算法功能
这个算法要做什么？

**示例**：
- "对光谱数据做平滑处理"
- "计算数值列表的移动平均"
- "检测时间序列中的异常值"

### 2. 输入数据格式
数据长什么样？

**示例**：
- "输入是 dict，含有 'wavelength' 和 'intensity' 两个列表"
- "输入是数值列表 [1, 2, 3, ...]"
- "输入是 dict，每个键对应一列数据"

### 3. 期望输出
希望得到哪些结果？

**示例**：
- "输出平滑后的强度序列"
- "输出移动平均值和残差"
- "输出异常值的索引和对应的数值"

### 4. 可调参数
有哪些可配置的参数？

**示例**：
- "窗口大小 window_size，整数，默认 5"
- "平滑程度 sigma，浮点数，默认 1.0"
- "阈值 threshold，浮点数，默认 3.0"

## 📚 完整示例

### 示例 1：移动平均算法

**需求描述**：
```
我需要一个对数值列表做移动平均的算法，
输入是 dict 含有 'values' 列表，
参数是窗口大小 window_size（整数，默认 5），
输出平滑后的序列 smoothed 和每点的残差 residuals
```

**生成的算法可以这样使用**：
```python
from core import SoftwareManager

manager = SoftwareManager()
result = manager.run_algorithm(
    "moving_average",
    data={"values": [10, 12, 15, 14, 18, 20, 22, 19, 21, 23]},
    params={"window_size": 3}
)

print(result["result"]["smoothed"])    # 平滑后的序列
print(result["result"]["residuals"])   # 残差
```

### 示例 2：高斯平滑算法

**需求描述**：
```
我需要一个对光谱数据做高斯平滑的算法，
输入是 dict 含有 'wavelength' 和 'intensity' 两个列表，
参数是平滑窗口 window（整数，默认 5）和标准差 sigma（浮点数，默认 1.0），
输出平滑后的强度 smoothed_intensity
```

### 示例 3：峰值检测算法

**需求描述**：
```
我需要一个检测数值序列中峰值的算法，
输入是数值列表，
参数是最小峰高 min_height（浮点数，默认 0.5）和最小峰距 min_distance（整数，默认 10），
输出峰值的索引 peak_indices 和对应的峰值 peak_values
```

## 🔧 算法接口规范

所有生成的算法都必须遵循以下接口规范：

### 1. 继承 BaseAlgorithm

```python
from software.algorithms.base import BaseAlgorithm

class YourAlgorithm(BaseAlgorithm):
    name = "your_algorithm"
    description = "算法功能描述"
    params_schema = {
        "param_name": {
            "type": "int",
            "description": "参数说明",
            "default": 5,
            "required": False
        }
    }
```

### 2. 实现 run() 方法

```python
def run(self, data, params=None):
    params = params or {}
    try:
        # 算法核心逻辑
        result = {...}
        return self._build_success(result, "执行成功")
    except Exception as e:
        return self._build_error(f"算法执行失败: {str(e)}")
```

### 3. 返回值格式

成功时：
```python
{
    "success": True,
    "result": {...},        # 算法输出结果
    "message": "执行成功"
}
```

失败时：
```python
{
    "success": False,
    "result": None,
    "message": "错误信息"
}
```

## 🔍 生成后的验证

算法生成后，系统会自动：

1. ✅ 检查代码语法是否正确
2. ✅ 验证是否继承 BaseAlgorithm
3. ✅ 验证是否实现 run() 方法
4. ✅ 保存到当前目录
5. ✅ 更新算法注册表

你可以通过以下方式验证算法是否可用：

```python
from core import SoftwareManager

manager = SoftwareManager()

# 查看所有可用算法
algorithms = manager.list_algorithms()
print(algorithms)

# 测试新算法
result = manager.run_algorithm("your_algorithm", data={...}, params={...})
print(result)
```

## 📊 算法注册表

系统维护两个算法注册表：

1. **内置算法注册表**：`software/algorithms/default/REGISTRY.json`
   - 包含系统预置的算法（data_statistics、data_normalization、spectrum_analysis）
   - 由系统维护，用户不应修改

2. **自定义算法注册表**：`software/algorithms/extra_algorithms_fromProjects/REGISTRY.json`
   - 包含用户生成的算法
   - 每次生成新算法时自动更新

## ⚠️ 注意事项

1. **算法名称唯一性**：如果生成的算法名称与已有算法重复，新算法会覆盖旧算法
2. **依赖库限制**：生成的算法只能使用标准库、numpy、math，不支持其他第三方库
3. **代码安全性**：生成的代码会直接执行，请确保描述清晰，避免生成不安全的代码
4. **重新加载**：生成新算法后，需要重新实例化 SoftwareManager 或调用 reload_algorithms() 方法

## 🐛 故障排除

### 问题 1：生成的算法无法使用

**解决方案**：
```python
from core import SoftwareManager

manager = SoftwareManager()
manager.reload_algorithms()  # 重新加载算法
```

### 问题 2：算法执行报错

**解决方案**：
1. 检查输入数据格式是否符合算法要求
2. 检查参数类型和取值范围
3. 查看生成的算法文件，手动修正代码

### 问题 3：生成的算法不符合预期

**解决方案**：
1. 重新描述需求，提供更详细的信息
2. 手动编辑生成的算法文件
3. 参考 `software/algorithms/default/` 中的示例算法

## 📞 技术支持

如有问题，请：
1. 查看生成的算法文件源码
2. 参考内置算法示例（`software/algorithms/default/`）
3. 检查系统日志和错误信息

---

**版本**：1.0.0  
**最后更新**：2026-04-15
