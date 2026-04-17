# Dialogue Data ：对话的历史记录

## 概述

`dialogue data` 文件夹用于存储每次运行应用时的对话会话数据。每次启动 `app.py` 时，系统会自动创建一个以时间戳命名的会话文件夹，所有该次会话产生的数据都会保存在对应的文件夹中。

## 文件夹结构

```
dialogue data/
├── 20260417_152030/          # 会话时间戳文件夹（格式：YYYYMMDD_HHMMSS）
│   ├── extract/              # 文献提取结果（带时间戳的存档文件）
│   │   └── fapbi3_passivator_20260417-153045.csv
│   ├── temporal/             # 临时工作文件
│   │   └── extraction.csv    # 最新提取结果（会被覆盖）
│   ├── results/              # 数据分析结果
│   │   ├── analysis_data_statistics.json          # 最新分析结果
│   │   └── analysis_data_statistics_20260417-154030.json  # 存档
│   └── experiment_designs/   # 实验设计文件
│       └── 旋涂实验_v1_2026-04-17T15-30-00.json
├── 20260417_160000/          # 另一个会话
│   ├── extract/
│   ├── temporal/
│   ├── results/
│   └── experiment_designs/
└── README.md                 # 本说明文件
```

## 会话时间戳

### 生成规则

- **时间戳格式**: `YYYYMMDD_HHMMSS`
- **生成时机**: 每次运行 `python app.py` 时自动生成
- **唯一性**: 每次启动应用都会创建新的时间戳文件夹

### 示例

```
20260417_152030  →  2026年4月17日 15:20:30 启动的会话
20260417_160000  →  2026年4月17日 16:00:00 启动的会话
```

## 子文件夹说明

### 1. extract/ - 文献提取结果

**用途**: 存储从PDF文献中提取的数据（CSV格式）

**文件命名规则**: `{任务描述前缀}_{时间戳}.csv`

**示例**:
```
fapbi3_passivator_20260417-153045.csv
experimental_parameters_20260417-154030.csv
```

**特点**:
- 每次提取任务都会生成一个新的带时间戳的文件
- 文件永久保存，不会被覆盖
- 可用于历史数据追溯和对比

### 2. temporal/ - 临时工作文件

**用途**: 存储当前会话的最新工作数据

**主要文件**: `extraction.csv`

**特点**:
- 固定文件名，每次提取任务会覆盖
- 用于快速访问最新数据
- 数据分析默认使用此文件

**使用场景**:
```python
# 数据分析时默认读取
csv_path = "dialogue data/20260417_152030/temporal/extraction.csv"
```

### 3. results/ - 数据分析结果

**用途**: 存储算法分析结果（JSON格式）

**文件类型**:
1. **最新结果**: `analysis_{算法名称}.json` - 会被覆盖
2. **存档结果**: `analysis_{算法名称}_{时间戳}.json` - 永久保存

**示例**:
```
analysis_data_statistics.json                    # 最新统计分析结果
analysis_data_statistics_20260417-154030.json    # 存档
analysis_spectrum_analysis.json                  # 最新光谱分析结果
analysis_spectrum_analysis_20260417-155000.json  # 存档
```

**JSON结构**:
```json
{
  "algorithm": "data_statistics",
  "file_path": "dialogue data/20260417_152030/temporal/extraction.csv",
  "params": {"include_correlation": true},
  "timestamp": "20260417-154030",
  "result": {
    "mean": 15.68,
    "std": 2.34,
    ...
  },
  "success": true
}
```

### 4. experiment_designs/ - 实验设计文件

**用途**: 存储实验设计方案（JSON格式）

**文件命名规则**: `{实验名称}_{创建时间}.json`

**示例**:
```
旋涂实验_v1_2026-04-17T15-30-00.json
多步实验流程_2026-04-17T16-00-00.json
```

**JSON结构**:
```json
{
  "experiment_name": "旋涂实验_v1",
  "created_at": "2026-04-17T15:30:00.000Z",
  "steps": [
    {
      "type": "tool",
      "name": "spin_coating",
      "description": "旋涂工具",
      "params": {
        "reagent": "PbI2",
        "volume": 100,
        "spin_speed": 3000,
        "spin_acc": 1000,
        "spin_dur": 30000
      }
    },
    {
      "type": "helper",
      "name": "WAIT",
      "description": "等待",
      "params": {
        "duration": 5000
      }
    }
  ]
}
```

## 使用方式

### 1. 自动管理（推荐）

应用会自动管理会话路径，无需手动干预：

```python
# 启动应用时自动创建会话文件夹
python app.py

# 所有数据自动保存到当前会话文件夹
# 例如：dialogue data/20260417_152030/
```

### 2. 手动访问

如需访问历史会话数据：

```python
# 查看所有会话
import os
sessions = os.listdir("dialogue data")
print(sessions)  # ['20260417_152030', '20260417_160000', ...]

# 读取特定会话的数据
import pandas as pd
df = pd.read_csv("dialogue data/20260417_152030/temporal/extraction.csv")
```

### 3. API 访问

通过 API 获取当前会话路径：

```javascript
// 前端 JavaScript
const response = await fetch('/api/get_session_path?subdir=temporal');
const data = await response.json();
console.log(data.path);  // "dialogue data/20260417_152030/temporal"
console.log(data.timestamp);  // "20260417_152030"
```

## 数据流转示例

### 完整工作流程

```
1. 启动应用
   python app.py
   → 创建会话文件夹: dialogue data/20260417_152030/

2. 上传PDF文献
   → 保存到: pdf_library/

3. 执行文献提取
   → 生成: dialogue data/20260417_152030/extract/fapbi3_passivator_20260417-153045.csv
   → 生成: dialogue data/20260417_152030/temporal/extraction.csv

4. 执行数据分析
   → 读取: dialogue data/20260417_152030/temporal/extraction.csv
   → 生成: dialogue data/20260417_152030/results/analysis_data_statistics.json
   → 生成: dialogue data/20260417_152030/results/analysis_data_statistics_20260417-154030.json

5. 设计实验
   → 生成: dialogue data/20260417_152030/experiment_designs/旋涂实验_v1_2026-04-17T15-30-00.json
```

## 数据管理建议

### 清理策略

1. **定期清理**: 建议定期清理旧的会话文件夹以节省空间
2. **保留重要数据**: 将重要的实验结果和分析数据备份到其他位置
3. **命名规范**: 导出重要实验设计时使用有意义的名称

### 备份建议

```bash
# 备份整个会话
cp -r "dialogue data/20260417_152030" "backups/important_session_20260417"

# 只备份实验设计
cp "dialogue data/20260417_152030/experiment_designs/"*.json "backups/experiments/"

# 只备份分析结果
cp "dialogue data/20260417_152030/results/"*.json "backups/analysis/"
```

### 空间管理

```bash
# 查看各会话占用空间
du -sh "dialogue data"/*

# 删除旧会话（谨慎操作）
rm -rf "dialogue data/20260417_152030"
```

## 注意事项

1. **不要手动修改时间戳文件夹名称** - 这可能导致应用无法正确访问数据
2. **不要删除正在使用的会话文件夹** - 等应用关闭后再清理
3. **temporal/extraction.csv 会被覆盖** - 如需保留，请复制到 extract/ 目录
4. **实验设计文件可以跨会话使用** - 可以手动复制到新会话的 experiment_designs/ 目录

## 故障排查

### 问题：找不到数据文件

**原因**: 可能在错误的会话文件夹中查找

**解决**:
```python
# 检查当前会话时间戳
# 查看应用启动日志：
# [会话管理] 应用启动，会话时间戳: 20260417_152030
# [会话管理] 数据保存路径: dialogue data/20260417_152030
```

### 问题：数据被覆盖

**原因**: temporal/ 目录中的文件会被覆盖

**解决**: 重要数据应从 extract/ 或 results/ 目录获取，这些文件带时间戳不会被覆盖

### 问题：磁盘空间不足

**原因**: 会话文件夹累积过多

**解决**: 定期清理旧的会话文件夹，或将重要数据迁移到外部存储

## 技术实现

### 会话时间戳生成

```python
from datetime import datetime

SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_BASE_PATH = os.path.join("dialogue data", SESSION_TIMESTAMP)
```

### 路径获取函数

```python
def get_session_path(subdir=""):
    """获取当前会话的数据路径"""
    if subdir:
        return os.path.join(SESSION_BASE_PATH, subdir)
    return SESSION_BASE_PATH
```

### 使用示例

```python
# 获取 temporal 路径
temporal_path = get_session_path("temporal")
# 返回: "dialogue data/20260417_152030/temporal"

# 获取 extract 路径
extract_path = get_session_path("extract")
# 返回: "dialogue data/20260417_152030/extract"
```

## 版本历史

- **v1.0** (2026-04-17): 初始版本，实现基于时间戳的会话管理系统

## 相关文档

- [CLAUDE.md](../CLAUDE.md) - 项目整体说明
- [app.py](../app.py) - 应用入口和会话管理实现
- [core/extraction_engine.py](../core/extraction_engine.py) - 提取引擎实现
- [core/software_manager.py](../core/software_manager.py) - 软件管理器实现
