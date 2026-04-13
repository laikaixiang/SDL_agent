# Software 模块

提供与硬件控制无关的纯软件算法和数据处理能力，可通过 Flask API 或 Python 直接调用。

## 目录结构

```
software/
├── __init__.py                             # 包入口，导出 SoftwareController
├── software_controller.py                 # 算法注册表与统一调用入口
├── readfile.py                             # CSV读取工具（LLM动态调用接口）
├── auto_analyze.py                         # 自动分析流水线（LLM选算法 + 执行）
├── README.md                              # 本文档
│
└── algorithms/
    ├── base.py                            # BaseAlgorithm 基类（统一接口规范）
    │
    ├── default/                           # 内置算法（开箱即用）
    │   ├── data_statistics.py             # 描述性统计分析
    │   ├── data_normalization.py          # 数据归一化/标准化
    │   └── spectrum_analysis.py           # 光谱峰值/FWHM/峰面积分析
    │
    └── extra_algorithms_fromProjects/     # 项目扩展算法
        └── prompt_template.py            # LLM 自动生成算法的工具
```

---

## 新增模块

### `readfile.py` — CSV读取工具

为自动分析流水线提供统一的CSV读取接口，LLM根据列名判断调用哪个函数，Python通过`READER_REGISTRY`动态分发。

**可用读取函数**：

- `read_column_names(csv_path)` → 返回所有列名
- `read_as_columns_dict(csv_path, columns=None)` → 返回{列名: [值列表]}
- `read_spectrum_format(csv_path, wavelength_col, intensity_col)` → 返回光谱格式{"wavelength": [...], "intensity": [...]}
- `read_numeric_columns(csv_path)` → 返回所有数值列{列名: [float, ...]}
- `read_single_column(csv_path, column)` → 返回单列值列表

**动态分发注册表**：

```python
READER_REGISTRY = {
    "read_as_columns_dict": read_as_columns_dict,
    "read_spectrum_format": read_spectrum_format,
    "read_numeric_columns": read_numeric_columns,
    "read_single_column": read_single_column,
}
```

### `auto_analyze.py` — 自动分析流水线

纯Python LLM分析流水线，不依赖Agent框架，实现"LLM读取列名 → 选择算法 → 执行分析 → 保存结果 → SSE推送"的全流程。

**工作流**：

1. 读取CSV列名
2. 调用LLM分析列名，选择最适合的算法和读取方式
3. 通过`READER_REGISTRY`动态调用读取函数
4. 执行选定算法
5. 保存结果到`results/`目录（覆盖写 + 时间戳存档）
6. 通过回调推送SSE消息

**SSE消息序列**：
- `info` → 正在读取CSV列名
- `info` → 已识别N列
- `progress` → LLM正在分析
- `info` → 已选定算法
- `info` → 正在读取数据
- `progress` → 正在执行算法
- `info` → 结果已保存
- `analysis_result` → 分析结果摘要
- `complete` → 分析完成

## 内置算法

### `data_statistics` — 描述性统计分析

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `columns` | list | 要分析的列名（dict 输入时有效），空则分析全部 | null |
| `include_correlation` | bool | 是否计算相关性矩阵 | true |

**输入**：`dict`（多列）或 `list`（单列）

**输出**：每列的 count / mean / median / std / variance / min / max / q25 / q75 及相关性矩阵

---

### `data_normalization` — 数据归一化/标准化

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `method` | str | `minmax` / `zscore` / `robust` | `minmax` |
| `columns` | list | 要处理的列名，空则处理全部 | null |
| `feature_range` | list | minmax 的目标区间 `[min, max]` | `[0, 1]` |

**输入**：`dict`（多列）或 `list`（单列）

**输出**：归一化后的数据 + 逆变换参数（`transform_params`）

---

### `spectrum_analysis` — 光谱分析

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `subtract_baseline` | bool | 是否扣除首尾均值基线 | true |
| `integration_range` | list | 峰面积计算的波长范围 `[start, end]` | null（全谱） |

**输入**（Excel 行格式）：
```json
{
  "wavelength": [400, 401, ..., 700],
  "intensity": [2, 1, ..., 234]
}
```
多条光谱：
```json
{
  "wavelength": [400, 401, ..., 700],
  "intensity": [[2,1,...,234], [3,2,...,198]]
}
```

**输出**：`peak_wavelength`（最高峰波长）、`peak_intensity`（峰强度）、`fwhm`（半高宽）、`peak_area`（峰面积）、`baseline`（估算基线）

---

## API 接口

### 获取算法列表

```
GET /api/software/algorithms
```

响应示例：
```json
[
  {"name": "data_statistics", "description": "...", "params_schema": {...}},
  {"name": "data_normalization", "description": "...", "params_schema": {...}},
  {"name": "spectrum_analysis", "description": "...", "params_schema": {...}}
]
```

---

### 运行算法

```
POST /api/software/run
Content-Type: application/json

{
  "algorithm": "spectrum_analysis",
  "data": {
    "wavelength": [400, 401, ..., 700],
    "intensity": [2, 1, ..., 234]
  },
  "params": {
    "subtract_baseline": true
  }
}
```

响应示例：
```json
{
  "success": true,
  "algorithm": "spectrum_analysis",
  "result": {
    "peak_wavelength": 532.0,
    "peak_intensity": 0.85,
    "fwhm": 15.3,
    "peak_area": 123.4,
    "baseline": 0.05
  },
  "message": "光谱分析完成"
}
```

---

### 从 CSV 数据运行算法（直接对提取结果做分析）

```
POST /api/software/run_on_csv
Content-Type: application/json

{
  "algorithm": "data_statistics",
  "params": {"include_correlation": true}
}
```
系统自动读取 `temporal/extraction.csv`，将数值列作为输入数据。

---

### 使用 LLM 生成新算法

```
POST /api/software/generate_algorithm
Content-Type: application/json

{
  "description": "我需要一个对光谱数据做高斯平滑的算法，输入是 wavelength 和 intensity 列表，窗口大小可配置，输出平滑后的强度序列"
}
```

响应：
```json
{
  "success": true,
  "name": "gaussian_smoothing",
  "filepath": "software/algorithms/extra_algorithms_fromProjects/gaussian_smoothing.py",
  "message": "算法已生成并保存，调用 SoftwareController() 后可立即使用"
}
```

---

## 添加新算法

### 方式一：通过 API 自动生成（推荐）

调用 `POST /api/software/generate_algorithm`，用自然语言描述需求即可。
系统会经过 **两步 LLM 调用**：
1. 提取算法规格（名称、输入格式、参数、输出字段）
2. 生成完整的 Python 代码并保存

生成的文件放在 `extra_algorithms_fromProjects/` 目录，重启后自动注册。

### 方式二：手动编写

在 `default/` 或 `extra_algorithms_fromProjects/` 中新建 `.py` 文件：

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from software.algorithms.base import BaseAlgorithm

class MyAlgorithm(BaseAlgorithm):
    name        = "my_algorithm"          # 唯一标识（小写+下划线）
    description = "算法功能说明"
    params_schema = {
        "window": {"type": "int", "description": "窗口大小", "default": 5}
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
    r = MyAlgorithm().run(data=[1, 2, 3, 4, 5])
    print(json.dumps(r, indent=2, ensure_ascii=False))
```

保存后，下次实例化 `SoftwareController()` 时自动注册，**无需修改任何其他文件**。

---

## 直接运行测试

每个算法文件底部均有 `if __name__ == "__main__":` 测试代码，可直接运行：

```bash
python software/algorithms/default/spectrum_analysis.py
python software/algorithms/default/data_statistics.py
python software/algorithms/default/data_normalization.py
python software/software_controller.py   # 测试所有算法
```
