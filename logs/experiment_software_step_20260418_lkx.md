# SDL_agent 工作日志 — 实验方案支持软件算法步骤

**更新时间**: 2026年4月18日  
**更新人员**: lkx  
**版本标识**: ExperimentManager 新增 `type: "software"` 步骤解析与执行

---

## 更新概述

为 `core/experiment_manager.py` 增加对软件算法步骤的支持，使实验方案 JSON 中可以直接调用 `software/algorithms/` 下注册的算法（如 `spectrum_analysis`），并提供输入文件、输出文件、用户参数等完整接口。

---

## 核心变更

### 修改文件

**`core/experiment_manager.py`**

| 变更点 | 说明 |
|--------|------|
| 新增 `import os` | 文件路径操作 |
| 新增 `from core.software_manager import SoftwareManager` | 引入算法执行器 |
| `__init__` 新增 `software_manager` 参数 | 支持外部注入，内部懒加载兜底 |
| 新增 `_get_software_manager()` | 懒加载 SoftwareManager |
| 新增 `_execute_software_algorithm(step)` | 软件算法步骤执行逻辑 |
| `execute_plan()` 新增 `software` 分支 | 在 helper/tool 分支前拦截处理 |
| `validate_plan()` 新增 `software` 分支 | 跳过 action_map 校验，运行时由 SoftwareManager 校验 |
| `json_to_visual()` 新增 `software` 标签 | 可视化节点显示 `算法:xxx` |

---

## 新增 JSON 步骤格式

```json
{
  "type": "software",
  "name": "spectrum_analysis",
  "params": {
    "subtract_baseline": true
  },
  "input_file": "dialogue data/xxx/temporal/extraction.csv",
  "output_file": "dialogue data/xxx/results/spectrum_out.json",
  "user_params": {
    "integration_range": [500, 600]
  },
  "description": "光谱数据分析：检测最高峰波长/强度、计算半高宽（FWHM）和峰面积"
}
```

字段说明：

| 字段 | 必填 | 说明 |
|------|------|------|
| `type` | 是 | 固定为 `"software"` |
| `name` | 是 | 算法名，对应 REGISTRY.json 中的 `name` |
| `params` | 否 | 传给算法 `run()` 的参数 |
| `input_file` | 否 | 输入 CSV 文件路径，不填则传空 dict |
| `output_file` | 否 | 结果 JSON 保存路径，不填则不保存 |
| `user_params` | 否 | 用户额外参数，与 `params` 合并（优先级更高） |

---

## 执行结果格式

`execute_plan()` 返回的 results 列表中，软件步骤条目新增 `detail` 字段：

```json
{
  "step": 1,
  "action": "spectrum_analysis",
  "description": "...",
  "result": "光谱分析完成",
  "detail": { "peak_wavelength": 532.0, "fwhm": 15.3, ... },
  "success": true
}
```

---

**更新人**: lkx  
**更新时间**: 2026年4月18日  
**状态**: 已完成
