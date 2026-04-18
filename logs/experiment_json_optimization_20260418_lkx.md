# 实验 JSON 传递与解析优化

**日期**: 2026-04-18  
**修改人**: lkx  
**版本标签**: v1.2.3-experiment-json-refactor

---

## 修改文件清单

| 文件路径 | 修改类型 | 说明 |
|---------|---------|------|
| `app.py` | 重构 | `execute_experiment_design` 路由内嵌执行器迁移到 `ExperimentManager.execute_plan` |
| `core/experiment_manager.py` | 增强 | 补全 `software` 类型步骤支持，新增 LOOP/GROUP/CONDITION/END/USER_INPUT helper |
| `templates/static/js/experiment/experiment_design.js` | 增强 | `renderExperimentSteps` 补全 `software` 步骤渲染，JSON 序列化透传 `input_file`/`output_file` |
| `CLAUDE.md` | 文档更新 | 补充执行阶段架构说明和 software 节点约定 |

---

## 问题背景

1. **app.py 内嵌执行器冗余**：`execute_experiment_design` 路由里有 70+ 行内嵌的 `execute_experiment_thread` 函数，只处理 `tool` 和 `helper` 类型，`software` 类型步骤无法执行
2. **software 步骤链路不完整**：
   - 后端 `ExperimentManager.execute_plan` 已支持 `software` 类型，但前端渲染时只显示紫色 JSON 字符串
   - `visual_to_json` 无法识别 `software` 节点，导致前端编辑后保存时丢失 `input_file`/`output_file` 字段
3. **helper 类型不完整**：`helper_map` 只有 `WAIT`，前端已有 LOOP/GROUP/CONDITION 模板但后端无对应执行器

---

## 解决方案

### 1. app.py 路由重构

**变更前**：
```python
def execute_experiment_design():
    # ... 70 行内嵌执行逻辑 ...
    def execute_experiment_thread():
        for step in steps:
            if step_type == 'tool':
                # 硬件工具执行
            elif step_type == 'helper':
                # 只处理 WAIT
```

**变更后**：
```python
def execute_experiment_design():
    from core.experiment_manager import ExperimentManager
    
    def _run():
        mgr = ExperimentManager(software_manager=software_manager)
        result = mgr.execute_plan(data, progress_callback=on_progress)
        # 通过 task_manager 推送进度
```

**收益**：
- 删除 70 行重复逻辑，路由层只负责任务调度
- `software` 类型步骤自动走 `_execute_software_algorithm`，无需额外适配
- 统一使用 `ExperimentManager.execute_plan` 的进度回调机制

---

### 2. experiment_manager.py 补全

#### 2.1 helper_map 扩展

```python
self.helper_map = {
    "WAIT":       self._execute_wait,
    "LOOP":       self._execute_loop,
    "GROUP":      self._execute_group,
    "CONDITION":  self._execute_condition,
    "END":        self._execute_end,
    "USER_INPUT": self._execute_user_input,
}
```

新增执行方法（当前为标记性执行，嵌套步骤展开留待后续实现）：
- `_execute_loop`: 记录循环次数
- `_execute_group`: 步骤组标记
- `_execute_condition`: 条件判断标记
- `_execute_end`: 结束点标记
- `_execute_user_input`: 用户输入标记

#### 2.2 visual_to_json 补全 software 节点识别

```python
if node_type == "wait":
    step_type = "helper"
    step_name = "WAIT"
elif node_type in ("loop", "group", "condition"):
    step_type = "helper"
    step_name = node_type.upper()
elif node_type.startswith("software:") or node.get("step_type") == "software":
    step_type = "software"
    step_name = node_type.replace("software:", "") or node.get("algo_name", node_type)
else:
    step_type = "tool"
    step_name = node_type

# software 步骤透传 input_file / output_file
if step_type == "software":
    if node.get("input_file"):
        step["input_file"] = node["input_file"]
    if node.get("output_file"):
        step["output_file"] = node["output_file"]
```

#### 2.3 _get_action_label 补全

```python
labels = {
    "spin_coating":    "旋涂",
    "set_temperature": "温度控制",
    "move_robot_arm":  "机械臂移动",
    "collect_spectrum":"光谱采集",
    "WAIT":            "等待",
    "LOOP":            "循环",
    "GROUP":           "步骤组",
    "CONDITION":       "条件判断",
}
```

---

### 3. experiment_design.js 前端补全

#### 3.1 renderExperimentSteps 补全 software 渲染

```javascript
if (step.type === 'tool') {
    for (const [k, v] of Object.entries(step.params)) {
        paramsHtml += `<div><strong>${k}:</strong> ${v}</div>`;
    }
} else if (step.type === 'software') {
    paramsHtml += `<div><strong>算法:</strong> ${step.name}</div>`;
    if (step.input_file)  paramsHtml += `<div><strong>输入:</strong> ${step.input_file}</div>`;
    if (step.output_file) paramsHtml += `<div><strong>输出:</strong> ${step.output_file}</div>`;
    for (const [k, v] of Object.entries(step.params || {})) {
        paramsHtml += `<div><strong>${k}:</strong> ${v}</div>`;
    }
} else {
    paramsHtml = `<div style="color:#7c3aed;font-style:italic;">${JSON.stringify(step.params)}</div>`;
}
```

#### 3.2 loadExperimentFromJSON 透传 software 字段

```javascript
experimentSteps = json.steps.map(step => {
    const s = {
        type:        step.type || 'tool',
        name:        step.name || step.action || '',
        params:      step.params || {},
        description: step.description || ''
    };
    if (step.type === 'software') {
        if (step.input_file)  s.input_file  = step.input_file;
        if (step.output_file) s.output_file = step.output_file;
        if (step.user_params) s.user_params = step.user_params;
    }
    return s;
});
```

#### 3.3 updateExperimentJSON 序列化保留 software 字段

```javascript
const steps = experimentSteps.map(step => {
    const s = { type: step.type, name: step.name, params: step.params, description: step.description };
    if (step.type === 'software') {
        if (step.input_file)  s.input_file  = step.input_file;
        if (step.output_file) s.output_file = step.output_file;
        if (step.user_params) s.user_params = step.user_params;
    }
    return s;
});
```

---

## 标准 JSON 格式

### software 步骤完整格式

```json
{
  "type": "software",
  "name": "data_statistics",
  "params": {
    "include_correlation": true
  },
  "input_file": "dialogue data/20260418_155027/temporal/extraction.csv",
  "output_file": "dialogue data/20260418_155027/results/stats.json",
  "description": "统计分析提取数据"
}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|-----|------|------|------|
| `type` | string | ✅ | 步骤类型：`tool` / `helper` / `software` |
| `name` | string | ✅ | 操作名称（tool/helper）或算法名称（software） |
| `params` | object | ✅ | 参数字典，传给算法的 `run(data, params)` |
| `description` | string | ❌ | 步骤描述（前端显示用） |
| `input_file` | string | ❌ | 输入 CSV 文件路径（仅 software 类型） |
| `output_file` | string | ❌ | 输出 JSON 文件路径（仅 software 类型） |
| `user_params` | object | ❌ | 用户额外参数，运行时合并到 `params`（仅 software 类型） |

---

## 测试验证

### 测试场景 1：纯硬件实验

```json
{
  "experiment_name": "旋涂测试",
  "steps": [
    {"type": "tool", "name": "spin_coating", "params": {"spin_speed": 3000, "spin_dur": 30000}},
    {"type": "helper", "name": "WAIT", "params": {"duration": 5000}},
    {"type": "tool", "name": "collect_spectrum", "params": {"duration": 60}}
  ]
}
```

**预期**：执行旋涂 → 等待 5 秒 → 采集光谱

---

### 测试场景 2：硬件 + 软件混合实验

```json
{
  "experiment_name": "旋涂+数据分析",
  "steps": [
    {"type": "tool", "name": "spin_coating", "params": {"spin_speed": 3000}},
    {"type": "helper", "name": "WAIT", "params": {"duration": 3000}},
    {"type": "tool", "name": "collect_spectrum", "params": {"duration": 60}},
    {
      "type": "software",
      "name": "spectrum_analysis",
      "params": {},
      "input_file": "dialogue data/20260418_155027/temporal/spectrum.csv",
      "output_file": "dialogue data/20260418_155027/results/spectrum_result.json"
    }
  ]
}
```

**预期**：
1. 执行旋涂
2. 等待 3 秒
3. 采集光谱（数据写入 `spectrum.csv`）
4. 调用 `spectrum_analysis` 算法分析光谱数据，结果保存到 `spectrum_result.json`

---

### 测试场景 3：前端编辑保存

1. 在实验设计画布中添加 `software` 步骤（通过 AI 生成或手动编辑 JSON）
2. 点击"保存实验设计"
3. 重新加载实验 JSON

**预期**：`input_file` 和 `output_file` 字段不丢失，前端正确渲染算法名称和文件路径

---

## 架构改进

### 变更前

```
app.py (execute_experiment_design)
  └─ execute_experiment_thread (内嵌 70 行)
       ├─ tool: hardware_controller.execute_tool_calls
       └─ helper: 只处理 WAIT
```

### 变更后

```
app.py (execute_experiment_design)
  └─ ExperimentManager.execute_plan
       ├─ tool: self.action_map[action](params)
       ├─ helper: self.helper_map[action](params)
       └─ software: self._execute_software_algorithm(step)
            └─ SoftwareManager.run_algorithm(algo_name, data, params)
```

**收益**：
- 单一职责：app.py 只负责路由和任务调度，执行逻辑全在 `ExperimentManager`
- 可扩展：新增 helper 类型只需在 `helper_map` 注册，无需修改 app.py
- 可测试：`ExperimentManager.execute_plan` 可独立测试，不依赖 Flask 上下文

---

## 后续优化方向

1. **嵌套步骤展开**：LOOP/GROUP/CONDITION 当前只是标记，需要实现嵌套步骤的递归执行
2. **条件判断求值**：CONDITION 的 `condition` 字段需要安全的表达式求值器（避免 `eval` 安全风险）
3. **用户输入交互**：USER_INPUT 需要前端弹窗或 SSE 双向通信支持
4. **实验编译器**：将 JSON 编译为独立的 Python 脚本，支持离线执行和版本控制
5. **可视化编辑器增强**：支持拖拽添加 `software` 步骤，自动补全 `input_file`/`output_file` 路径

---

## 兼容性说明

- **向后兼容**：旧格式 `action` 字段仍然支持，`execute_plan` 会自动识别 `action` 或 `name`
- **前端兼容**：`loadExperimentFromJSON` 同时支持 `step.name` 和 `step.action`
- **节点类型约定**：`visual_to_json` 支持两种 software 节点标记方式：
  - `type="software:algo_name"`（推荐）
  - `step_type="software"` + `algo_name` 字段

---

## 总结

本次优化完成了实验 JSON 传递与解析的全链路打通：

1. **后端执行器统一**：删除 app.py 内嵌逻辑，统一使用 `ExperimentManager.execute_plan`
2. **software 步骤完整支持**：后端解析 → 执行 → 前端渲染 → JSON 序列化全流程打通
3. **helper 类型补全**：新增 LOOP/GROUP/CONDITION/END/USER_INPUT 执行器（标记性实现）
4. **文档同步更新**：CLAUDE.md 补充执行阶段架构说明和 software 节点约定

现在用户可以在实验设计中混合使用硬件工具、辅助函数和软件算法，实现更复杂的自动化实验流程。
