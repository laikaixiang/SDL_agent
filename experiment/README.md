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

## 完整调用链：从用户点击到前端渲染

以用户点击"实验设计"后输入"帮我设计一个实验"为例：

### 第一阶段：模式切换

```
点击菜单"🧪 实验设计"
  → openExperimentDesignDialog()          experiment_design.js:13
  → setMode('hardware_design', '实验设计：', '🧪 实验设计对话')
  → currentMode.prefix = "实验设计："     state.js（全局状态）
```

### 第二阶段：第一次请求 `/api/chat`（模式识别，不调用 LLM）

```
用户输入"帮我设计一个实验" → 点击发送
  → sendMessage()                          chat.js:29
  → finalPayload = "实验设计：帮我设计一个实验"

POST /api/chat
Body: { "action": "chat", "message": "实验设计：帮我设计一个实验" }

  → app.py:204  检测到前缀"实验设计："
  → handle_hardware_request()
  → app.py:326  mode = "design"，直接返回（不调用 LLM）

Response: {
  "type": "experiment_design_mode",
  "command": "帮我设计一个实验",
  "reply": "🔬 实验设计模式\n\n..."
}
```

### 第三阶段：前端分发 → 第二次请求 `/api/experiment_chat`（真正调用 LLM）

```
_dispatchJsonResponse(data)               chat.js:77
  → type === 'experiment_design_mode'
  → appendMessage(data.reply, 'ai')       聊天框显示提示文字
  → startExperimentChat("帮我设计一个实验")

POST /api/experiment_chat
Body: { "session_id": "exp_1745xxxxxxx", "message": "帮我设计一个实验" }

  → app.py:804  experiment_chat()
  → ExperimentDesignAgent()               core/field_inference.py:173
  → agent.parse_experiment_design("帮我设计一个实验")
```

### 第四阶段：LLM 生成 JSON（耗时 10-15 秒）

```
parse_experiment_design()                 field_inference.py:366
  → 构造 prompt = system_prompt（含所有工具/算法描述，约2300字符）
                + "用户需求：帮我设计一个实验"
  → llm_client.call_api(
        model=MODEL_NAME_TALK,
        messages=[{"role":"user","content":prompt}],
        temperature=0.3,
        max_tokens=2048
    )
  → 清理 markdown 标记（```json ... ```）
  → json.loads(content)
  → validate_experiment_json()            每个 step 必须有 type/name/params
  → 返回 (True, experiment_json)
```

### 第五阶段：格式转换 → 返回前端

```
app.py:854
  → converter.json_to_visual(result)      experiment/format.py:22
  → steps 数组 → nodes + edges 图结构

Response: {
  "type": "experiment_design",
  "experiment_json": { "experiment_name": "...", "steps": [...] },
  "visual_data": { "nodes": [...], "edges": [...] },
  "reply": "✅ 已生成实验设计方案：... 共N个步骤"
}
```

### 第六阶段：前端渲染

```
experiment_chat.js:40
  → appendMessage(data.reply, 'ai')       聊天框显示成功消息
  → loadExperimentFromJSON(data.experiment_json)  渲染画布节点图
  → showNotification('✅ 实验设计已生成并推送到画布', 'success')
```

### 关键说明

- **两次 HTTP 请求**：第一次 `/api/chat` 只做前缀识别，不调用 LLM，立即返回；第二次 `/api/experiment_chat` 才真正调用 LLM
- **前端超时**：`experiment_chat.js` 使用 `AbortController` 设置 30 秒超时，LLM 生成约需 10-15 秒
- **system_prompt 来源**：`ExperimentDesignAgent.__init__` 初始化时从 `hardware/tools/REGISTRY.json`、`SoftwareController`、内置 helper 列表动态生成
- **visual_data 用途**：仅供前端画布渲染，执行时使用 `experiment_json`（标准格式）

## 测试

- 执行器测试: 见 `test/compile_test/`
- 编译器测试: `python experiment/compiler.py`
- 方案2测试: 见 `test/experiment_design_v2/`
- 超时诊断测试: `platform_init/test/experiment_design_test/test_complete_flow.py`

## 相关文档

- `CLAUDE.md` - 完整项目文档
- `experiment/COMPILER.md` - 编译器详细文档
- `test/experiment_design_v2/README.md` - 方案2测试说明
- `logs/20260420_experiment_design_timeout_fix_lkx.md` - 超时问题修复日志

