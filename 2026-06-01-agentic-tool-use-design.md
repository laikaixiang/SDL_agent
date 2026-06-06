# SDL_agent Agentic 改造设计

**日期**: 2026-06-01
**分支**: v2.5-多轮对话+tool_use
**目标**: 将前缀路由驱动的 chatbot 改造为真正的 Agent——LLM 自主选择工具、可中断询问用户、分层子 Agent 执行任务。

## 概述

当前系统通过用户手动输入前缀（`帮我搜寻：`、`硬件控制：`等）来触发不同功能模块。改造后，LLM 通过 tool-use 循环自主决定何时调用哪个工具，无需用户记忆前缀。

## 决策汇总

| 决策 | 结论 |
|------|------|
| 改造策略 | 渐进式：新增 `/api/chat/agent` 端点，与现有 `/api/chat` 并存 |
| 工具粒度 | 分层混合：主 Agent 用粗粒度工具，提取模块有内部子 Agent |
| 中断机制 | 工具级 `ask_user` + 消息级 SSE 确认事件，先阻塞式（TODO: 非阻塞持久化） |
| 思考模式 | 自动检测 config.json 模型：DeepSeek=标准 tool-use + 前端显示隐式思考；MiniMax=交错思维链保留 reasoning_details |
| 上下文控制 | 子 Agent 无对话历史，仅 system prompt + task；大结果只传摘要给主 Agent |

## 1. 多层 Agent 架构

### 1.1 三层结构

```
用户
 │
 ├── 主 Agent (Orchestrator, thin, ~2K tokens system prompt)
 │    上下文：当前对话 + 最近 N 轮 tool 结果摘要
 │    职责：理解意图、编排 Agent Team、确认危险操作、汇总结果
 │    工具：spawn_agent, ask_user, cancel_task + 全部功能工具
 │
 ├── Domain Agent 层（领域子 Agent，各司其职）
 │    │  每个 Domain Agent = 独立 LLM 实例，无对话历史
 │    │  由主 Agent 通过 spawn_agent(template, task) 动态创建
 │    │
 │    ├── 文献检索 Agent  ── 语义搜索 + 页面筛选
 │    ├── 文献提取 Agent  ── 逐页 VL 提取
 │    ├── 总结 Agent      ── 清洗去重 + 结构化
 │    ├── 实验设计 Agent  ── plan → confirm → execute
 │    ├── 硬件控制 Agent  ── 命令解析 + MQTT 执行
 │    └── 数据分析 Agent  ── 算法选择 + 执行
 │
 └── Agent Team 层（由主 Agent 动态编排）
       │
       ├── 并行 Team：多个 Agent 同时执行独立任务
       │    例：3 个检索 Agent 并行搜不同关键词 → 汇总 → 总结 Agent
       │
       ├── 串行 Team：Agent 流水线，上游输出 = 下游输入
       │    例：检索 Agent → 提取 Agent → 总结 Agent → 主 Agent
       │
       └── 混合 Team：并行 + 串行组合
            例：  ┌ 检索 #1 ┐
                  │ 检索 #2 ├→ 提取 Agent → 总结 Agent → 主 Agent
                  └ 检索 #3 ┘
```

### 1.2 Agent 编排协议

主 Agent 不硬编码任何子 Agent 类型，而是通过一个通用 `spawn_agent` 工具来动态创建：

```python
spawn_agent(
    template: str,      # Agent 模板名，映射到 prompts/zh/agents/{template}.yaml
    task: str,          # 任务描述
    context: dict = {}, # 可选的上下文数据
    mode: str = "single" # "single" | "parallel" | "pipeline"
) -> dict
```

**模板文件结构** (`prompts/zh/agents/{template}.yaml`)：

```yaml
name: literature_searcher
system_prompt: |
  你是文献检索专家。根据任务描述，在文献库中进行语义搜索，
  返回最相关的文献列表和页面范围。
  每次行动前先在 <think> 中分析搜索结果，再决定下一步。
tools: [search_literature, page_preview]
max_turns: 5
output_schema: {results: list, summary: str}
```

### 1.3 Agent Team 模式

**并行模式** — 主 Agent 发现任务可分解时，一次 spawn_agent 调用创建多个同名 Agent 实例：

```
用户："帮我调研钙钛矿钝化剂的三种主流策略"
  → 主 Agent 思考：3 个独立搜索方向
  → spawn_agent(template="literature_searcher",
                 task="小分子钝化剂研究进展",
                 mode="parallel",
                 siblings=[
                   {task: "聚合物钝化剂研究进展"},
                   {task: "无机盐钝化剂研究进展"}
                 ])
  → 3 个检索 Agent 并行执行
  → 结果汇总到主 Agent → 判断是否需要提取/总结
```

**流水线模式** — 上游 Agent 输出自动成为下游 Agent 输入：

```
spawn_agent(
    template="extraction_pipeline",
    task="从相关文献中提取钝化剂性能参数",
    mode="pipeline",
    pipeline=[
      {template: "literature_searcher", task: "检索相关文献"},
      {template: "literature_extractor", task: "从检索结果中逐页提取"},
      {template: "summarizer", task: "清洗去重，输出结构化 CSV"}
    ]
)
```

### 1.4 上下文隔离策略

| Agent 层 | 能看到什么 | 看不到什么 |
|----------|-----------|-----------|
| 主 Agent | 对话历史 + 所有 tool 结果摘要 | 子 Agent 内部细节、中间推理 |
| Domain Agent | 自身的 system prompt + task + context | 用户对话历史、其他 Agent 的内部状态 |
| Pipeline Agent | system prompt + 上游 Agent 的输出摘要 | 用户对话历史、并行分支的其他 Agent |

**大结果截断**：当 tool 结果很大时（如提取结果 CSV），只把摘要（行数、关键字段、文件路径）注入下游 agent，完整数据存文件。

### 1.5 Agent 注册表

所有可用的 Agent 模板注册在 `prompts/zh/agents/` 下，主 Agent 启动时扫描加载，作为 `spawn_agent` 的 `template` 可选值注入 system prompt：

```
prompts/zh/agents/
├── literature_searcher.yaml    # 文献检索 Agent
├── literature_extractor.yaml   # 文献提取 Agent
├── summarizer.yaml             # 总结清洗 Agent
├── experiment_designer.yaml    # 实验设计 Agent
├── hardware_controller.yaml    # 硬件控制 Agent
├── data_analyst.yaml           # 数据分析 Agent
├── algorithm_generator.yaml    # 算法生成 Agent
└── extraction_pipeline.yaml    # 提取流水线（组合模板）
```

## 2. UnifiedToolRegistry

合并现有 3 套平行注册体系：

```
UnifiedToolRegistry
  ├─ TOOLS_SCHEMA (OpenAI tools 格式, 供 LLM)
  └─ ToolExecutor (dispatch, 供 AgentLoop)

来源:
  hardware/tools/REGISTRY.json  ──→  scan_hardware_tools()
  software/algorithms/          ──→  scan_software_algorithms()
  内置 agent 工具               ──→  BUILTIN_TOOLS
```

### Phase 1 工具（8 个）

| Tool | 来源 | 说明 |
|------|------|------|
| `spawn_agent` | 内置 | 动态创建子 Agent 实例。params: `{template, task, context?, mode?}` |
| `ask_user` | 内置 | 向用户提问，暂停等待回答。params: `{question, options?}` |
| `search_literature` | 现有 SemanticSearch | 语义搜索文献库。params: `{query, top_k?}` |
| `control_hardware` | 现有 HardwareController | 执行硬件操作。params: `{command}` |
| `design_experiment` | 现有 ExperimentDesignAgent | 生成实验设计 JSON。内部 spawn 实验设计子 Agent。params: `{description}` |
| `analyze_data` | 现有 SoftwareManager | 分析 CSV 数据。params: `{csv_path?, algorithm?}` |
| `generate_algorithm` | 现有 AlgorithmGuide | 引导生成算法代码。params: `{description}` |
| `cancel_task` | 内置 | 取消正在执行的任务。params: `{task_id?}` |

### ask_user 约束

- 仅用于：意图模糊、危险操作确认、多策略选择
- 禁止用于：问已定义的参数范围、问设定好的默认值
- 约束写入主 Agent system prompt

### TODO: Phase 2 细粒度工具

文献提取子 Agent 内部工具链：`query_database`、`page_filter`、`extract_page`、`deduplicate_results` 等。

## 3. Agent Loop

### 3.1 主 Agent Loop

```
用户消息 → /api/chat/agent
              │
              ▼
    ┌─────────────────────────────┐
    │   AgentLoop.run(messages)   │
    │                             │
    │  while turn < max_turns:    │
    │    stream = LLM(            │
    │      messages,              │
    │      tools=TOOLS_SCHEMA,    │
    │      stream=True            │
    │    )                        │
    │                             │
    │    for chunk in stream:     │
    │      ├─ reasoning_content   │
    │      │  → SSE: thinking_*   │  ← 所有模型都可见
    │      ├─ content             │
    │      │  → SSE: text_delta   │
    │      ├─ tool_calls (delta)  │
    │      │  → 按 index 累积拼接  │
    │      │  → SSE: tool_call_*  │
    │      └─ finish_reason       │
    │         ├─ "stop" → DONE    │
    │         └─ "tool_calls"     │
    │            │                │
    │            ├─ name=="ask_user"
    │            │  → SSE: agent_question
    │            │  → 暂停，等用户回答
    │            │  → 继续循环
    │            │
    │            ├─ name=="spawn_agent"
    │            │  → 创建子 Agent 实例
    │            │  → 按 mode 编排执行（见 3.2）
    │            │  → 返回摘要给主 Agent
    │            │  → 继续循环
    │            │
    │            └─ 其他 tool
    │               → ToolExecutor
    │               → SSE: result
    │               → 继续循环
    └─────────────────────────────┘
```

- `max_turns` 默认 15，防止无限循环
- DeepSeek：reasoning_content 推送到前端 thinking 气泡
- MiniMax：检测模型名含 `minimax` 时保留 reasoning_details 在 messages 中传回

### 3.2 spawn_agent 编排引擎

```
主 Agent tool_call: spawn_agent(template, task, mode, ...)
              │
              ▼
    ┌─────────────────────────────────┐
    │   AgentOrchestrator.spawn()     │
    │                                 │
    │  mode == "single":               │
    │    创建 1 个 Agent 实例          │
    │    sub_agent.run(task)           │
    │    → 返回结果摘要                │
    │                                 │
    │  mode == "parallel":             │
    │    创建 N 个 Agent 实例          │
    │    ThreadPoolExecutor 并行执行    │
    │    → 收集全部 summary             │
    │    → 自动 spawn summarizer       │
    │    → 返回合并结果                │
    │                                 │
    │  mode == "pipeline":             │
    │    按 pipeline 列表顺序执行      │
    │    Agent1.run() → output         │
    │    Agent2.run(context=output)    │
    │    Agent3.run(context=output)    │
    │    → 返回最终结果                │
    └─────────────────────────────────┘
```

### 3.3 子 Agent 实例

```python
class SubAgent:
    def __init__(self, template_name: str):
        template = load_yaml(f"prompts/zh/agents/{template_name}.yaml")
        self.system_prompt = template["system_prompt"]
        self.tools = resolve_tools(template["tools"])     # 从 UnifiedToolRegistry 选取
        self.max_turns = template.get("max_turns", 5)
        self.llm = LLMClient(...)                          # 独立 LLM 实例

    def run(self, task: str, context: dict = None) -> dict:
        """执行子 Agent 的 tool-use 循环，返回结果摘要"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task}
        ]
        if context:
            messages.insert(1, {"role": "system", "content": f"上下文数据: {json.dumps(context)}"})

        # 复用相同的 AgentLoop 引擎（无对话历史，仅 task + context）
        loop = AgentLoop(self.llm, self.tools, max_turns=self.max_turns)
        return loop.run(messages)
```

## 4. SSE 协议

### 现有事件（不变）

`thinking_start` / `thinking_delta` / `thinking_end` / `text_start` / `text_delta` / `text_end` / `error` / `done`

### 新增事件

| 事件 | Payload | 说明 |
|------|---------|------|
| `tool_call_start` | `{index, name, id}` | LLM 决定调工具 |
| `tool_call_args` | `{index, delta}` | 参数流式增量 |
| `tool_call_end` | `{index, name, args}` | 参数完整，可展示卡片 |
| `tool_result` | `{index, name, result, status}` | 工具执行结果 |
| `agent_question` | `{question, options?}` | ask_user 触发，前端暂停 |
| `team_spawn` | `{mode, agents: [{id, template, task}]}` | Agent Team 创建 |
| `team_progress` | `{agent_id, status, summary?}` | 单个子 Agent 完成 |
| `team_done` | `{mode, results: [...]}` | Agent Team 全部完成 |
| `agent_done` | `{}` | Agent loop 结束 |
| `agent_error` | `{message}` | Loop 层面错误 |

### 完整轮次示例

```
thinking_start → thinking_delta* → thinking_end
tool_call_start  {index:0, name:"extract_literature", id:"call_1"}
tool_call_args   {index:0, delta:"{\"task_description\":\"..."}
tool_call_end    {index:0, name:"extract_literature", args:{...}}
tool_result      {index:0, status:"ok", result:"提取完成：45条记录"}
thinking_start → thinking_delta* → thinking_end     ← 反思 tool 结果
text_start → text_delta* → text_end
agent_done
```

## 5. 前端改造

### chat store 新增状态

```typescript
agentMode: boolean
activeToolCalls: Map<index, {name, status, args, result?}>
agentQuestion: {question, options?} | null
agentThinking: string  // 实时思考内容
```

### 消息气泡新增类型

- **Thinking 气泡** — 折叠/可展开，显示 agent 推理过程
- **Tool Call 卡片** — 图标 + 工具名 + 参数摘要 + 状态（pending → running → done/error）
- **Agent Team 卡片** — 显示并行/流水线中的子 Agent 列表，每个带进度指示
  ```
  ┌─────────────────────────────────────────┐
  │ 🔄 文献调研 Agent Team (parallel)        │
  │  ├── 📚 小分子钝化剂  ✓ 完成 (8篇)      │
  │  ├── 📚 聚合物钝化剂  ⏳ 搜索中...       │
  │  └── 📚 无机盐钝化剂  ⏳ 搜索中...       │
  └─────────────────────────────────────────┘
  ```
- **Agent Question 卡片** — 问题文本 + 可选项按钮，暂停等待用户
- **子 Agent 进度**（Phase 2）— 提取子 Agent 的实时进度条

### 输入框适配

agent 模式等待 ask_user 时：输入框高亮，placeholder 改为 "回答 agent 的问题..."

## 6. 分阶段实施

| 阶段 | 内容 | 产出 |
|------|------|------|
| Phase 1 | Agent 引擎 + 流式 tool-use loop + 8 个工具 + `ask_user` + `spawn_agent`(single) | Agent 可对话、选工具、中断问用户、创建单个子 Agent |
| Phase 2 | `spawn_agent` 并行/流水线模式 + 文献提取 Agent Team（检索→提取→总结）| 提取全流程 agent 化，用户可见 Agent Team 卡片 |
| Phase 3 | 实验设计子 Agent + 其余 Domain Agent 模板完善 | 全部功能模块 agent 化 |
| Phase 4 | 前端 Agent Team 可视化增强 + 非阻塞暂停（TODO） | 完整 agent 体验 |

## 7. 兼容性

- 新端点 `/api/chat/agent` 与现有 `/api/chat` 并存
- 现有前缀路由（`帮我搜寻：` 等）继续工作，逐步迁移
- 前端通过路由或开关切换 agent 模式 vs 传统模式
