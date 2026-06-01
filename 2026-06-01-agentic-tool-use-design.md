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

## 1. Agent 分层架构

```
用户
 │
 ├── 主 Agent (thin, ~2K tokens system prompt)
 │    上下文：当前对话 + 最近 N 轮 tool 结果摘要
 │    职责：理解意图、选工具、确认危险操作、汇总子 agent 结果
 │
 ├── tool_call → 文献提取子 Agent
 │    │  system prompt: prompts/zh/extraction/_system.yaml
 │    │  上下文：仅 task_description + fields（无对话历史）
 │    │
 │    ├── 内部可并行分发搜索（多关键词并行 → 合并结果）
 │    ├── 内部逐页提取（每页一次 VL call）
 │    └── 总结子 Agent 清洗去重 → 返回摘要给主 Agent
 │
 ├── tool_call → 实验设计子 Agent
 │    │  system prompt: prompts/zh/experiment/_system.yaml
 │    │  上下文：仅 design_description（无对话历史）
 │    │  模式：plan → confirm → execute（不走 tool-use，直接生成 JSON）
 │    │  输出：{experiment_json, visual_data, reply}
 │    │
 │    └── 确认流程：生成后推 canvas + 回复摘要，用户修改后编译执行
 │
 ├── tool_call → 硬件控制（直接执行，已有完整逻辑）
 └── tool_call → 数据分析（直接执行，已有完整逻辑）
```

### 上下文隔离策略

| Agent | 能看到什么 | 看不到什么 |
|-------|-----------|-----------|
| 主 Agent | 对话历史 + tool 结果摘要 | 子 agent 内部细节 |
| 提取子 Agent | task_description + fields + system prompt | 用户对话历史 |
| 实验设计子 Agent | design_description + system prompt | 用户对话历史 |
| 总结子 Agent | 原始提取结果 | 用户对话历史 |

### 大结果截断

当 tool 结果很大时（如提取结果 CSV），只把摘要（行数、关键字段、文件路径）注入主 agent 消息，完整数据存文件。

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

### Phase 1 工具（7 个）

| Tool | 来源 | 说明 |
|------|------|------|
| `ask_user` | 内置 | 向用户提问，暂停等待回答。params: `{question, options?}` |
| `extract_literature` | 现有 ExtractionEngine | 从 PDF 提取结构化数据，内部启动提取子 Agent |
| `search_literature` | 现有 SemanticSearch | 语义搜索文献库。params: `{query, top_k?}` |
| `control_hardware` | 现有 HardwareController | 执行硬件操作。params: `{command}` |
| `design_experiment` | 现有 ExperimentDesignAgent | 生成实验设计 JSON。params: `{description}` |
| `analyze_data` | 现有 SoftwareManager | 分析 CSV 数据。params: `{csv_path?, algorithm?}` |
| `generate_algorithm` | 现有 AlgorithmGuide | 引导生成算法代码。params: `{description}` |

### ask_user 约束

- 仅用于：意图模糊、危险操作确认、多策略选择
- 禁止用于：问已定义的参数范围、问设定好的默认值
- 约束写入主 Agent system prompt

### TODO: Phase 2 细粒度工具

文献提取子 Agent 内部工具链：`query_database`、`page_filter`、`extract_page`、`deduplicate_results` 等。

## 3. Agent Loop

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
    │            │                │
    │            └─ 其他 tool      │
    │               → ToolExecutor│
    │               → SSE: result │
    │               → 继续循环     │
    └─────────────────────────────┘
```

- `max_turns` 默认 15，防止无限循环
- DeepSeek：reasoning_content 推送到前端 thinking 气泡
- MiniMax：检测模型名含 `minimax` 时保留 reasoning_details 在 messages 中传回

### 子 Agent 调度

```
主 Agent tool_call extract_literature
  → 创建提取子 Agent 实例（system prompt = prompts/zh/extraction/_system.yaml）
  → 子 Agent 内部：
      并行 search_literature × N 关键词
      → 合并 → 逐页 VL 提取
      → 总结 Agent 清洗去重
  → 返回摘要给主 Agent：
      "提取完成：从 8 篇文献中提取 45 条记录，涉及 12 个字段。
       关键字段：钝化剂名称、效率、稳定性。数据文件：session/extract/xxx.csv"
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
- **Agent Question 卡片** — 问题文本 + 可选项按钮，暂停等待用户
- **子 Agent 进度**（Phase 2）— 提取子 Agent 的实时进度条

### 输入框适配

agent 模式等待 ask_user 时：输入框高亮，placeholder 改为 "回答 agent 的问题..."

## 6. 分阶段实施

| 阶段 | 内容 | 产出 |
|------|------|------|
| Phase 1 | Agent 引擎 + 流式 tool-use loop + 7 个工具 + `ask_user` | Agent 可对话、选工具、中断问用户 |
| Phase 2 | 文献提取子 Agent（内部并行搜索 + 逐页提取 + 总结 Agent） | 提取全流程 agent 化 |
| Phase 3 | 实验设计子 Agent + 其余工具完善 | 全部功能模块 agent 化 |
| Phase 4 | 前端 tool-call 可视化增强 + 非阻塞暂停（TODO） | 完整 agent 体验 |

## 7. 兼容性

- 新端点 `/api/chat/agent` 与现有 `/api/chat` 并存
- 现有前缀路由（`帮我搜寻：` 等）继续工作，逐步迁移
- 前端通过路由或开关切换 agent 模式 vs 传统模式
