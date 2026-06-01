# SDL_agent v2.5 Agentic 改造 — 工作总结

**日期**: 2026-06-01  
**分支**: `v2.5-多轮对话+tool_use`  
**提交数**: 27 commits  

---

## 目标

将前缀路由驱动的 chatbot 升级为真正的 AI Agent——LLM 通过 tool-use 循环自主选择工具、可中断询问用户、分层 Multi-Agent 协同执行任务。

---

## 新增/修改文件

### 后端引擎

| 文件 | 操作 | 行数 | 说明 |
|------|------|------|------|
| `core/agent_tools.py` | 新建 | ~460 | 22 个统一工具注册（6 builtin + 10 hardware + 6 software） |
| `core/agent_loop.py` | 新建 | ~530 | AgentLoop 流式 tool-use 引擎 + AgentOrchestrator + SubAgent |
| `utils/stream_adapter.py` | 修改 | +100 | 新增 tool_call delta 累积 + SSE 事件类型（TOOL_CALL_START/ARGS/END） |
| `app.py` | 修改 | +280 | `/api/chat/agent` + `/api/chat/agent/respond` 路由 + agent 初始化 |
| `core/config.py` | 修改 | +3 | AGENT_MAX_TURNS, AGENT_ENABLED 配置项 |
| `config.example.json` | 修改 | +3 | agent 配置模板 |

### 子 Agent 模板

| 文件 | 工具 | 说明 |
|------|------|------|
| `prompts/zh/agents/literature_searcher.yaml` | search_literature + ask_user | 文献检索 Agent |
| `prompts/zh/agents/literature_extractor.yaml` | preview_pdf_page + extract_from_pdf + ask_user | 文献提取 Agent |
| `prompts/zh/agents/experiment_designer.yaml` | design_experiment + ask_user | 实验设计 Agent |
| `prompts/zh/agents/data_analyst.yaml` | 3 algorithms + ask_user | 数据分析 Agent |
| `prompts/zh/agents/summarizer.yaml` | 纯文本 | 数据总结 Agent |
| `prompts/zh/agents/extraction_pipeline.yaml` | search + preview + spawn_agent | 提取流水线协调器 |

### 前端

| 文件 | 操作 | 说明 |
|------|------|------|
| `frontend/src/types/chat.ts` | 修改 | ToolCallInfo/AgentQuestion/AgentEvent/TeamAgentInfo 类型 |
| `frontend/src/api/chat.ts` | 修改 | sendAgentMessage() + respondToAgent() + team 事件解析 |
| `frontend/src/components/chat/AgentToolCard.vue` | 新建 | 工具调用状态卡片（running/done/error） |
| `frontend/src/components/chat/AgentTeamCard.vue` | 新建 | Agent Team 进度卡片（并行/流水线可视化） |
| `frontend/src/components/chat/AgentQuestionCard.vue` | 新建 | ask_user 问题卡片（含选项按钮） |
| `frontend/src/components/chat/ThinkingBubble.vue` | 新建 | 可折叠思考过程气泡 |
| `frontend/src/components/chat/ChatContainer.vue` | 修改 | 集成全量 Agent UI 组件 + 状态管理 |
| `frontend/src/components/chat/InputBar.vue` | 修改 | Agent 模式 placeholder + ask_user 响应 |

### 测试

| 目录 | 文件 | 测试数 | 覆盖范围 |
|------|------|--------|---------|
| `platform_init/test/agent/` | test_agent_tools.py | 14 | 工具注册、schema 构建、dispatch |
| | test_stream_adapter.py | 10 | tool_call delta 累积、状态重置、边界情况 |
| | test_agent_loop.py | 11 | AgentLoop 生命周期（Mock LLM）、ask_user、max_turns |
| `platform_init/test/agent_phase2/` | test_phase2_tools.py | 8 | extract_from_pdf/preview_pdf_page 注册与调用 |
| | test_pipeline_orchestration.py | 7 | pipeline 模式、context 传递、错误恢复 |
| | test_extraction_flow.py | 5 | 集成流程：search→extract→summarize |
| `platform_init/test/agent_phase3/` | test_phase3_templates.py | 12 | 6 个模板加载、工具子集筛选、编排器 spawn |
| `platform_init/test/agent_system/` | test_tool_registry.py | 11 | 22 tools 注册、schema 验证 |
| | test_stream_protocol.py | 11 | SSE 协议：thinking/tool_call/text 转换 |
| | test_agent_lifecycle.py | 11 | AgentLoop 完整生命周期 |
| | test_sub_agent_system.py | 11 | SubAgent 模板加载 + 编排器 spawn/pipeline |
| | test_integration.py | 8 | 端到端：search→extract / design / analyze / 错误恢复 |
| **合计** | **12 files** | **119** | **全部通过** |

### 文档

| 文件 | 说明 |
|------|------|
| `2026-06-01-agentic-tool-use-design.md` | 设计文档（Multi-Agent 架构 + SSE 协议 + 分阶段计划） |
| `docs/superpowers/plans/2026-06-01-agentic-phase1.md` | Phase 1 实施计划（8 tasks） |
| `README.md` | 更新 v2.5 Agentic 改造章节 |

---

## 架构变化

```
旧 (v2.3):  用户 → 手动输入前缀 → app.py 前缀路由 → 单一功能模块

新 (v2.5):  用户 → /api/chat/agent → AgentLoop (22 tools)
                   ├── think → tool_call → execute → result → think → reply
                   ├── ask_user → 中断等待 → 用户回答 → 继续
                   └── spawn_agent → 子 Agent Team (parallel/pipeline/single)
```

---

## 4 阶段交付

| Phase | 内容 | 关键产出 |
|-------|------|---------|
| 1 | Agent 引擎 + 流式 tool-use + 22 工具 + ask_user | AgentLoop 跑通，可自主选工具、中断问用户 |
| 2 | 文献提取管线（search→extract→summarize）+ pipeline 模式 | 完整提取流程 agent 化 |
| 3 | 6 个子 Agent 模板完善（每个都有 ask_user + 专业工具） | 全部功能模块 agent 化 |
| 4 | Agent Team 可视化 + 前端增强 | AgentTeamCard / 进度条 / 零 TS 错误 |

---

## 技术要点

- **Tool-Use 循环**: `LLM(stream=True, tools=22)` → `StreamAdapter` 累积 tool_call delta → `executor.dispatch()` → tool result 追加到 messages → 继续
- **流式 SSH 事件**: `thinking_*` / `tool_call_start/args/end` / `tool_result` / `agent_question` / `team_spawn/progress/done` / `agent_done`
- **上下文隔离**: 子 Agent 无对话历史，仅 system_prompt + task；大结果截断 2000 字符
- **DeepSeek 隐式思考**: `reasoning_content` → 前端 thinking 气泡
- **ask_user 阻塞式**: `queue.Queue.get(timeout=300)`，前端 `/api/chat/agent/respond` 唤醒
- **向后兼容**: `/api/chat/agent` 与旧 `/api/chat` 前缀路由并存
