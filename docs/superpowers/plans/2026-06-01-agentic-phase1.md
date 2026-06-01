# Phase 1: Agentic 改造 — Agent 引擎 + 流式 Tool-Use Loop

> **For agentic workers:** Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** 构建核心 Agent 引擎：主 Agent 通过流式 tool-use loop 自主选择工具，支持 `ask_user` 中断和 `spawn_agent` 单 Agent 创建。

**Architecture:** 新增 `core/agent_loop.py` + `core/agent_tools.py`，扩展 `StreamAdapter` 支持 tool_call delta 事件，新增 `/api/chat/agent` 端点。子 Agent 模板在 `prompts/zh/agents/*.yaml`。前端 Vue 3 新增 Agent UI 组件。

**Tech Stack:** Python 3.10+ / Flask / Vue 3 + TypeScript + Pinia

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `core/agent_tools.py` | Create | UnifiedToolExecutor: 从 hardware/software 扫描 + 内置工具 |
| `core/agent_loop.py` | Create | AgentLoop + AgentOrchestrator + SubAgent |
| `utils/stream_adapter.py` | Modify | 新增 tool_call delta 累积和 TOOL_CALL_* 事件 |
| `prompts/zh/agents/literature_searcher.yaml` | Create | 文献检索子 Agent 模板 |
| `prompts/zh/agents/experiment_designer.yaml` | Create | 实验设计子 Agent 模板 |
| `prompts/zh/agents/summarizer.yaml` | Create | 总结子 Agent 模板 |
| `prompts/zh/agents/data_analyst.yaml` | Create | 数据分析子 Agent 模板 |
| `core/config.py` | Modify | 新增 AGENT_MAX_TURNS, AGENT_ENABLED |
| `config.example.json` | Modify | 新增 agent 配置项 |
| `app.py` | Modify | 新增 /api/chat/agent + /api/chat/agent/respond 路由 |
| `frontend/src/types/chat.ts` | Modify | 新增 AgentEvent, ToolCallInfo, AgentQuestion 类型 |
| `frontend/src/api/chat.ts` | Modify | 新增 sendAgentMessage(), respondToAgent() |
| `frontend/src/components/chat/AgentToolCard.vue` | Create | 工具调用状态卡片 |
| `frontend/src/components/chat/AgentQuestionCard.vue` | Create | ask_user 问题卡片 |
| `frontend/src/components/chat/ThinkingBubble.vue` | Create | 可折叠思考过程气泡 |
| `frontend/src/components/chat/ChatContainer.vue` | Modify | 渲染 Agent 事件气泡 |
| `frontend/src/components/chat/InputBar.vue` | Modify | Agent 模式 placeholder + disabled 状态 |

---

### Task 1: 创建 `core/agent_tools.py` — 统一工具注册与执行

**Files:** Create: `core/agent_tools.py`

- [ ] **Step 1: 实现 UnifiedToolExecutor**

定义 `AgentTool` dataclass（name, description, parameters, required, func, category, dangerous）。

`BUILTIN_TOOLS` 列表包含 `ask_user`：no-op 函数（返回 `"__ASK_USER_PENDING__"`），由 AgentLoop 拦截。

`scan_hardware_tools()` — 遍历 `hardware.ToolRegistry.get_all()`，将 registry 的 params dict 转为 OpenAI JSON Schema 格式（`{type: "object", properties: {...}}`），标记 `dangerous=True`。func 封装为 `lambda args, name=name: _dispatch_hardware(name, args)`。

`scan_software_algorithms()` — 调用 `SoftwareController.list_algorithms()`，同样转为 JSON Schema。func 封装调用 `controller.execute_algorithm(name, **args)`。

`UnifiedToolExecutor` 类：
- `__init__(tools: list[AgentTool])` — 构建 `{name: AgentTool}` dict
- `build_openai_tools()` — 返回 OpenAI tools 参数格式列表
- `dispatch(name, arguments) -> str` — 按 name 查找并执行
- `is_hardware_tool(name) -> bool`
- `get(name) -> AgentTool | None`

`create_main_executor() -> UnifiedToolExecutor` — 扫描 hardware + software + 合并 builtin，打印日志返回 executor。

- [ ] **Step 2: 验证导入**

```bash
cd /d/PycharmProjects/sdl_agent && python -c "from core.agent_tools import create_main_executor; e=create_main_executor(); print(e.names); print(len(e.build_openai_tools()),'schemas')"
```
Expected: 输出工具名称列表 + schema 数量（>= 7）

- [ ] **Step 3: Commit**

```bash
git add core/agent_tools.py && git commit -m "feat: add UnifiedToolRegistry and UnifiedToolExecutor"
```

---

### Task 2: 更新 `utils/stream_adapter.py` — 流式 tool_call delta 支持

**Files:** Modify: `utils/stream_adapter.py`

- [ ] **Step 1: 实现 tool_call delta 累积**

新增三个事件常量：
```python
TOOL_CALL_START = "tool_call_start"
TOOL_CALL_ARGS = "tool_call_args"
TOOL_CALL_END = "tool_call_end"
```

新增实例属性 `_pending_tool_calls: list[dict]`，每个 slot = `{index, id, name, args_buf, started}`。

新增方法 `_handle_tool_calls(tool_calls: list)`：
1. 先 flush thinking/text buffer（如果正在输出）
2. 遍历 tool_calls，按 index 确保 slot 存在
3. 累积 `id`、`function.name`、`function.arguments` 到 slot
4. 首次出现 name 时 emit `TOOL_CALL_START`（含 index, name, call_id）
5. 每次 arguments delta emit `TOOL_CALL_ARGS`（含 index, delta）

新增方法 `_flush_tool_calls()` — 遍历 pending，对每个 started 的 slot：
- 尝试 `json.loads(args_buf)` 解析完整 arguments
- emit `TOOL_CALL_END`（含 index, name, call_id, arguments）

新增方法 `get_pending_tool_calls() -> list[dict]` — 返回当前 pending 列表，供 AgentLoop 判断是否有 tool_call。

更新 `_flush()` — 在 flush thinking/text 之后调用 `_flush_tool_calls()`。

- [ ] **Step 2: 验证 delta 累积**

```bash
cd /d/PycharmProjects/sdl_agent && python -c "
from utils.stream_adapter import StreamAdapter
a=StreamAdapter()
ev=list(a.adapt([
 'data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"x\",\"arguments\":\"{\\\"a\\\"\"}}]}]}]}',
 'data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\":1}\"}}]}]}]}',
]))
end=[e for e in ev if e['type']=='tool_call_end']
assert len(end)==1 and end[0]['arguments']=={'a':1}, f'{end}'
print('OK')
"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add utils/stream_adapter.py && git commit -m "feat: add streaming tool_call delta accumulation to StreamAdapter"
```

---

### Task 3: 创建 `core/agent_loop.py` — Agent Loop 引擎

**Files:** Create: `core/agent_loop.py`

- [ ] **Step 1: 实现 AgentLoop**

核心类 `AgentLoop`：

```python
class AgentLoop:
    def __init__(self, llm, executor, model, max_turns=15, extra_body=None): ...

    def run(self, messages, event_callback=None, ask_user_queue=None) -> dict:
        """主循环: while turn < max_turns → stream → check pending → execute → continue"""
        # 1. llm.stream_raw(model, messages, tools=executor.build_openai_tools()) → raw lines
        # 2. StreamAdapter().adapt(raw) → typed events → event_callback(event)
        # 3. 流结束后 adapter.get_pending_tool_calls():
        #    空 → flush text_end → append assistant msg → return {final_message, tool_turns, error}
        #    非空 → flush TOOL_CALL_END → 构建 assistant msg with tool_calls → messages.append
        #    每个 tc: name=="ask_user" && ask_user_queue → agent_question event → queue.get() 阻塞
        #             其他 → executor.dispatch() → tool_result event → messages.append tool result
        # 4. 继续下一轮
```

关键数据结构：
```python
@dataclass
class AgentTurn:
    tool_name: str
    arguments: dict
    result: str
    status: str  # "success"|"error"
```

返回格式：
```python
{"final_message": assistant_msg_dict | None, "tool_turns": [AgentTurn, ...], "error": str | None}
```

- [ ] **Step 2: 实现 SubAgent**

```python
class SubAgent:
    def __init__(self, template_path, executor=None):
        # 加载 YAML: system_prompt, tools, max_turns
        # 从 executor 子集构建自己的 executor
        # 创建独立 LLMClient 实例

    def run(self, task, context=None) -> dict:
        # messages = [system: system_prompt] + [system: context] + [user: task]
        # AgentLoop.run(messages) → return result
```

- [ ] **Step 3: 实现 AgentOrchestrator**

```python
class AgentOrchestrator:
    def __init__(self, templates_dir="prompts/zh/agents"): ...
    def list_templates(self) -> list[str]: ...
    def spawn(self, template, task, context=None) -> dict:
        # SubAgent(template_path).run(task, context) → return result
    def spawn_parallel(self, template, tasks) -> list[dict]:
        # ThreadPoolExecutor, 最多 4 workers, 并行 run，返回按序 results
```

- [ ] **Step 4: 验证导入**

```bash
cd /d/PycharmProjects/sdl_agent && python -c "from core.agent_loop import AgentLoop, AgentOrchestrator, SubAgent; print('OK')"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add core/agent_loop.py && git commit -m "feat: add AgentLoop, SubAgent, AgentOrchestrator"
```

---

### Task 4: 创建子 Agent 模板 + 配置项

**Files:** Create: 4 个 `prompts/zh/agents/*.yaml` | Modify: `core/config.py`, `config.example.json`

- [ ] **Step 1: 创建模板文件**

每个 YAML 格式：
```yaml
name: <template_name>
description: <one-line description>
system_prompt: |
  <full system prompt>
tools: [tool_name, ...]
max_turns: <int>
```

创建 `literature_searcher.yaml`（tools: [search_literature], max_turns: 5）
创建 `experiment_designer.yaml`（tools: [], max_turns: 3，输出 Unified JSON Format）
创建 `summarizer.yaml`（tools: [], max_turns: 3）
创建 `data_analyst.yaml`（tools: [data_statistics, data_normalization, spectrum_analysis], max_turns: 5）

- [ ] **Step 2: 添加配置项**

在 `core/config.py` class body 末尾：
```python
AGENT_MAX_TURNS: int = _external.get("AGENT_MAX_TURNS", 15)
AGENT_ENABLED: bool = _external.get("AGENT_ENABLED", True)
```

在 `config.example.json` 末尾（`}` 前）：
```json
"_Agent配置": "=== Agent tool-use 循环配置 ===",
"AGENT_MAX_TURNS": 15,
"AGENT_ENABLED": true
```

- [ ] **Step 3: Commit**

```bash
git add prompts/zh/agents/ core/config.py config.example.json
git commit -m "feat: add sub-agent templates and agent config options"
```

---

### Task 5: 添加 Agent 路由到 `app.py`

**Files:** Modify: `app.py`

- [ ] **Step 1: 在 import 区域添加**

```python
from core.agent_tools import create_main_executor, AgentTool, UnifiedToolExecutor
from core.agent_loop import AgentLoop, AgentOrchestrator
import queue as queue_module
```

- [ ] **Step 2: 在 `__main__` 区域，app.run() 之前初始化**

```python
if config.AGENT_ENABLED:
    print("[Agent] Initializing agent toolkit...")
    _agent_executor = create_main_executor()
    _agent_orchestrator = AgentOrchestrator()
    print(f"[Agent]   Tools: {_agent_executor.names}")
    print(f"[Agent]   Templates: {_agent_orchestrator.list_templates()}")
else:
    _agent_executor = None
    _agent_orchestrator = None
```

- [ ] **Step 3: 添加模块级 ask_user session 存储**

```python
_agent_ask_queues: dict = {}  # session_id → queue.Queue
```

- [ ] **Step 4: 添加 spawn_agent 实现函数**

```python
def _spawn_agent_impl(template, task, context=None, mode="single", siblings=None) -> dict:
    if _agent_orchestrator is None: return {"result": "Agent 未启用"}
    if mode == "parallel" and siblings:
        results = _agent_orchestrator.spawn_parallel(template, [s["task"] for s in siblings])
        return {"result": "并行执行完成\n" + "\n".join(
            str(r.get("final_message",{}).get("content","")[:200]) for r in results if r
        )}
    result = _agent_orchestrator.spawn(template, task, context)
    return {"result": str(result.get("final_message",{}).get("content", result))}
```

- [ ] **Step 5: 添加 `/api/chat/agent` 路由**

```python
@app.route('/api/chat/agent', methods=['POST'])
def chat_agent():
    # 1. 解析请求 {message, session_id, history}
    # 2. 构建 messages 列表（处理 reasoning_content 保留）
    # 3. 创建 session executor（_agent_executor + spawn_agent tool）
    # 4. 创建 ask_queue，存入 _agent_ask_queues
    # 5. 创建 AgentLoop 实例
    # 6. daemon thread 运行 loop.run(messages, event_callback, ask_queue)
    # 7. 主线程轮询 event queue → SSE yield，用 Event 等 loop 结束
    # 8. cleanup _agent_ask_queues，emit agent_done / agent_error
    # 返回 Response(agent_sse_events(), mimetype="text/event-stream")
```

- [ ] **Step 6: 添加 `/api/chat/agent/respond` 路由**

```python
@app.route('/api/chat/agent/respond', methods=['POST'])
def chat_agent_respond():
    # 1. 取 {session_id, answer}
    # 2. _agent_ask_queues.get(session_id).put(answer)
    # 3. return jsonify({"type":"ok"})
```

- [ ] **Step 7: 验证端点**

```bash
cd /d/PycharmProjects/sdl_agent && python -c "
from app import app
with app.test_client() as c:
    r = c.post('/api/chat/agent', json={'message':'你好','session_id':'t'})
    print('Status:', r.status_code, 'CT:', r.content_type)
"
```
Expected: `Status: 200 CT: text/event-stream`

- [ ] **Step 8: Commit**

```bash
git add app.py && git commit -m "feat: add /api/chat/agent SSE endpoint with tool-use loop"
```

---

### Task 6: 更新前端类型和 API

**Files:** Modify: `frontend/src/types/chat.ts`, `frontend/src/api/chat.ts`

- [ ] **Step 1: 在 `types/chat.ts` 末尾新增类型**

```typescript
// Agent types
export interface ToolCallInfo {
  index: number; name: string; callId: string
  arguments: Record<string,unknown>; result?: string
  status: 'running' | 'done' | 'error'
}
export interface AgentQuestion { question: string; options?: string }
export interface AgentEvent {
  type: string; text?: string; index?: number; name?: string
  call_id?: string; arguments?: Record<string,unknown>
  delta?: string; result?: string; status?: string
  question?: string; options?: string; message?: string
}
```

- [ ] **Step 2: 在 `api/chat.ts` 末尾新增 `sendAgentMessage()` 和 `respondToAgent()`**

`sendAgentMessage(body, callbacks, signal)`:
- fetch POST `/api/chat/agent` → SSE 解析
- 按 event.type 分发到 callbacks: onTextChunk/Complete, onThinkingChunk/Complete, onToolCallStart/Args/End, onToolResult, onAgentQuestion, onError, onDone
- 流结束时 return `{text, error?}`

`respondToAgent(sessionId, answer)`:
- fetch POST `/api/chat/agent/respond` → JSON

- [ ] **Step 3: 验证 TypeScript 编译**

```bash
cd /d/PycharmProjects/sdl_agent/frontend && npx vue-tsc -b --noEmit 2>&1 | head -20
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/types/chat.ts frontend/src/api/chat.ts
git commit -m "feat: add agent types and API wrappers"
```

---

### Task 7: 创建 Agent UI 组件并集成到 ChatContainer

**Files:** Create: 3 个 Vue 组件 | Modify: `ChatContainer.vue`, `InputBar.vue`

- [ ] **Step 1: 创建 `AgentToolCard.vue`**

Props: `{name: string, args?: object, result?: string, status: 'running'|'done'|'error'}`
- 按 name 映射 emoji 图标
- 显示参数前 2 个 key=value
- 状态颜色：running=蓝框, done=绿框, error=红框
- result 截断 300 字符
- scoped CSS

- [ ] **Step 2: 创建 `AgentQuestionCard.vue`**

Props: `{question: string, options?: string}`
Emits: `select(answer: string)`
- 黄色边框警告框
- 问题文本
- 如果 options 是合法 JSON array，渲染选项按钮；点击 emit select
- scoped CSS

- [ ] **Step 3: 创建 `ThinkingBubble.vue`**

Props: `{text: string, duration?: number}`
- `<details>` 折叠/展开
- summary: "🧠 思考过程" + duration "Xs"
- pre 显示 text
- scoped CSS（灰色背景，小字号）

- [ ] **Step 4: 更新 `ChatContainer.vue`**

在 `<script setup>` 中添加 agent 状态：
```typescript
const agentActiveToolCalls = ref<Map<number, ToolCallInfo>>(new Map())
const agentQuestion = ref<AgentQuestion | null>(null)
const agentThinking = ref('')
const agentThinkingDuration = ref(0)
const agentSessionId = ref('sess_' + Date.now())
```

新增方法 `sendToAgent(message)` 和 `handleAgentQuestionAnswer(answer)`。

在 template 中，最后一条 AI 消息气泡之后插入：
```vue
<ThinkingBubble v-if="agentThinking" :text="agentThinking" :duration="agentThinkingDuration" />
<AgentToolCard v-for="[idx, tc] in agentActiveToolCalls" :key="idx" v-bind="tc" />
<AgentQuestionCard v-if="agentQuestion" v-bind="agentQuestion" @select="handleAgentQuestionAnswer" />
```

- [ ] **Step 5: 更新 `InputBar.vue`**

placeholder 计算属性：agentQuestion → "回答 Agent 的问题..."，isStreaming → "AI 正在回复..."，默认 → "输入消息..."
disabled: `isStreaming && !agentQuestion`
发送时：如果 agentQuestion 存在，调 `handleAgentQuestionAnswer(inputText)` 而非正常 send

- [ ] **Step 6: 构建验证**

```bash
cd /d/PycharmProjects/sdl_agent/frontend && npm run build:flask
```
Expected: 构建成功

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/chat/AgentToolCard.vue frontend/src/components/chat/AgentQuestionCard.vue frontend/src/components/chat/ThinkingBubble.vue frontend/src/components/chat/ChatContainer.vue frontend/src/components/chat/InputBar.vue
git commit -m "feat: add agent UI components and ChatContainer integration"
```

---

### Task 8: 端到端验证

- [ ] **Step 1: 启动 Flask** → `python app.py`，确认日志显示 Agent 初始化成功
- [ ] **Step 2: 普通对话** → `curl POST /api/chat/agent -d '{"message":"你好","session_id":"t1"}'` → SSE 流含 thinking_* + text_* + agent_done
- [ ] **Step 3: tool call** → `curl ... '{"message":"帮我搜索钙钛矿文献","session_id":"t2"}'` → SSE 含 tool_call_* + tool_result + agent_done
- [ ] **Step 4: ask_user** → 发模糊请求，确认收到 agent_question → curl POST `/api/chat/agent/respond` 回答 → 确认继续完成
- [ ] **Step 5: 前端构建并浏览器打开** → 确认现有功能不受影响，agent 模式下新气泡渲染正确

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: Phase 1 agent — end-to-end verified"
```

---

## Self-Review

- [x] 8 tools: ask_user, spawn_agent + scan_hardware + scan_software — 完整定义
- [x] Streaming tool_call delta: `_handle_tool_calls` + `_flush_tool_calls` + `get_pending_tool_calls`
- [x] AgentLoop: stream → pending check → execute → append → continue loop
- [x] ask_user: agent_question event + queue.Queue 阻塞 + /respond 唤醒
- [x] spawn_agent: single via AgentOrchestrator.spawn()
- [x] SubAgent: YAML template → SubAgent → AgentLoop.run()
- [x] SSE events: thinking_*, text_*, TOOL_CALL_*, tool_result, agent_question, team_*, agent_done, agent_error
- [x] Frontend: types + API + 3 new components + ChatContainer/InputBar integration
- [x] Backward compat: /api/chat 不变，前缀路由不变
- [x] 无 TBD/TODO/placeholder
