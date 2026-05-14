# SDL Agent Frontend (Vue 3 / TypeScript)

基于 Vue 3 + TypeScript + Vite 6 重构的前端，取代 `templates/index.html` 旧版 UI，通过 `/v2` 路由提供。

## 技术栈

| 类别 | 选型 |
|------|------|
| 框架 | Vue 3.5 (Composition API + `<script setup>`) |
| 语言 | TypeScript 5.6 |
| 构建 | Vite 6 |
| 路由 | Vue Router 4 (hash mode) |
| 状态管理 | Pinia 2 |
| 图标 | lucide-vue-next |
| 后端代理 | Vite dev proxy → Flask `127.0.0.1:5000` |

## 项目结构

```
frontend/
├── src/
│   ├── main.ts                  # 入口：挂载 app + router + pinia
│   ├── App.vue                  # 根组件（layout shell）
│   ├── router.ts                # 路由表（hash mode）
│   ├── env.d.ts                 # TypeScript 声明
│   ├── api/                     # HTTP API 封装
│   │   ├── client.ts            # fetch 封装 + 错误处理
│   │   ├── chat.ts              # /api/chat + uploadPDF
│   │   ├── experiment.ts        # /api/experiment_*, /api/compile_*
│   │   ├── hardware.ts          # /api/hardware_*
│   │   ├── search.ts            # /api/semantic_search, /api/page_image
│   │   ├── analysis.ts          # /api/list_algorithms, /api/run_analysis
│   │   └── history.ts           # /api/history/*
│   ├── stores/                  # Pinia 状态管理
│   │   ├── chat.ts              # 对话消息 + 提取模式 + SSE + 字段确认
│   │   ├── experiment.ts        # 实验设计（步骤 CRUD、嵌套计算、折叠）
│   │   ├── hardware.ts          # 硬件工具列表 + 单步执行
│   │   ├── search.ts            # 语义搜索
│   │   ├── analysis.ts          # 数据分析
│   │   ├── layout.ts            # 全局 UI 状态（侧栏、任务进度、绿点确认）
│   │   └── theme.ts             # 主题切换
│   ├── composables/
│   │   └── useSSE.ts            # SSE (Server-Sent Events) composable
│   ├── components/
│   │   ├── common/              # 通用组件（Badge, LoadingSpinner, EmptyState, …）
│   │   ├── layout/              # 布局组件（Sidebar, TopBar, HistoryPanel, NavPanel, TaskPanel）
│   │   ├── chat/                # 对话组件
│   │   │   ├── ChatContainer.vue # 对话容器（消息列表 + 确认卡片 + InputBar）
│   │   │   ├── MessageBubble.vue # 消息气泡
│   │   │   └── InputBar.vue      # 输入栏（模式切换 + 气泡 + 发送/中断）
│   │   ├── experiment/          # 实验设计组件
│   │   │   ├── ElementPanel.vue # 左侧元素面板（工具/算法/辅助）
│   │   │   ├── StepCanvas.vue   # 步骤画布（gutter + 嵌套 + 折叠）
│   │   │   ├── StepCard.vue     # 步骤卡片
│   │   │   ├── StepEditor.vue   # 步骤参数编辑器
│   │   │   └── CodeArea.vue     # 底部代码区（JSON/Python 切换）
│   │   ├── search/              # 语义搜索组件（SearchBar, SearchResultList, SearchResultCard, PagePreview）
│   │   ├── modals/              # 弹窗组件（ModalContainer, ConfirmDialog, …）
│   │   └── cards/               # 卡片组件
│   ├── pages/                   # 页面级组件
│   │   ├── ChatPage.vue         # 对话页（主区域）
│   │   ├── ExperimentPage.vue   # 实验设计
│   │   ├── ExtractionPage.vue   # 文献提取（搜索 UI + PDF 预览面板）
│   │   ├── HardwarePage.vue     # 硬件控制
│   │   └── AnalysisPage.vue     # 数据分析
│   └── types/                   # TypeScript 类型定义
├── index.html                   # Vite 入口 HTML
├── package.json
├── vite.config.ts
└── tsconfig.json
```

## 路由

所有页面通过 RouterView（ChatPage 常驻） + TaskPanel（右侧动态加载）组合渲染：

| 区域 | 路径 | 组件 |
|------|------|------|
| 主区域（RouterView） | `/v2` | ChatPage |
| 右侧面板（TaskPanel） | 动态加载 | ExtractionPage / HardwarePage / ExperimentPage / AnalysisPage |

## 开发与测试

```bash
cd frontend
npm install
npm run dev              # http://localhost:5173/v2（API 代理到 :5000）
npx vue-tsc -b           # 类型检查
npm run build:flask      # 生产构建（base=/v2-static/）
cd .. && python platform_init/test/frontend/test_frontend.py  # 10 项集成测试
```

### 调试：配置变更不生效（改 config.json 后前端仍用旧值）

**技术路线**：前端 Vue SPA → `/api/*` fetch → Vite 代理（dev）或同源（prod）→ Flask `127.0.0.1:5000` → `core/config.py` 的 `_external` dict（模块导入时一次性从 config.json 加载到内存）。

**关键认知**：`config.json` 只在 `python app.py` 启动时读取一次，Flask 进程内存中缓存全量配置。改 config 后必须重启 Flask（`Ctrl+C` → `python app.py`），不存在热重载。

**验证方法**：
1. 在 `app.py` 的 `/api/chat` 路由入口加 `print(f"[DEBUG] MODEL_NAME_TALK={Config.MODEL_NAME_TALK}")`
2. 重启后发消息，看终端有无该输出 — 无输出说明请求未到达本进程（端口残留）
3. 用 `platform_init/test/api_test/api_test.py` 直接调 API 验证 config 读取正确性（独立进程，不受残留影响）
4. `netstat -ano | grep ":5000.*LISTENING"` 确认只有一个进程

**故障排查决策树**：
```
改 config.json 后前端仍用旧值
  ├─ 终端有 [DEBUG /api/chat] 输出？
  │   ├─ 是 → 看 MODEL_NAME_TALK 值是否正确
  │   │   ├─ 正确 → 问题在更下游（API 调用、模型提供商路由）
  │   │   └─ 错误 → 检查环境变量覆盖（env var > config.json）
  │   └─ 否 → 请求未到达本进程
  │       └─ netstat 查端口 → 多进程 → kill 旧进程 → 重启
  └─ 仍不行 → 重启电脑（最常见兜底方案）
```

### 坑：git checkout 后 /v2 白屏或 404

**原因**：`frontend/dist/` 虽在 `.gitignore`，但早期有部分文件（`index.html`、`Badge-*.css` 等）被误提交到 git。checkout 时这些旧文件会覆盖本地构建产物，导致 `index.html` 引用的 JS/CSS 文件名与实际构建产物不匹配。

**症状**：
- 访问 `/v2` 页面白屏（浏览器加载 JS 404）
- 或 Flask 返回 404（`index.html` 被删掉）

**修复**：
```bash
cd frontend && npm run build:flask
```

每次 `git checkout` 切换分支/修订后，如果 dist 被污染，重新构建即可。

### 坑：前端-后端集成 500 错误（`{"error":"服务器内部错误"}`）

**原因 1 — Windows 中文打印崩溃**：`python app.py` 运行时 `sys.stdout.encoding` 默认为 `cp936`/`gbk`，`print()` 输出含特殊 Unicode 字符的 LLM 响应时抛出 `UnicodeEncodeError`，被 Flask 500 handler 捕获。

**修复**：`app.py` 最顶部（所有 import 之前）已加入：
```python
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
```

**原因 2 — 端口残留**：多次启动 Flask 后可能有旧进程残留端口 5000，新代码不生效。

**排查**：
```bash
netstat -ano | findstr ":5000.*LISTENING"   # 检查端口占用
taskkill //F //PID <pid>                     # 杀掉旧进程
```
不要用 `taskkill //F //IM python.exe`（会杀掉自己的测试脚本）。

**原因 2 延伸 — 改 config.json 后前端仍使用旧模型**：症状是改了 `config.json` 的模型名，重启 Flask 后前端聊天 AI 仍自称旧模型，且终端无新代码的 debug 输出。根因同样是旧 `python.exe` 进程仍占着 5000 端口，所有 `/api/*` 请求打到旧进程（内存中缓存着旧 config 值），新进程或绑定失败或虽然绑定成功但请求被旧进程抢先处理。现象类似"代码不生效"或"前端调了另一个后端"，实际上就是端口残留。

**排查**：同上 `netstat -ano | findstr ":5000.*LISTENING"`，杀掉所有 LISTENING 的 PID 后重启。若 taskkill 杀不掉，重启电脑是最可靠的方式。

**原因 3 — LLM 返回空 JSON**：`LongCat-Flash-Thinking` 模型在 JSON 前输出推理文本，简单 `json.loads(content)` 失败。`core/field_inference.py:389` 的 `_parse_experiment_json()` 已实现多策略解析（清理 markdown → 提取 `{...}` 块）。

> 完整记录见 `DEBUG_INTEGRATION_GUIDE.md`

### 坑：前端不要构造对用户的回应文本

- **AI 回复、实验结果、错误说明**等文本由 Python 后端 `reply` 字段统一返回，前端直接使用
- **按钮标签、placeholder、loading 提示**等纯 UI 文案前端自行管理，但不要与 `app.py` 中已定义的文案重复
- 具体的前端-后端数据流和接口规范见 `DEBUG_INTEGRATION_GUIDE.md`

## 技术路线与操作方法

### 前端-后端分工

```
用户输入 → TS（UI层，传参） → Python（业务层，处理） → TS（UI层，展示）
```

| 职责 | TypeScript | Python |
|------|-----------|--------|
| 用户输入 | 拼接模式前缀、构造请求体 | — |
| 业务逻辑 | — | LLM 调用、JSON 解析、数据处理 |
| 对用户回应 | **仅展示** `data.reply` | **统一生成**所有 AI 回复、结果说明、错误文本 |
| Canvas 更新 | `loadFromJSON()` 渲染步骤 | `json_to_visual()` 生成 nodes/edges |
| UI 文案 | 按钮标签、placeholder（不跟后端重复） | — |

### 实验设计开发流程

1. **参考旧版实现**：`templates/static/js/chat/chat.js` + `experiment/experiment_chat.js` 是经过验证的参考实现
2. **后端先行**：先用 Flask test client 验证 `/api/experiment_chat` 返回正确
3. **前端对接**：按下方数据流实现 `chat.ts` 模式处理，**不构造回应文本**
4. **验证**：`npx vue-tsc --noEmit` → `npm run build:flask` → 浏览器测试

### 新增模式的通用步骤

1. 在 `chat.ts` 的 `MODE_PREFIX` 和 `MODE_LABEL` 中注册
2. 在 `app.py` 的 `/api/chat` 中添加前缀检测 → 分发到 handler
3. Handler 返回 `{type, reply, ...}` — `reply` 包含所有用户可见文本
4. 前端 `send()` 中按 `result.type` 分支处理，调用具体 API
5. 前端 **不构造回应文本**，直接使用 `expData.reply`

### 日常操作

```bash
# 启动后端
python app.py                              # http://127.0.0.1:5000

# 前端开发
cd frontend && npm run dev                 # http://localhost:5173/v2（代理到 :5000）

# 类型检查 + 构建
cd frontend && npx vue-tsc --noEmit        # 类型检查
cd frontend && npm run build:flask         # 生产构建（输出到 dist/）

# 测试后端 API（无需浏览器）
python -c "
from app import app
with app.test_client() as c:
    resp = c.post('/api/experiment_chat',
        data=json.dumps({'message': '设计一个旋涂实验'}),
        content_type='application/json')
    print(resp.status_code, resp.get_json()['type'])
"

# 检查端口残留
netstat -ano | findstr ":5000.*LISTENING"   # 查看占用进程
taskkill //F //PID <pid>                     # 清理

# 完整测试
python test/experiment_stream_test/test_stream.py  # 6 项集成测试
```

### 实验设计完整数据流

```
用户输入 "设计旋涂实验"
  │
  ▼
[1] TS  chatStore.send() → 拼接 "实验设计：" 前缀
       └─ POST /api/chat { message, action, history }
  │
  ▼
[2] Py  /api/chat → handle_hardware_request()
       └─ 返回 { type: "experiment_design_mode", command, reply }
  │
  ▼
[3] TS  sendChatMessage 将 data.reply 写入消息气泡
       └─ result.type === 'experiment_design_mode'
       └─ generateExperiment(cmd)
  │
  ▼
[4] TS  POST /api/experiment_chat { message: cmd }
  │
  ▼
[5] Py  ExperimentDesignAgent()
       └─ parse_experiment_design() → LLM → _parse_experiment_json()
       └─ json_to_visual() → nodes/edges
       └─ 返回 { type: "experiment_design", experiment_json, visual_data, reply }
  │
  ▼
[6] TS  expStore.loadFromJSON(experiment_json)  更新 canvas
       └─ addMessage('ai', reply)               显示结果
```

> 详细接口规范、错误处理、调试清单见 `DEBUG_INTEGRATION_GUIDE.md`

---

## 更新日志（2026-05-10）

### 1. 文献提取与语义搜索合并

**删除**：`pages/SearchPage.vue`、`stores/extraction.ts`、`api/extraction.ts`

**Sidebar/HistoryPanel/NavPanel/TaskPanel**：移除 "语义搜索" 入口，只保留 "文献提取"

**ExtractionPage 重写**为搜索界面：
- SearchBar（顶部搜索框） + SearchResultList（结果列表） + PagePreview（页面预览）
- "提取此页" 按钮 → 切换到对话页的提取模式

### 2. 对话页模式气泡系统

输入框左侧可显示模式气泡，支持四种模式自动添加前缀：

| 模式 | 气泡 | 自动添加前缀 | 后端路由 |
|------|------|------------|---------|
| 文献提取 | 📄 文献提取 | `帮我搜寻：` | `handle_extraction_request` |
| 硬件控制 | ⚙️ 硬件控制 | `硬件控制：` | `handle_hardware_request` |
| 实验设计 | 🧪 实验设计 | `实验设计：` | `handle_hardware_request` |
| 数据分析 | 📈 数据分析 | `数据分析` | `handle_data_analysis` |

- 点击工具栏模式按钮 → 气泡出现 + 按钮高亮；再次点击 → 回到普通模式
- 发送消息后自动回到普通模式（提取模式除外，需等待两轮确认）
- 气泡可 × 关闭；空输入在提取模式下按 Enter 触发默认 FAPbI3 提取

**实现**（`stores/chat.ts`）：
```ts
export const MODE_PREFIX: Record<ChatMode, string> = {
  normal: '', extraction: '帮我搜寻：', hardware: '硬件控制：',
  experiment: '实验设计：', analysis: '数据分析',
}
export const MODE_LABEL: Record<ChatMode, string> = {
  normal: '', extraction: '📄 文献提取', hardware: '⚙️ 硬件控制',
  experiment: '🧪 实验设计', analysis: '📈 数据分析',
}
```

### 3. 两轮提取确认流程

匹配旧版 `templates/` 的提取交互：

1. **Round 1**：发送 `"帮我搜寻：<描述>"` → 后端 LLM 推断字段 → 返回 `field_confirm`
2. **内联确认卡片**：在 AI 消息下方显示字段标签 + 两个操作按钮
3. **Round 2**：点击 "✅ 确认提取" → 发送 `action: 'start_extraction'` → 后端启动提取线程 → 返回 `task_trigger` → 连接 SSE
4. 点击 "❌ 修改要求" → 重新进入提取模式，可输入补充说明

**交互式字段编辑**：确认卡片中字段可双击编辑、× 删除、底部输入框添加新字段，确认时使用编辑后的字段列表。

**后端历史记忆**：修改要求时对话历史传递给 LLM（`handle_extraction_request` → `infer_fields` → prompt 前追加 history），让 LLM 根据上文调整字段。

### 4. PDF 预览面板

PDF 页面预览从对话气泡移到 ExtractionPage 右侧面板：

- **提取开始时立即弹出**，显示 "等待连接…" → 首帧到达后显示 PDF 页面
- 深色头部：`AI 正在阅读...` + 文件名 + 页码
- 绿色扫描线动画（当 `extractionRunning` 时）
- × 可关闭；关闭后出现 "显示 PDF 预览" 虚线按钮可重新打开
- 参考旧版 `templates/index.html` 的 `#pdf-panel` 设计

### 5. 提取中断模式

提取运行时输入框行为：

- 发送按钮 → 红色转圈按钮（`.cancel-mode` + `.btn-spinner`），点击调用 `/api/cancel_task`
- 输入框禁用，placeholder："提取任务运行中..."
- SSE 连接断开 → 状态复位

**实现**（`stores/chat.ts`）：`cancelExtractionTask()` 调用 cancel API + `extractionDisconnect?.()` 断开 SSE。

### 6. SSE 事件路由

`connectExtractionSSE()` 处理所有 SSE 事件类型：

| SSE type | 处理 |
|----------|------|
| `info` / `progress` | `layout.updateTaskStatus('extraction', 'running')` |
| `page_reading` | 设置 `currentPage`（触发 PDF 预览） |
| `reading_start` | 清空 `currentPage` |
| `finding` | 格式化 `details` 键值对 → `addMessage('ai', …)` |
| `complete` | 停用 `extractionRunning`，显示摘要 |
| `error` | 停用状态，显示错误消息 |

### 7. 绿点确认

任务完成后 NavPanel 图标显示绿点，点击任务图标后绿点消失（`updateTaskStatus` → `acknowledgeTask`）。

### 8. 后端改进

**JSON 解析增强**（`extract/extraction_engine.py`）：
- Prompt 强化：明确 JSON 输出规则（转义双引号、禁用中文引号、禁止尾随逗号）
- 三重回退解析策略：
  - 策略 1：标准正则提取 + `json.loads`
  - 策略 2：`_fix_common_json_errors()` 修复常见错误后重试
  - 策略 3：`_extract_json_heuristic()` 启发式提取 `"data": [...]` 块

**历史记忆**（`core/field_inference.py` + `app.py`）：
- `infer_fields(task_description, history)` 接受对话历史
- 修改字段时 LLM 可参考之前的建议进行调整

### 9. 实验设计面板优化

参见上文 "实验设计面板：步骤画布优化" 章节。

### 10. 后端 Prompt 集中管理系统

后端所有 LLM prompt 已从业务代码中提取到 `prompts/` 目录，16 个 YAML 文件覆盖 5 个业务模块。

**前端可用的新 API**：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/prompts` | GET | 列出所有 prompt 元信息 |
| `/api/prompts/<name>` | GET | 获取单个 prompt 详情 |
| `/api/prompts/<name>` | PUT | 修改 prompt，写入 overrides |
| `/api/prompts/<name>/reset` | POST | 撤销修改 |
| `/api/prompts/reload` | POST | 重新加载全部 |
| `/api/prompts/optimize` | POST | LLM 优化建议 |
| `/api/prompts/test` | POST | 试跑测试 |

前端如需 prompt 管理面板，直接调用这些 API 即可，无需新增后端路由。

### 11. 提取结果来源追踪 + PDF 预览

每条提取结果自动附带 `_source_doc`（PDF 文件名）+ `_source_page`（页码）。

**新增 PDF 预览 API**：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/page_preview` | GET | 获取 PDF 页面图片 + 文本 + 关键词高亮。参数：`?doc=...&page=...&query=...` |
| `/api/page_context` | POST | Batch 读取页面上下文。body：`{"results": [{"doc":..., "page":..., "query":...}]}` |

前端可在提取结果表格中为每条记录添加"查看来源"按钮，点击后调 `/api/page_preview` 打开 PDF 预览面板，对照原文核验准确性。

### 12. 提取质量检查

后端新增 `extract/quality_checker.py`，在保存 CSV 前自动执行：
- 稀疏检测：字段填充率 < 30% 的记录自动删除
- 重复检测：完全一致或子集关系的记录保留信息量最大的

配置项：`QUALITY_CHECK_ENABLED`、`QUALITY_SPARSE_THRESHOLD`。

### 整体数据流

```
用户输入 → InputBar.vue → ChatStore.send() → POST /api/chat
                                                    ↓
                        ┌───────────────────────────┴───────────────────────┐
                        ↓                          ↓                          ↓
              普通对话 (chat)             实验设计 (experiment)          文献提取 (extraction)
              handle_normal_chat          handle_hardware_request        handle_extraction_request
              (流式 SSE)                  (返回 experiment_design_mode)  (返回 field_confirm)
                                                   ↓                          ↓
                                          ChatStore 检测到              用户确认 →
                                          experiment_design_mode        action: start_extraction
                                                   ↓                          ↓
                                          调用 generateExperiment()     SSE /api/task_stream
                                          → /api/experiment_chat        → findings 推送到对话气泡
                                                   ↓
                                          ExperimentDesignAgent
                                          .parse_experiment_design()
                                          （三重回退 JSON 解析）
                                                   ↓
                                          推送到 ExperimentStore
                                          .loadFromJSON()
                                                   ↓
                                          聊天框输出步骤摘要
```

### 各模式路由表

| 模式 | 前端前缀 | 第一跳 (后端) | 第二跳 (后端) | 前端结果处理 |
|------|---------|-------------|-------------|------------|
| 普通对话 | 无 | `handle_normal_chat` → 流式 SSE | — | 字符逐步追加到 AI 气泡 |
| 文献提取 | `帮我搜寻：` | `handle_extraction_request` → `field_confirm` | 用户确认 → `start_extraction` → SSE `/api/task_stream` | findings 推送到对话气泡，PDF 预览在右侧面板 |
| 实验设计 | `实验设计：` | `handle_hardware_request` → `experiment_design_mode` | `ChatStore` 自动调用 `/api/experiment_chat` | JSON 推送到实验面板，步骤摘要到聊天框 |
| 硬件控制 | `硬件控制：` | `handle_hardware_request` → `hardware_confirm` | 用户确认 → `start_hardware` | MQTT 指令执行 |
| 数据分析 | `数据分析` | `handle_data_analysis` → 智能交互 | — | 算法列表 / 执行结果 |
| 算法生成 | `生成算法：` | `handle_generate_algorithm` | — | 算法代码 |

### 实验设计完整调用链

```
1. 用户在输入框选择 "🧪 实验设计" 模式
2. 输入 "帮我设计一个实验" 并按 Enter
3. InputBar.submit() → ChatStore.send()
4. MODE_PREFIX['experiment'] = '实验设计：' → finalText = '实验设计：帮我设计一个实验'
5. POST /api/chat {message: '实验设计：帮我设计一个实验', action: '', history: [...]}
6. 后端 chat() 检测 startswith("实验设计：") → handle_hardware_request()
7. handle_hardware_request() 检测 mode="design" → 返回 {type: 'experiment_design_mode', command: '帮我设计一个实验', reply: '...'}
8. ChatStore.send() 检测 result.type === 'experiment_design_mode' → 显示 "🔬 AI 正在设计实验方案..."
9. 调用 generateExperiment(command) → POST /api/experiment_chat {message: '帮我设计一个实验'}
10. 后端 ExperimentDesignAgent.parse_experiment_design() 调用 LLM（10-15 秒）
11. LLM 返回 JSON → 三重回退解析 → 验证格式 → 转换 visual_data
12. 返回 {type: 'experiment_design', experiment_json: {...}, visual_data: {...}, reply: '...'}
13. ChatStore 接收 expData → expStore.loadFromJSON(json) 推送到实验面板
14. 聊天框显示步骤摘要：实验名 + 步骤列表
```

### 调试方法

**后端快速验证（无需浏览器、无需启动 Flask）：**
```bash
cd D:/PycharmProjects/SDL_agent
python -c "
import sys, json; sys.path.insert(0, '.')
from app import app
with app.test_client() as c:
    # 测试实验设计
    resp = c.post('/api/experiment_chat',
        data=json.dumps({'message': '设计一个旋涂实验'}),
        content_type='application/json')
    print(resp.status_code)
    data = resp.get_json()
    print(data.get('type'))
"
```

**前端类型检查与构建：**
```bash
cd frontend
npx vue-tsc -b          # 类型检查（确保无 TS 错误）
npm run build:flask      # 生产构建（输出到 dist/）
```

**常见问题排查：**

| 症状 | 可能原因 | 排查方法 |
|------|---------|---------|
| `❌ 实验设计失败：{"error":"服务器内部错误"}` | Flask 500 错误 | 检查 Flask 控制台 traceback；常见：`rich`/`fastmcp` 版本冲突、JSON 解析失败 |
| 实验设计模式无反应 | Flask 未重启（Python 模块缓存） | Ctrl+C 停止 → 清除 `__pycache__` → `python app.py` 重启 |
| JSON 解析失败 | LLM 返回了非标准格式 | 查看控制台 `[实验设计] LLM原始输出` 日志 |
| 前端变更不生效 | 未重新构建 | `cd frontend && npm run build:flask` |
| `RichHandler` 崩溃 | `rich` 版本过低 | `pip install --upgrade rich` → 15.0.0 |
| TypeScript 编译错误 | 类型定义不同步 | `cd frontend && npx vue-tsc -b` 查看具体错误 |

### 关键文件速查

| 功能 | 后端文件 | 前端文件 |
|------|---------|---------|
| 实验设计生成 | `core/field_inference.py:ExperimentDesignAgent` | `stores/chat.ts:send()` (experiment 分支) |
| JSON 解析 | `core/field_inference.py:_parse_experiment_json()` | — |
| 格式转换 | `experiment/format.py:ExperimentFormatConverter` | — |
| 实验面板 | — | `stores/experiment.ts`, `pages/ExperimentPage.vue` |
| API 请求封装 | `app.py:/api/experiment_chat` | `api/experiment.ts:generateExperiment()` |
| API 类型定义 | — | `api/chat.ts:JsonResponse`, `ChatResult` |
| 模式气泡 | — | `stores/chat.ts:MODE_PREFIX`, `MODE_LABEL` |
| SSE 任务流 | `app.py:/api/task_stream` | `composables/useSSE.ts` |
