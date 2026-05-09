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
