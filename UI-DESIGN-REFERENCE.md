# SDL_agent 前端重构 —— 设计参考文档

Status: **规划中 (2026-05-07)**
技术栈: **Vite 6 + Vue 3.5 + TypeScript 5 + Pinia 2 + Vue Router 4**
架构: **1 个 SPA，6 条路由，1 个 Vue Router**

---

## 目标

1. **Phase 4 语义搜索前端**: 搜索栏、结果卡片、页面预览、"从此页提取"
2. **前端全量重构**: 从 17 个全局 JS 文件 + 620 行硬编码 CSS → Vue 3 + TypeScript + Vite 单页面项目
3. **覆盖所有后端功能**: 聊天、文献提取、硬件控制、实验设计、数据分析、算法生成

### 选型理由

- **不选原生 JS 渐进**: DOM 操作代码转为 SPA 全部丢弃，写两遍 JS。CSS 变量在任何方案都 100% 复用
- **不选 MPA 多页面**: 6 页之间侧边栏切换就是组件切换，不是页面导航，独立打包浪费资源
- **选单 SPA**: 组件全共享，侧边栏只渲染一次，Flask 只需一个 fallback route

---

## 布局设计（参考 DeepSeek / 豆包 / 通义千问）

### 主页面布局 (Chat)

```
┌──────────────────────────────────────────────────────┐
│  TopBar                                    [🌓]     │
├──────────┬───────────────────────────────────────────┤
│          │  对话区域 (ChatContainer)                  │
│ Sidebar  │  ├─ MessageBubble (user)                  │
│          │  ├─ MessageBubble (ai)                    │
│  💬 对话  │  ├─ MessageStreaming (流式)               │
│  📊 搜索  │  └─ InputBar                             │
│  📄 提取  │                                           │
│  ⚙️ 硬件  │  右侧面板 (可滑出)                         │
│  🧪 实验  │  ├─ PDF 预览                              │
│  📈 分析  │  ├─ 步骤面板                              │
│          │  └─ 实验设计画布                            │
└──────────┴───────────────────────────────────────────┘
```

### 搜索页面布局 (Phase 4)

```
┌──────────────────────────────────────────────────────┐
│  TopBar                                    [🌓]     │
├──────────┬───────────────────────────────────────────┤
│          │  🔍  [搜索全文献库...              ]       │
│ Sidebar  ├───────────────────────────────────────────┤
│          │  共 N 页已索引                              │
│          ├───────────────────────────────────────────┤
│          │  ┌──────────────────────────────────┐     │
│          │  │ nature_xxx.pdf  p.3      0.85   │     │
│          │  │ "The passivation of..."          │     │
│          │  │          [查看页面] [提取此页]    │     │
│          │  └──────────────────────────────────┘     │
│          │  ┌──────────────────────────────────┐     │
│          │  │ ...                               │     │
│          │  └──────────────────────────────────┘     │
└──────────┴───────────────────────────────────────────┘
```

---

## 技术栈

| 层 | 技术 | 说明 |
|----|------|------|
| 构建 | Vite 6 | 开发 HMR + 生产构建 |
| UI 框架 | Vue 3.5 | Composition API + `<script setup>` |
| 语言 | TypeScript 5 | 严格模式 |
| 状态管理 | Pinia 2 | 按页面/功能拆分 store |
| 路由 | Vue Router 4 | 6 条路由，hash 或 history 模式 |
| 样式 | CSS 变量 + scoped | 无需 Tailwind |
| HTTP | fetch 封装 | `api/` 层统一处理 |
| 图标 | SVG inline | lucide 图标集 |

**不引入**: UI 组件库 (Ant Design / Element Plus)、TailwindCSS

---

## 路由设计

```typescript
// router.ts
const routes = [
  { path: '/',           name: 'chat',        component: ChatPage },
  { path: '/search',     name: 'search',      component: SearchPage },
  { path: '/extraction', name: 'extraction',  component: ExtractionPage },
  { path: '/hardware',   name: 'hardware',    component: HardwarePage },
  { path: '/experiment', name: 'experiment',  component: ExperimentPage },
  { path: '/analysis',   name: 'analysis',    component: AnalysisPage },
]
```

- 1 个 `index.html`、1 个 `main.ts`、1 个 `createApp`
- 侧边栏在 `App.vue` 渲染一次，`<RouterView>` 切换页面内容
- 所有组件共享，无重复打包

---

## 项目结构

```
frontend/
├── package.json
├── tsconfig.json
├── vite.config.ts
├── index.html                          # Vite 入口
│
├── src/
│   ├── main.ts                         # createApp 挂载
│   ├── App.vue                         # Sidebar + TopBar + <RouterView>
│   ├── router.ts                       # 6 条路由
│   │
│   ├── types/                          # TypeScript 类型定义
│   │   ├── api.ts                      # API 响应类型
│   │   ├── chat.ts                     # 消息/流式类型
│   │   ├── extraction.ts               # 提取字段/结果类型
│   │   ├── hardware.ts                 # 工具/参数类型
│   │   ├── experiment.ts               # 实验步骤/JSON 类型
│   │   └── search.ts                   # 搜索结果类型 (Phase 4)
│   │
│   ├── api/                            # API 请求层
│   │   ├── client.ts                   # fetch 封装 + 错误处理
│   │   ├── chat.ts                     # /api/chat
│   │   ├── extraction.ts               # /api/upload, /api/task_stream
│   │   ├── search.ts                   # /api/semantic_search, /api/page_image
│   │   ├── hardware.ts                 # /api/hardware_tools
│   │   ├── experiment.ts               # /api/experiment_chat, /api/compile_experiment
│   │   ├── analysis.ts                 # /api/list_algorithms, /api/run_algorithm
│   │   └── history.ts                  # /api/history/save_batch
│   │
│   ├── stores/                         # Pinia stores
│   │   ├── chat.ts                     # 聊天状态 + 消息列表
│   │   ├── extraction.ts               # 提取任务状态
│   │   ├── search.ts                   # 搜索状态 (Phase 4)
│   │   ├── hardware.ts                 # 硬件工具 + 运行状态
│   │   ├── experiment.ts               # 实验设计步骤 + 画布
│   │   ├── theme.ts                    # 主题切换 (亮色/暗色)
│   │   └── notification.ts             # 全局通知队列
│   │
│   ├── composables/                    # 可复用逻辑 (Composition API)
│   │   ├── useSSE.ts                   # SSE 事件流封装
│   │   ├── useStreaming.ts             # 流式文本逐字渲染
│   │   ├── useFileUpload.ts            # 文件上传 + 拖拽
│   │   ├── useKeyboard.ts              # 快捷键注册
│   │   └── useTheme.ts                 # 主题 hook
│   │
│   ├── pages/                          # 页面组件 (6 routes → 6 .vue)
│   │   ├── ChatPage.vue
│   │   ├── SearchPage.vue              # Phase 4
│   │   ├── ExtractionPage.vue
│   │   ├── HardwarePage.vue
│   │   ├── ExperimentPage.vue
│   │   └── AnalysisPage.vue
│   │
│   ├── components/                     # 共享组件
│   │   ├── layout/
│   │   │   ├── AppLayout.vue           # 通用页面布局
│   │   │   ├── Sidebar.vue             # 侧边导航栏
│   │   │   └── TopBar.vue              # 顶部操作栏
│   │   ├── chat/
│   │   │   ├── ChatContainer.vue       # 聊天区主容器
│   │   │   ├── MessageBubble.vue       # 消息气泡 (user/ai)
│   │   │   ├── MessageStreaming.vue    # 流式消息 (逐字动画)
│   │   │   ├── ThinkingBlock.vue       # AI 思考过程块 (可折叠)
│   │   │   └── InputBar.vue            # 底部输入区
│   │   ├── cards/
│   │   │   ├── ResultCard.vue          # 通用结果卡片
│   │   │   ├── FileCard.vue            # 文件信息卡片
│   │   │   └── StatsCard.vue           # 统计数值卡片
│   │   ├── modals/
│   │   │   ├── ModalContainer.vue      # 弹窗容器 (teleport)
│   │   │   ├── FileSelector.vue        # 文件选择器
│   │   │   ├── ConfirmDialog.vue       # 确认对话框
│   │   │   └── SummaryModal.vue        # 提取完成摘要
│   │   ├── search/                     # Phase 4 搜索组件
│   │   │   ├── SearchBar.vue           # 搜索栏
│   │   │   ├── SearchResultList.vue    # 搜索结果列表
│   │   │   ├── SearchResultCard.vue    # 单条搜索结果卡片
│   │   │   └── PagePreview.vue         # PDF 页面图片预览 (右侧滑出)
│   │   ├── experiment/
│   │   │   ├── CanvasArea.vue          # 实验设计画布
│   │   │   ├── StepNode.vue            # 步骤节点
│   │   │   ├── CodeEditor.vue          # JSON/Python 代码查看器
│   │   │   └── ExperimentToolbar.vue   # 工具栏
│   │   ├── analysis/
│   │   │   ├── AlgorithmList.vue       # 算法列表
│   │   │   └── AnalysisResult.vue      # 分析结果展示
│   │   └── common/
│   │       ├── StatusDot.vue           # 状态指示灯
│   │       ├── LoadingSpinner.vue      # 加载动画
│   │       ├── Badge.vue               # 标签/徽章
│   │       ├── Tooltip.vue             # 工具提示
│   │       ├── EmptyState.vue          # 空状态占位
│   │       └── IconButton.vue          # 图标按钮
│   │
│   └── styles/                         # 全局样式
│       ├── tokens.css                  # CSS 变量 (亮色 + 暗色)
│       ├── reset.css                   # 重置 + 排版
│       ├── layout.css                  # 通用布局类
│       ├── scrollbar.css               # 滚动条美化
│       ├── animations.css              # 关键帧动画
│       └── main.css                    # @import 以上所有
│
└── public/                             # 静态资源
    └── favicon.ico
```

---

## CSS 设计 Token

```css
:root {
  --color-primary: #2563eb;
  --color-primary-soft: rgba(37, 99, 235, 0.15);
  --color-primary-mute: rgba(37, 99, 235, 0.06);
  --color-success: #10b981;
  --color-warning: #f59e0b;
  --color-error: #ef4444;
  --color-text: #1f2937;
  --color-text-secondary: #6b7280;
  --color-text-tertiary: #9ca3af;
  --color-bg: #f3f4f6;
  --color-bg-soft: #f9fafb;
  --color-bg-mute: #e5e7eb;
  --color-surface: #ffffff;
  --color-border: rgba(0, 0, 0, 0.08);
  --color-border-strong: rgba(0, 0, 0, 0.15);
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 16px;
  --space-xs: 4px; --space-sm: 8px; --space-md: 12px;
  --space-lg: 16px; --space-xl: 20px; --space-2xl: 24px;
  --transition-fast: 0.15s ease;
  --transition-normal: 0.2s ease;
  --transition-slow: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
  --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
  --shadow-lg: 0 10px 25px rgba(0,0,0,0.05);
  --navbar-height: 64px;
  --sidebar-width: 260px;
  --panel-width: 360px;
}

[data-theme="dark"] {
  --color-primary: #3b82f6;
  --color-primary-soft: rgba(59, 130, 246, 0.2);
  --color-text: rgba(255, 255, 245, 0.9);
  --color-text-secondary: rgba(235, 235, 245, 0.6);
  --color-text-tertiary: rgba(235, 235, 245, 0.38);
  --color-bg: #181818;
  --color-bg-soft: #222222;
  --color-bg-mute: #333333;
  --color-surface: #1a1a2e;
  --color-border: rgba(255, 255, 255, 0.08);
  --color-border-strong: rgba(255, 255, 255, 0.15);
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
  --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
  --shadow-lg: 0 10px 25px rgba(0,0,0,0.5);
}
```

---

## Phase 4 语义搜索 — Store 设计

```typescript
// stores/search.ts
export const useSearchStore = defineStore('search', () => {
  const query = ref('')
  const results = ref<SearchResult[]>([])
  const loading = ref(false)
  const totalPages = ref(0)
  const previewPage = ref<PagePreview | null>(null)

  async function search() {
    loading.value = true
    const resp = await fetch('/api/semantic_search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query.value, top_k: 10 })
    })
    const data = await resp.json()
    results.value = data.results
    totalPages.value = data.total_pages_indexed
    loading.value = false
  }

  async function loadPageImage(pdfPath: string, pageNum: number) {
    const resp = await fetch('/api/page_image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pdf_path: pdfPath, page_num: pageNum })
    })
    const data = await resp.json()
    previewPage.value = { imageBase64: data.image_base64, pageNum }
  }

  return { query, results, loading, totalPages, previewPage, search, loadPageImage }
})
```

---

## Flask 集成

**开发**: Vite dev server (5173) proxy `/api` → Flask (5000)，HMR 即时生效

**生产**: `npm run build` → `frontend/dist/` → Flask static folder

```python
# app.py — SPA fallback: 所有前端路由返回同一个 index.html
@app.route('/')
@app.route('/search')
@app.route('/extraction')
@app.route('/hardware')
@app.route('/experiment')
@app.route('/analysis')
def serve_frontend():
    return app.send_static_file('dist/index.html')

# API 路由 (/api/*) 在 Flask 中优先匹配，不受影响
```

---

## 实施步骤

| 步骤 | 内容 | 工时 |
|------|------|------|
| Step 1 | 初始化 Vite + Vue + TS 项目，CSS 变量 + 暗色模式 + 路由骨架 | 0.5d |
| Step 2 | 共享组件库 | 1d |
| Step 3 | 主页面 (Chat) | 1.5d |
| Step 4 | 文献提取页面 | 1d |
| **Step 5** | **Phase 4 语义搜索** | **1d** |
| Step 6 | 硬件控制页面 | 1d |
| Step 7 | 实验设计页面 | 1.5d |
| Step 8 | 数据分析页面 | 1d |
| Step 9 | Flask 集成 | 0.5d |
| Step 10 | 测试 + 打磨 | 1d |

**总计**: ~10 天
