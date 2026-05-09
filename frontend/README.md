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
│   │   ├── chat.ts              # /api/chat
│   │   ├── experiment.ts        # /api/experiment_*, /api/compile_*
│   │   ├── extraction.ts        # /api/upload, /api/extraction_*
│   │   ├── hardware.ts          # /api/hardware_*
│   │   ├── search.ts            # /api/semantic_search, /api/page_image
│   │   ├── analysis.ts          # /api/list_algorithms, /api/run_analysis
│   │   └── history.ts           # /api/history/*
│   ├── stores/                  # Pinia 状态管理
│   │   ├── chat.ts              # 对话消息 + SSE streaming
│   │   ├── experiment.ts        # 实验设计（步骤 CRUD、嵌套计算、折叠）
│   │   ├── extraction.ts        # 文献提取流程
│   │   ├── hardware.ts          # 硬件工具列表 + 单步执行
│   │   ├── search.ts            # Phase 3 语义搜索
│   │   ├── analysis.ts          # 数据分析
│   │   ├── layout.ts            # 全局 UI 状态（侧栏、任务进度）
│   │   └── theme.ts             # 主题切换
│   ├── composables/
│   │   └── useSSE.ts            # SSE (Server-Sent Events) composable
│   ├── components/
│   │   ├── common/              # 通用组件（Badge, LoadingSpinner, StatusDot, …）
│   │   ├── layout/              # 布局组件（Sidebar, TopBar, NavPanel, TaskPanel）
│   │   ├── chat/                # 对话组件（MessageBubble, ChatContainer, InputBar）
│   │   ├── experiment/          # 实验设计组件
│   │   │   ├── ElementPanel.vue # 左侧元素面板（工具/算法/辅助）
│   │   │   ├── StepCanvas.vue   # 步骤画布（gutter + 嵌套 + 折叠）
│   │   │   ├── StepCard.vue     # 步骤卡片
│   │   │   ├── StepEditor.vue   # 步骤参数编辑器
│   │   │   └── CodeArea.vue     # 底部代码区（JSON/Python 切换）
│   │   ├── search/              # 语义搜索组件
│   │   ├── modals/              # 弹窗组件
│   │   └── cards/               # 卡片组件
│   ├── pages/                   # 页面级组件（路由目标）
│   │   ├── ChatPage.vue
│   │   ├── ExperimentPage.vue
│   │   ├── ExtractionPage.vue
│   │   ├── HardwarePage.vue
│   │   ├── AnalysisPage.vue
│   │   └── SearchPage.vue
│   └── types/                   # TypeScript 类型定义
│       ├── experiment.ts
│       ├── chat.ts
│       ├── hardware.ts
│       ├── extraction.ts
│       ├── search.ts
│       └── api.ts
├── index.html                   # Vite 入口 HTML
├── package.json
├── vite.config.ts
├── tsconfig.json
└── tsconfig.app.json
```

## 路由

| 路径 | 页面 | 说明 |
|------|------|------|
| `/v2` / `/v2/chat` | ChatPage | AI 对话 |
| `/v2/experiment` | ExperimentPage | 实验设计 |
| `/v2/extraction` | ExtractionPage | 文献提取 |
| `/v2/hardware` | HardwarePage | 硬件控制 |
| `/v2/analysis` | AnalysisPage | 数据分析 |
| `/v2/search` | SearchPage | 语义搜索 |

所有 `/v2/*` SPA 路由由 Flask 返回同一个 `dist/index.html`（前端路由接管）。

## 开发

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器（热更新，API 代理到 Flask :5000）
npm run dev          # http://localhost:5173/v2

# 类型检查
npx vue-tsc -b

# 生产构建（base=/v2-static/，产物输出到 dist/）
npm run build:flask
```

## 构建产物

构建后 `dist/` 目录通过 Flask `static_folder` 挂载到 `/v2-static/` URL 路径：

```
dist/
├── index.html          # Vue SPA 入口（含 <div id="app"> + /v2-static/assets/ 引用）
└── assets/
    ├── index-*.js      # 主 bundle
    ├── *.js            # 按路由拆分（≥6 个 page chunk）
    └── *.css           # CSS（≥2 个文件）
```

## 测试

```bash
# 构建 + 集成测试
cd frontend && npm run build:flask
cd .. && python platform_init/test/frontend/test_frontend.py
```

验证项：
1. 旧版 UI `/` 可访问
2. 新版 `/v2` 返回 Vue SPA
3. 所有 `/v2/*` SPA 路由返回 index.html
4. API 路由不受影响
5. 静态资源可访问
6. Phase 4 语义搜索 API 正常
7. 构建产物完整性（main bundle + page chunks + CSS）

---

## 实验设计面板：步骤画布优化

### 背景

原先步骤画布是一个扁平列表，所有步骤（工具/算法/辅助）以相同卡片样式排列，没有块级结构可视化。

### 当前设计（v2-changeUI 分支）

#### 整体布局

```
┌──────────┬──┬──────────────────────────────────────────┐
│ Element  │# │  Step Content (with indentation)          │
│ Panel    │  │                                          │
│ (180px)  │1 │  spin_coating          [↑][↓][✏][🗑]    │
│          │2 │▼ LOOP (3次)            [↑][↓][✏][🗑]    │
│          │3 ││ set_temperature       [↑][↓][✏][🗑]    │
│          │4 ││ collect_spectrum      [↑][↓][✏][🗑]    │
│          │5 │▲ END                   [↑][↓][✏][🗑]    │
│          │7 │▼ CONDITION (temp>100)  [↑][↓][✏][🗑]    │
│          │⚠ │  ⚠ CONDITION 缺少对应的 END               │
└──────────┴──┴──────────────────────────────────────────┘
```

#### Gutter（序号列）

- **宽度**：42px，右侧 border 分隔
- **序号**：右对齐、等宽字体（Consolas/Monaco）、11px 灰色
- **块标记**：▼（可点击展开/折叠）、▲（END 结束标记）
- **错误态**：最后一步序号变红（`#ef4444`）加粗体

#### 嵌套缩进与引导线

- **缩进单位**：`INDENT = 20px`，通过 `paddingLeft` 逐级叠加
- **引导线**：绝对定位的竖线（`border-left`），标识块的作用域范围
- **块类型卡片**：
  - 块起始（LOOP/GROUP/CONDITION）：左侧 3px 橙色边框 + 淡黄背景
  - 块结束（END）：左侧 3px 灰色边框 + 降低透明度

#### 折叠展开

- **点击 ▼** → 收起块内所有步骤，▼ 变为 ▶
- **点击 ▶** → 展开恢复，▶ 变回 ▼
- **实现**：`hiddenStepIndices` computed 计算被折叠隐藏的 index 集合，通过深度追踪匹配 END

#### 错误检测

- **缺少 END**：遍历结束后栈非空 → 最后一步序号变红 → 底部黄色警告条
- **孤立 END**：END 前面对应不上任何块起始 → 对应行高亮 → 底部警告

### 实现文件

| 文件 | 变更 |
|------|------|
| `src/stores/experiment.ts` | 新增 `nestingInfo`、`blockErrors` computed；`collapsedBlocks`、`toggleCollapse`、`hiddenStepIndices` |
| `src/components/experiment/StepCanvas.vue` | 重写：gutter 列 + 缩进 + 引导线 + 折叠按钮 + 警告条 |
| `src/components/experiment/StepCard.vue` | 简化：移除序号圆标/拖拽手柄，新增 block 类型样式 |

### 核心算法

**嵌套层级计算**（`nestingInfo` computed）：

遍历 `steps[]` 扁平数组，用栈追踪未闭合的块：
- LOOP/GROUP/CONDITION → 记录当前层级 → 压栈（level++）
- END → 弹栈（level--）→ 记录新层级
- 普通步骤 → 记录当前层级

**折叠范围计算**（`hiddenStepIndices` computed）：

对每个折叠的块起始索引，向前扫描，用深度计数器找到匹配的 END（depth 回到 0），将 startIdx+1 到 END 之间所有 index 加入隐藏集合。支持嵌套块内的 END 匹配。

### 关键常量

| 常量 | 值 | 位置 |
|------|----|------|
| `INDENT` | 20px | StepCanvas.vue |
| `gutter width` | 42px | StepCanvas.vue CSS |
| `BLOCK_OPENERS` | `['LOOP', 'GROUP', 'CONDITION']` | experiment.ts store |
