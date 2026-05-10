# Markdown 预览功能 完成日志

**日期**: 2026-05-10
**作者**: lkx
**版本**: v1.0

---

## 1. 概述

为 Vue SPA 前端（`/v2`）的 AI 对话消息实现 markdown 渲染，支持标题、表格、代码块、LaTeX 公式等。

## 2. 新建文件

| 文件 | 说明 |
|------|------|
| `frontend/src/utils/markdown.ts` | `markdownToHtml()` — 渲染流水线：提取 LaTeX → marked → 回填 katex |

## 3. 修改文件

| 文件 | 变更 |
|------|------|
| `frontend/package.json` | 新增依赖 `marked` `katex` `@types/katex` |
| `frontend/src/main.ts` | 新增 `import 'katex/dist/katex.min.css'` |
| `frontend/src/components/chat/MessageBubble.vue` | AI 消息用 `v-html` 渲染 markdown；新增 scoped `.markdown-body` 样式（h1-h3/p/code/pre/table/blockquote 等） |

## 4. 渲染流水线

```
原始文本
  → 提取 $$...$$ 替换为占位符 \x00DM{n}\x00
  → 提取 $...$   替换为占位符 \x00IM{n}\x00
  → marked.parse() 渲染标准 markdown（GFM 表格/代码/列表）
  → 回填 display math: katex.renderToString(formula, {displayMode: true})
  → 回填 inline math:  katex.renderToString(formula, {displayMode: false})
```

- 用户消息保持 `{{ content }}` 纯文本
- AI 消息使用 `v-html="markdownToHtml(content)"`
- 流式过程中部分 markdown 即时渲染，流结束完整渲染

## 5. 踩坑

- **工作目录错误**: `npm install katex` 在项目根目录执行，导致根目录生成 `package.json` + `node_modules/`，frontend 内缺少依赖
- **`/Git` 前缀**: 根目录存在 `package.json` 时执行 `npx vite build --base=/v2-static/`，Git Bash 环境下路径被错误拼接为 `/Git/v2-static/...`
- **修复**: 删除根目录 `package.json`/`node_modules/`，从 `frontend/` 目录用 `npm run build:flask` 构建，产物路径恢复为 `/v2-static/assets/...`
