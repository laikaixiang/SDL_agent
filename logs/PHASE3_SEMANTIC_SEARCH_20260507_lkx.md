# Phase 3 语义搜索（后端）完成日志

**日期**: 2026-05-07
**作者**: lkx
**版本**: v1.0 (Phase 3 backend complete)

---

## 1. 概述

成功实现 RAG 增强文献提取方案 **Phase 3 —— 语义搜索**（后端 API）。

### 核心目标

用户用自然语言查询搜索全文献库，后端通过 embedding 向量 + ChromaDB 搜索 + SQLite 元数据联合检索，返回匹配页面列表及其文本片段和相似度。

### API

**`POST /api/semantic_search`** — 语义搜索
```json
// Request
{"query": "钙钛矿钝化剂效率对比", "top_k": 10}

// Response
{
  "success": true,
  "query": "钙钛矿钝化剂效率对比",
  "total_pages_indexed": 150,
  "results": [
    {"page_id": "...", "pdf_path": "...", "pdf_name": "...",
     "page_num": 2, "text_snippet": "...", "similarity": 0.85}
  ]
}
```

**`POST /api/page_image`** — 获取 PDF 页面图片（base64），供 Phase 4 前端使用

## 2. 新建文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `extract/semantic_search.py` | ~130 | `SemanticSearch` 类：`search(query, top_k)` → embed → vector search → SQLite enrich → results |

## 3. 修改文件

| 文件 | 变更 |
|------|------|
| `app.py` | 启动时初始化 `embedding_service` / `vector_store` / `SemanticSearch`；注入 ExtractionEngine 避免重复创建 ChromaDB 连接；新增 2 个 API 路由 |
| `extract/extraction_engine.py` | `_init_page_filter_services()` 支持外部注入实例（`if self.embedding_service is None` 时才创建新的） |
| `core/config.py` | `SEMANTIC_SEARCH_ENABLED=True` |
| `config.json` | `SEMANTIC_SEARCH_ENABLED: true` |
| `config.example.json` | 同上 |

## 4. 架构设计

```
app.py 启动
  ├── create_embedding_service()        ← 提前初始化（不延迟）
  ├── ChromaVectorStore(persist_dir)    ← 提前初始化
  ├── SemanticSearch(embedding, vs, db) ← Phase 3
  └── ExtractionEngine(task_manager)
        └── _init_page_filter_services()
              └── 复用已注入的 embedding/vector_store ← 不重复创建
```

## 5. 关键技术决策

### 5.1 服务实例共享

embedding_service 和 vector_store 在 app.py 启动时初始化一次，同时注入到 ExtractionEngine 和 SemanticSearch，避免重复创建 ChromaDB 连接。

### 5.2 搜索流程

1. `embed_text(query)` → 查询向量
2. `vector_store.search(query_embedding, top_k)` → 相似页面的 page_id + distance
3. SQLite `page_metadata.db` 批量查询 text_content、pdf_path、page_num
4. distance → similarity (1 - distance)
5. 组装返回结果

### 5.3 优雅降级

- embedding/vector_store 初始化失败 → `_semantic_search_instance = None`
- API 调用时检查 None → 返回 503 错误
- 不影响已有的提取功能

## 6. 测试结果

```
=== English: "perovskite passivation" ===
  nature_articles_s41467-019-10985-5.pdf p3 sim=0.3581
  nature_articles_s41467-019-10985-5.pdf p1 sim=0.2878
  nature_articles_s41467-019-10985-5.pdf p8 sim=0.2781

=== Chinese: "钙钛矿钝化剂" ===
  nature_articles_s41467-019-10985-5.pdf p3 sim=0.2827
  nature_articles_s41467-019-10985-5.pdf p1 sim=0.2672
  nature_articles_s41467-019-10985-5.pdf p6 sim=0.2594

Total pages indexed: 9
```

- Phase 1 测试: 10/10 pass
- Phase 2 测试: 12/12 pass
- app.py 导入正常

## 7. 暂不实施（Phase 4）

- 前端搜索栏 + 结果面板 UI
- 搜索结果卡片（文件名、页码、文本片段、相似度排序）
- "查看页面"按钮 → PDF 页面图片预览
- "从此页提取"按钮 → 触发定向提取

## 8. 文件变更总览

```
新建（2个）:
  extract/semantic_search.py
  logs/PHASE3_SEMANTIC_SEARCH_20260507_lkx.md

修改（8个）:
  app.py                        (初始化 + 2 API)
  extract/extraction_engine.py  (外部注入)
  core/config.py                (SEMANTIC_SEARCH_ENABLED=True)
  config.json                   (同上)
  config.example.json           (同上)
  CLAUDE.md                     (Phase 3 DONE)
  README.md                     (Phase 3 已完成)
  rag_extraction_enhancement_design.md (Phase 3 DONE + Phase 4 TODO)
```
