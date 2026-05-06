# Phase 2 Few-Shot 示例检索 完成日志

**日期**: 2026-05-07
**作者**: lkx
**版本**: v1.0 (Phase 2 complete)

---

## 1. 概述

成功实现 RAG 增强文献提取方案 **Phase 2 —— Few-Shot 示例检索**。

### 核心目标

在 LLM 提取前，从历史提取结果中检索与当前任务最相似的提取记录，作为 Few-Shot 示例注入 system prompt，提高 LLM 提取的准确性和一致性。

### 数据流

```
首次提取:
  PDF page → LLM call → parse JSON → save to extraction_history.db (SQLite)

后续提取:
  task_description → embed_text → vector search → find similar pages
                  → query SQLite for extraction records from those pages
                  → inject top-3 examples into system prompt
                  → LLM call (now with few-shot examples)
```

---

## 2. 新建文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `core/few_shot_retriever.py` | ~170 | Few-Shot 检索器：`FewShotRetriever` 类，SQLite 历史管理 + 向量搜索联合检索 |

## 3. 修改文件

| 文件 | 变更 |
|------|------|
| `core/extraction_engine.py` | +70 行：新增 `few_shot_retriever` 属性、`_inject_few_shot_examples()`、`_save_to_extraction_history()`、`task_description` 参数传递链（`_process_single_pdf` → `_process_single_page` → `_process_with_vision` / `_process_with_text`） |
| `core/config.py` | `FEW_SHOT_ENABLED: True`（从 False 改为 True） |
| `rag_extraction_enhancement_design.md` | 状态更新为 Phase 1+2 done，Phase 2 文件表标记 DONE |
| `CLAUDE.md` | RAG 章节更新：Phase 2 标记 DONE，新增 Phase 2 实现小结 |
| `README.md` | Section 十三更新：Phase 2 标记 ✅ 已完成，新增文件清单和工作流程 |

---

## 4. 架构设计

### FewShotRetriever

```
FewShotRetriever
├── _init_db()              → 创建 extraction_history 表 + 索引
├── save_extraction()        → 存储 LLM 提取结果（JSON 格式）
├── retrieve_examples()      → 向量搜索 + SQLite 联合检索
│   1. embed_text(task_description)
│   2. vector_store.search(query_embedding, top_k*3)
│   3. 对每个相似页面查 SQLite extraction_history
│   4. 去重 + 清理内部字段 → 返回 top_k 示例
└── count()                  → 历史记录总数
```

### SQLite 表结构

```sql
CREATE TABLE IF NOT EXISTS extraction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id TEXT NOT NULL,
    source_doc TEXT,
    task_description TEXT NOT NULL,
    extracted_json TEXT NOT NULL,   -- 完整提取数据 JSON
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_history_page_id ON extraction_history(page_id);
CREATE INDEX IF NOT EXISTS idx_history_task ON extraction_history(task_description);
```

### ExtractionEngine 集成

```
ExtractionEngine
├── _init_page_filter_services()
│   ├── self.embedding_service = create_embedding_service()  ← 提升为实例属性
│   ├── self.vector_store = ChromaVectorStore(...)           ← 提升为实例属性
│   ├── self.page_indexer = PageIndexer(...)                  ← Phase 1
│   ├── self.page_filter = PageFilter(...)                    ← Phase 1
│   └── self.few_shot_retriever = FewShotRetriever(...)       ← Phase 2 NEW
│
├── _process_with_vision() / _process_with_text()
│   ├── sys_prompt = self._inject_few_shot_examples(sys_prompt, task_description, fields)
│   │   ├── self.few_shot_retriever.retrieve_examples(task_description, fields, top_k=3)
│   │   └── 示例: "📋 参考历史提取示例：\n示例1: {...}\n示例2: {...}"
│   └── self._save_to_extraction_history(pdf_path, page_num, result["data"], ...)
│       └── self.few_shot_retriever.save_extraction(page_id, item_dict, task_description, source_doc)
```

---

## 5. 关键技术决策

### 5.1 示例注入位置

Few-Shot 示例放在 system prompt **最前面**（在原始系统提示词之前），格式为：

```
📋 参考历史提取示例（从相似页面中提取的数据，供你参考格式和内容）：
示例 1: {"passivation_agent": "PEAI", "concentration": "5 mg/mL", ...}
示例 2: {"passivation_agent": "4F-PEAI", ...}
请参考以上示例的提取风格和详细程度来处理当前页面。

[原始 system prompt...]
```

放在最前面确保 LLM 首先看到示例格式，建立输出预期。

### 5.2 示例检索策略

- 向量搜索 `top_k * 3` 个相似页面（留缓冲，因为部分页面无提取记录）
- 每个页面只取最新一条提取记录（`ORDER BY created_at DESC LIMIT 1`）
- 页面级去重（`seen_pages` set）
- 返回前清除内部字段（`_source_doc` 等以 `_` 开头的 key）

### 5.3 存储粒度

每条 LLM 返回的 data 数组中的每个 item 作为一条独立记录存储。这样：
- 同一页面可存储多条提取结果（不同数据行）
- 检索时取最新的记录作为该页面的代表示例
- 完整 JSON 存储，保留所有字段信息

### 5.4 优雅降级

- `FEW_SHOT_ENABLED=False` → 不初始化 FewShotRetriever
- 初始化失败 → `self.few_shot_retriever = None`
- 检索时检查 `if not self.few_shot_retriever: return sys_prompt`（原样返回）
- 保存时检查 `if not self.few_shot_retriever: return`（静默跳过）
- 保存抛异常 → `try/except pass`（不影响提取流程）

### 5.5 与 Phase 1 的关系

Phase 1 和 Phase 2 共享 `embedding_service` 和 `vector_store` 实例。两者都从 `_init_page_filter_services()` 初始化，共享生命周期。Phase 2 依赖 Phase 1 的基础设施但可独立禁用（`FEW_SHOT_ENABLED=False` 时 Phase 1 仍正常工作）。

---

## 6. 使用说明

### 首次使用

无需额外配置。`FEW_SHOT_ENABLED=True`（默认），首次提取时：
1. 没有历史记录，`retrieve_examples()` 返回空列表
2. System prompt 不变，正常提取
3. 提取完成后自动保存到 `extraction_history.db`
4. 第二次提取相同或类似任务时，自动检索并注入示例

### 查看历史记录数

```python
from core.few_shot_retriever import FewShotRetriever
# ... 初始化后
print(f"历史提取记录: {retriever.count()} 条")
```

### 禁用 Few-Shot

```python
# core/config.py
FEW_SHOT_ENABLED = False
```

---

## 7. 后续规划

### Phase 3: 语义搜索
- `SEMANTIC_SEARCH_ENABLED` 配置已加，逻辑未实现
- 文件：`core/semantic_search.py`
- API 路由：`/api/semantic_search`
- 目标：用户自然语言搜索全文献库，命中后再做深度提取

---

## 8. 文件变更总览

```
新建（1个）:
  core/few_shot_retriever.py

修改（5个）:
  core/extraction_engine.py  (+70 行: FewShotRetriever 集成)
  core/config.py             (FEW_SHOT_ENABLED: False→True)
  CLAUDE.md                  (RAG section 更新 Phase 2 DONE)
  README.md                  (Section 十三 更新 Phase 2 完成)
  rag_extraction_enhancement_design.md  (status update)

新建（1个）:
  logs/PHASE2_FEW_SHOT_COMPLETE_20260507_lkx.md  (本文件)
```
