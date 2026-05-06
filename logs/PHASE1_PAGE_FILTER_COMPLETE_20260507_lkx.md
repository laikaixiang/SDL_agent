# Phase 1 页面预筛选 完成日志

**日期**: 2026-05-06 ~ 2026-05-07
**作者**: lkx
**版本**: v1.0 (Phase 1 complete)

---

## 1. 概述

成功实现 RAG 增强文献提取方案 **Phase 1 —— 基于 Embedding 向量相似度的 PDF 页面预筛选**。

### 核心目标

在 PDF 提取任务中，用 Embedding 向量判断每页与任务描述的语义相关性，跳过无关页面（致谢、参考文献、背景无关段落），减少 LLM 调用次数，节省 Token 和时间。

### 数据流变化

```
Before: PDF page → extract text → LLM call → parse JSON → CSV
After:  PDF page → extract text → embedding → cosine_similarity(task_embed, page_embed)
                            ↓                           ↓
                       irrelevant (skip)          relevant → LLM call
```

---

## 2. 新建文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `core/embedding_service.py` | ~300 | Embedding 服务抽象层：`APIEmbeddingService`（通用 OpenAI 兼容接口，支持 SiliconFlow/DeepSeek）、`JinaEmbeddingService`（多模态图文）、`LocalEmbeddingService`（TODO 占位）+ 工厂函数 |
| `core/vector_store.py` | ~280 | 向量存储抽象层：`ChromaVectorStore`（ChromaDB 持久化 + cosine 距离 + upsert 去重）+ `PgvectorVectorStore`（TODO 占位） |
| `core/page_indexer.py` | ~210 | PDF 页面预索引：`make_page_id()` / `compute_content_hash()` + `PageIndexer`（SQLite 元数据库 + 内容 hash 增量去重 + 幂等索引） |
| `core/page_filter.py` | ~160 | 页面预筛选：`PageFilter.set_task()` 缓存任务向量 + `should_process()` 逐页余弦相似度比较 |
| `platform_init/test/phase1_page_filter/test_phase1.py` | ~340 | 10 项功能测试（页面 ID 生成 / hash 去重 / 余弦相似度 / ChromaDB CRUD / 配置项 / ABC 抽象类 / 工厂函数 / ExtractionEngine 集成 / Embedding API / PageFilter 筛选） |
| `platform_init/test/phase1_page_filter/test_model_comparison.py` | ~190 | 模型对比测试：BAAI/bge-large-en-v1.5 vs Qwen/Qwen3-VL-Embedding-8B，同一 PDF 逐页输出相似度 + 统计对比 + 结论 |
| `logs/PHASE1_PAGE_FILTER_20260506_lkx.md` | ~200 | 初始实现日志（架构设计、测试数据、替代方案） |
| `logs/PHASE1_PAGE_FILTER_COMPLETE_20260507_lkx.md` | 本文件 | 完成日志 |

## 3. 修改文件

| 文件 | 变更 |
|------|------|
| `core/config.py` | 新增 17 个配置项：Embedding（7项，含 `EMBEDDING_BACKEND`/`EMBEDDING_API_KEY`/`EMBEDDING_MODEL`/`EMBEDDING_MAX_CHARS` 等）+ VectorStore（2项）+ PageFilter（3项）+ FewShot（2项 flag）+ SemanticSearch（1项 flag） |
| `core/extraction_engine.py` | 新增 `_init_page_filter_services()` 优雅降级初始化、`process_pdf_library()` 增加预索引步骤、`_process_single_pdf()` 页面循环插入 `page_filter.should_process()` 检查、新增 `task_description` 参数传递链 |
| `requirements.txt` | 添加 `chromadb` 依赖 |
| `CLAUDE.md` | RAG TODO 章节更新为 Phase 1 已完成的实现总结 |
| `README.md` | Section 十三 更新为 Phase 1 已完成的实现总结 |
| `rag_extraction_enhancement_design.md` | 状态从 "pending implementation" 更新为 "Phase 1 implemented" |

---

## 4. 架构设计

### 类层次

```
EmbeddingService (ABC)
├── APIEmbeddingService     ← EMBEDDING_BACKEND="api"   → SiliconFlow / DeepSeek / ...
├── JinaEmbeddingService    ← EMBEDDING_BACKEND="jina"  → Jina AI 多模态图文
└── LocalEmbeddingService   ← EMBEDDING_BACKEND="local" → TODO 未来本地部署

VectorStore (ABC)
├── ChromaVectorStore       ← VECTOR_STORE_BACKEND="chromadb"
└── PgvectorVectorStore     ← VECTOR_STORE_BACKEND="pgvector" (TODO)

PageIndexer  → 依赖 EmbeddingService + VectorStore + PDFProcessor + SQLite
PageFilter   → 依赖 EmbeddingService + VectorStore
```

### ExtractionEngine 集成流程

```
ExtractionEngine.process_pdf_library(task_description, fields)
    │
    ├── _init_page_filter_services()
    │   ├── create_embedding_service() → APIEmbeddingService (SiliconFlow BGE)
    │   ├── ChromaVectorStore(persist_dir="dialogue data/vector_store")
    │   ├── PageIndexer(embedding, vector_store, sqlite)  ← 预索引器
    │   └── PageFilter(embedding, vector_store, threshold=0.3)  ← 筛选器
    │   异常 → page_filter = None (优雅降级，不影响正常提取)
    │
    ├── page_indexer.index_all_pdfs()  ← 增量索引（跳过内容未变更的页面）
    │
    ├── page_filter.set_task(task_description)  ← 缓存任务向量
    │
    └── for each PDF → for each page:
            if page_filter.should_process(pdf_path, page_num):
                _process_single_page()  → LLM 调用
            else:
                skip  ← 节省 Token + 时间
```

---

## 5. 关键技术决策

### 5.1 Embedding 后端选择

经过三轮迭代，从最初的 `"jina"` 演进为 `"api"` / `"jina"` / `"local"` 三选一：

- **"api"**（当前默认）：通用 OpenAI 兼容格式，通过 `EMBEDDING_API_URL` / `EMBEDDING_MODEL` 自由配置，默认指向 SiliconFlow `BAAI/bge-large-en-v1.5`
- **"jina"**：Jina 原生多模态格式 `[{"text":..., "image":...}]`，支持图文混合输入，保留给需要页面图片 embedding 的场景

### 5.2 模型选型

对两个候选模型在同一 PDF 上做了 A/B 对比测试：

| 指标 | BAAI/bge-large-en-v1.5 | Qwen/Qwen3-VL-Embedding-8B |
|------|------------------------|---------------------------|
| 维度 | 1024 | 4096 |
| 平均相似度 | 0.7310 | 0.2483 |
| Spread (max-min) | **0.2301** | 0.1964 |
| 通过率 (t=0.3) | **100% (9/9)** | 22.2% (2/9) |
| 多模态 | 否 | 是 |

**结论**：英文科学文献场景下 `BAAI/bge-large-en-v1.5` 明显更优——区分度更大（Spread 0.23 vs 0.20），语义排序更合理（标题页得分最高，补充信息页得分最低），通过率合理（全文关于钙钛矿钝化，所有页都应处理）。

### 5.3 文本截断

科学文献文本 token 密度高（化学式、数字、特殊符号），2000 字符也可能超出 BGE 的 512 token 限制。通过 `EMBEDDING_MAX_CHARS=1000` 保守截断，API 413 错误彻底解决。

### 5.4 类型安全

- ChromaDB 以 numpy array 存储向量，`get_embedding()` 显式转为 Python `list`
- API 返回的 embedding 中可能存在 `int` 类型值（如精确的 `0`），`_call_api()` 强制 `float(x)` 转换

---

## 6. 测试覆盖

### test_phase1.py (10 项)

| # | 测试项 | 状态 |
|---|--------|------|
| 1 | make_page_id / compute_content_hash | PASS |
| 2 | 内容 hash 去重逻辑 | PASS |
| 3 | 余弦相似度计算 | PASS |
| 4 | ChromaDB CRUD + upsert + search | PASS |
| 5 | 配置项存在性与类型 | PASS |
| 6 | ABC 抽象基类强制实现 | PASS |
| 7 | 工厂函数（api/jina/local/未知） | PASS |
| 8 | ExtractionEngine 集成 + 优雅降级 | PASS |
| 9 | Embedding API 实际调用 | PASS |
| 10 | PageFilter 语义筛选 | PASS |

### test_model_comparison.py

同一 PDF + 同一任务描述，BGE-en 和 Qwen3-VL 两个模型逐页对比，输出相似度 + 统计 + 结论。

---

## 7. 部署说明

### 用户需配置

在 `core/config.py` 中（已配置，按需修改）：

```python
EMBEDDING_BACKEND    = "api"                            # 使用云端 API
EMBEDDING_API_KEY    = "sk-xxx"                        # SiliconFlow API Key
EMBEDDING_MODEL      = "BAAI/bge-large-en-v1.5"        # 推荐英文文献模型
PAGE_FILTER_ENABLED  = True                             # 开启预筛选
PAGE_FILTER_THRESHOLD = 0.3                              # 余弦相似度阈值
EMBEDDING_MAX_CHARS  = 1000                              # 文本截断长度
FEW_SHOT_ENABLED     = False                             # Phase 2（未实现）
SEMANTIC_SEARCH_ENABLED = False                          # Phase 3（未实现）
```

### 无需前端改动

Phase 1 的预筛选在 `ExtractionEngine` 内部完成，前端无感知。用户正常使用文献提取功能，后端自动在页面循环中筛选。

---

## 8. 后续规划

### Phase 2: Few-shot 增强
- `FEW_SHOT_ENABLED` / `FEW_SHOT_TOP_K` 配置已加，逻辑未实现
- 文件：`core/few_shot_retriever.py`
- 目标：LLM 调用时从历史提取结果中检索相似案例作为 prompt 示例

### Phase 3: 语义搜索
- `SEMANTIC_SEARCH_ENABLED` 配置已加，逻辑未实现
- 文件：`core/semantic_search.py`
- 附录 API 路由：`/api/semantic_search`
- 目标：用户自然语言搜索全文献库，命中后再做深度提取

### 大规模迁移
- pgvector 接口已预留（`PgvectorVectorStore`）
- 本地模型接口已预留（`LocalEmbeddingService`）
- 迁移时机：ChromaDB 向量数超过 50 万条

---

## 9. 文件变更总览

```
新建（8个）:
  core/embedding_service.py
  core/vector_store.py
  core/page_indexer.py
  core/page_filter.py
  platform_init/test/phase1_page_filter/test_phase1.py
  platform_init/test/phase1_page_filter/test_model_comparison.py
  logs/PHASE1_PAGE_FILTER_20260506_lkx.md
  logs/PHASE1_PAGE_FILTER_COMPLETE_20260507_lkx.md

修改（6个）:
  core/config.py            (+35 行)
  core/extraction_engine.py (+60 行)
  requirements.txt          (+1 行)
  CLAUDE.md                 (RAG section 重写)
  README.md                 (Section 十三 重写)
  rag_extraction_enhancement_design.md  (status update)
```
