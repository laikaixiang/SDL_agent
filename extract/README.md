# extract/ — PDF 文献提取子系统

> 核心代码: 1 个提取引擎 (`extraction_engine.py`) + 9 个支撑模块
> 设计文档: `extract/LITERATURE_INDEXER_DESIGN.md` / `extract/PLAN_2026-05-10-pdf-metadata-extraction.md`

---

## 1. 总体架构

```
                          用户输入 "帮我搜寻：钙钛矿钝化剂参数"
                                       │
                                       ▼
              ┌──────────────────────────────────────────────┐
              │  app.py /api/chat  前缀路由  "帮我搜寻："    │
              │  → 第一次: 返回 field_confirm 让用户确认字段 │
              │  → 确认:  POST /api/extract  启动任务          │
              └──────────────────┬───────────────────────────┘
                                 │
                                 ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │                    ExtractionEngine.process_pdf_library           │
   │  ┌─────────────────────────────────────────────────────────────┐  │
   │  │ 1. 字段推断  FieldInference.infer_fields(task)              │  │
   │  │    →  ["钝化剂名称", "分子式", "效率", "器件结构"]        │  │
   │  └─────────────────────────────────────────────────────────────┘  │
   │  ┌─────────────────────────────────────────────────────────────┐  │
   │  │ 2. 动态模型  create_dynamic_model(fields)                   │  │
   │  │    →  Pydantic 模型, 供 LLM 结构化输出使用                 │  │
   │  └─────────────────────────────────────────────────────────────┘  │
   │  ┌─────────────────────────────────────────────────────────────┐  │
   │  │ 3. Phase 1 页面预筛选  (PageIndexer + PageFilter)         │  │
   │  │    → 跳过无关页面, 减少 LLM 调用                            │  │
   │  └─────────────────────────────────────────────────────────────┘  │
   │  ┌─────────────────────────────────────────────────────────────┐  │
   │  │ 4. 逐页处理                                                  │  │
   │  │    ├─ Vision 模式: PDF → JPEG → VL 模型 → JSON             │  │
   │  │    └─ Text 模式:  PDF → Markdown → LLM → JSON             │  │
   │  └─────────────────────────────────────────────────────────────┘  │
   │  ┌─────────────────────────────────────────────────────────────┐  │
   │  │ 5. Phase 2 Few-Shot 注入  (FewShotRetriever)               │  │
   │  │    → 历史提取结果作为示例注入 system prompt                  │  │
   │  └─────────────────────────────────────────────────────────────┘  │
   │  ┌─────────────────────────────────────────────────────────────┐  │
   │  │ 6. 质量检查  QualityChecker (稀疏 + 重复)                    │  │
   │  │ 7. 去重      Dedup (按 fields[0] 合并)                       │  │
   │  │ 8. 保存 CSV  → extract/  + temporal/                          │  │
   │  └─────────────────────────────────────────────────────────────┘  │
   └───────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
   dialogue data/<session>/temporal/extraction.csv   ←  全局共享, 默认文件
   dialogue data/<session>/extract/<prefix>_YYYYMMDD-HHMMSS.csv  ←  时间戳归档
```

---

## 2. 模块职责表

| 文件 | 职责 | 调用方 |
|------|------|--------|
| `extraction_engine.py` | 主入口: 字段推断、模型构造、逐页调度、结果保存 | `app.py` `/api/extract` |
| `pdf_processor.py` | PDF 元信息、文本提取、页→图片、混合模式切换 | `extraction_engine` |
| `field_inference.py` | 根据任务描述推断要提取的字段(LLM 调用) | `extraction_engine` |
| `embedding_service.py` | 多模态 embedding 抽象层 (api / jina / local) | indexer / filter / search |
| `vector_store.py` | 向量存储抽象层 (ChromaDB / 预留 pgvector) | indexer / filter / search |
| `page_indexer.py` | 一次性预索引所有 PDF 页面 (text + image → embedding) | Phase 1 启动时 |
| `page_filter.py` | 运行时逐页相关性判断 (任务描述 vs 页面向量) | `_process_single_pdf` |
| `few_shot_retriever.py` | 检索历史提取结果作为 Few-Shot 示例 | LLM 调用前 |
| `semantic_search.py` | 用户搜索文献 (Phase 3 backend) | `/api/semantic_search` |
| `literature_indexer.py` | SQLite 文献注册表 + 批量元数据提取 | `/api/literature/index` |
| `algorithm_guide.py` | 4 步引导式算法生成 (Q&A 流程) | `/api/algorithm_gen/guide` |
| `dedup.py` | 按 `fields[0]` 合并重复行 (longest / first_non_empty) | 保存 CSV 前 |
| `quality_checker.py` | 稀疏记录 + 重复记录检测 (纯规则, 不调 LLM) | 保存 CSV 前 |

---

## 3. 数据流详细分解

### 3.1 字段推断 (FieldInference)

```python
# core/field_inference.py
def infer_fields(task_description: str, history=None) -> Tuple[bool, List[str] | str]:
    """根据任务描述, 让 LLM 推荐要提取的字段"""
    # Prompt: prompts/zh/field_inference/_infer_fields.yaml
    # 输出:  ["钝化剂名称", "分子式", "效率", "器件结构"]
```

- 用户可手动调整
- 确认后, 这组字段决定:
  1. Pydantic 动态模型 schema
  2. CSV 列名
  3. 去重的 key (`fields[0]`)

### 3.2 动态 Pydantic 模型

```python
DynamicRecord = field_inference.create_dynamic_model(fields)
# 动态创建:
# class DynamicRecord(BaseModel):
#     钝化剂名称: str
#     分子式: str
#     效率: float
#     器件结构: str

# LLM 用这个 schema 作为 JSON 输出约束
schema_str = json.dumps(LocalExtractionResponse.model_json_schema())
# 传给 LLM 的 system prompt 里包含这个 schema
```

### 3.3 Phase 1: 页面预筛选

**目标:** 跳过无关页面(如参考文献、背景介绍), 减少 80% 以上的 LLM 调用

**组件:**

- `PageIndexer.index_all_pdfs()` — 一次性遍历所有 PDF, 计算每页 embedding
  - 输入: `dialogue data/PDF_TARGET/*.pdf`
  - 输出: ChromaDB (`<CHROMADB_PERSIST_DIR>`) + SQLite (`page_metadata.db`)
  - 增量: 用 `content_hash` 跳过未变更页面

- `PageFilter.set_task()` — 预嵌入任务描述
  ```python
  self.task_filter.set_task("提取 FAPbI3 钝化剂参数")
  # 内部: self._task_embedding = embed_text(task_description)
  ```

- `PageFilter.should_process(pdf, page_num)` — 余弦相似度
  ```python
  score = cosine(self._task_embedding, page_embedding)
  return score >= threshold  # 默认 0.3
  ```

**效果:** 一次 PDF 文献库 1000 页 → 实际处理 50-200 页(节省 LLM 调用)

### 3.4 模式切换: Vision vs Text

`pdf_processor.get_extraction_mode()` 决定用哪种方式:

| 模式 | 适用 | 流程 | 成本 |
|------|------|------|------|
| `vision` | 含图表/分子式/复杂版式 | 页→JPEG→VL 模型 | 高 (~$0.01/页) |
| `text` | 纯文字 + 简单表格 | 页→Markdown→LLM | 低 (~$0.001/页) |
| `auto` | 默认 (按页判断) | 内容 < 50 字符用 text | 视情况 |

详见 `pdf_processor.py` 文档。

### 3.5 Phase 2: Few-Shot 注入

**目的:** 用历史成功提取结果作为示例, 提高新提取的准确性

**数据流:**

```
LLM 提取成功
    ↓
_save_to_extraction_history()
    ├─ SQLite: extraction_history.db (page_id, extracted JSON, task_description)
    └─ ChromaDB: 同样的向量 (用于按语义检索)

下次提取前:
    ↓
few_shot_retriever.retrieve_examples(task_description, fields, top_k=3)
    ├─ embed_text(task_description) → 向量
    ├─ vector_store.search(向量) → top_k*3 相似页面
    ├─ SQLite 查这些 page_id 的历史提取结果
    └─ 返回去重后的 top_k 个示例
    
_inject_few_shot_examples(sys_prompt, ...)
    └─ 把示例拼到 system prompt 里:
       "以下是历史成功提取的示例:
        示例 1: {...}
        示例 2: {...}
        ..."
```

**关键:** 提取历史是持续积累的资产, 用得越多越准

### 3.6 质量检查 (QualityChecker)

在保存 CSV 前, 删除两类问题数据:

| 检查 | 规则 | 配置 |
|------|------|------|
| 稀疏检测 | 字段填充率 < 阈值 (默认 0.3) | `QUALITY_SPARSE_THRESHOLD` |
| 重复检测 | A 行完全包含 B 行, 或字段完全相同 | 内置规则 |

**纯 Python 实现**, 不调 LLM, 毫秒级。

### 3.7 去重 (Dedup)

按 `fields[0]` (通常是"实体名称")分组:

```python
# dedup.py
deduplicate_extraction_results(
    data, fields,
    normalize="strip",         # 去除首尾空格
    merge_strategy="longest",  # 同字段取最长的
    add_metadata=True,         # 添加 _occurrence_count, _source_docs
)
# 同一钝化剂出现在 3 页 → 合并为 1 行, 字段取最完整的那条
```

**示例:**
```
原始:
  | 钝化剂 | 效率 | 分子式           |
  | PEAI   | 22%  |                  |  ← 缺分子式
  | PEAI   | 22%  | C6H5C2H4NH3I     |
  | PEAI   | 22.1%| C6H5C2H4NH3I     |

去重后:
  | 钝化剂 | 效率 | 分子式           | _occurrence_count |
  | PEAI   | 22.1%| C6H5C2H4NH3I     | 3                 |
```

### 3.8 保存

两条输出:
1. **全局临时** — `dialogue data/<session>/temporal/extraction.csv` (工作文件, 每次新任务覆盖)
2. **时间戳归档** — `dialogue data/<session>/extract/<prefix>_<YYYYMMDD-HHMMSS>.csv` (永久记录)

---

## 4. RAG 设计要点

### 4.1 为什么用 RAG (Phase 1-3)

直接 LLM 处理所有页面:
- ❌ 1000 页 PDF = 1000 次 LLM 调用 = $$$ + 慢
- ❌ 大量无关页面 (参考文献、致谢、目录)
- ❌ 上下文被噪声淹没

引入 RAG 后:
- ✅ 一次性预索引 → 后续提取前先过滤
- ✅ 相似度阈值可调, 召回率 / 成本可权衡
- ✅ Few-Shot 示例自动检索, 越用越准

### 4.2 相似度计算

- **算法:** 余弦相似度 (cosine similarity)
- **范围:** `[-1, 1]`, 但 embedding 通常非负所以实际 `[0, 1]`
- **阈值:** 默认 `0.3` (保守, 宁可多处理不可漏数据)
  - `0.3` — 几乎不漏数据
  - `0.5` — 中等, 跳过明显不相关
  - `0.7` — 激进, 只处理高度相关 (可能漏)

### 4.3 Embedding 后端选择

| 后端 | 适用 | 速度 | 成本 |
|------|------|------|------|
| SiliconFlow `BAAI/bge-large-zh-v1.5` | 中文文献 | 快 | 低 (~$0.0001/页) |
| Jina `jina-clip-v2` | 多模态 (文本+图片) | 中 | 中 |
| 本地模型 (TODO) | 离线场景 | 取决于硬件 | 0 |

切换只需 `config.EMBEDDING_BACKEND = "..."`, 业务代码零改动 (抽象类 `EmbeddingService`)。

### 4.4 向量存储后端

- **当前:** ChromaDB (`<CHROMADB_PERSIST_DIR>`) — 适合 < 50 万条向量
- **预留:** pgvector — 大规模部署迁移路径

业务代码只依赖 `VectorStore` 抽象接口, 切换后端零改动。

---

## 5. 数据持久化

### 5.1 文件系统

```
dialogue data/
├── PDF_TARGET/                                ← 用户文献库
│   ├── *.pdf
│   └── literature_registry.db                 ← 标题→文件名映射 (UNIQUE)
└── <session>/
    ├── temporal/
    │   └── extraction.csv                     ← 全局共享工作文件
    └── extract/
        └── <prefix>_<YYYYMMDD-HHMMSS>.csv    ← 时间戳归档
```

### 5.2 索引数据库

```
<CHROMADB_PERSIST_DIR>/
├── chroma.sqlite3                             ← 向量数据
├── page_metadata.db                           ← Phase 1 页面元数据
│   ├── page_id  pdf_path  page_num  text_content
│   └── content_hash (增量检测)
├── extraction_history.db                      ← Phase 2 历史提取
│   ├── page_id  extracted_json  task_description
│   └── timestamp
└── literature_registry.db                     ← Phase 3+ 文献注册表
    ├── title (UNIQUE)  current_filename  doi
    ├── authors  abstract_summary  innovation_points
    ├── key_image_* (关键图坐标)  extraction_status
    └── created_at  updated_at
```

---

## 6. 关键配置项 (`core/config.py`)

| 配置 | 默认 | 说明 |
|------|------|------|
| `EMBEDDING_BACKEND` | `"api"` | api / jina / local |
| `EMBEDDING_MODEL` | `"BAAI/bge-large-en-v1.5"` | embedding 模型名 |
| `PAGE_FILTER_ENABLED` | `True` | 是否启用 Phase 1 预筛选 |
| `PAGE_FILTER_THRESHOLD` | `0.3` | 余弦相似度阈值 (越小越保守) |
| `FEW_SHOT_ENABLED` | `True` | 是否启用 Phase 2 Few-Shot |
| `FEW_SHOT_TOP_K` | `3` | 注入 prompt 的示例数 |
| `SEMANTIC_SEARCH_ENABLED` | `True` | 是否启用 Phase 3 语义搜索 |
| `DEDUP_ENABLED` | `True` | 是否在保存前去重 |
| `DEDUP_NORMALIZE` | `"strip"` | 实体名规范化: strip / lower / strict |
| `DEDUP_MERGE_STRATEGY` | `"longest"` | 同字段合并: longest / first_non_empty |
| `DEDUP_ADD_METADATA` | `True` | 是否添加 _occurrence_count 等元数据 |
| `QUALITY_CHECK_ENABLED` | `True` | 是否运行 QualityChecker |
| `QUALITY_SPARSE_THRESHOLD` | `0.3` | 稀疏阈值 (填充率 < 此值视为稀疏) |
| `CHROMADB_PERSIST_DIR` | `"dialogue data/vector_store"` | 向量库存储目录 |
| `LITERATURE_REGISTRY_DB_PATH` | `"dialogue data/PDF_TARGET/literature_registry.db"` | 文献注册表路径 |

---

## 7. 调试 / 测试

### 7.1 测试脚本

```bash
# Phase 1 页面预筛选
python platform_init/test/phase1_page_filter/test_phase1.py

# Phase 2 Few-Shot
python platform_init/test/phase2_few_shot/test_phase2.py

# Phase 3 语义搜索 (Mock + 真实 API)
python platform_init/test/phase3_semantic_search/test_phase3.py

# 去重
python platform_init/test/dedup/test_dedup.py
```

### 7.2 端到端流程

```bash
# 启动 Flask
python app.py

# 浏览器测试 (V2):
#   1. 切换到「文献提取」模式
#   2. 输入 "提取钙钛矿钝化剂参数"
#   3. 确认字段
#   4. 等待: 索引 → 逐页筛选 → 提取 → 质量检查 → 去重 → CSV
#   5. 检查 dialogue data/<session>/temporal/extraction.csv
```

### 7.3 常见问题

| 症状 | 排查 |
|------|------|
| 提取超时 | 减小 `PAGE_FILTER_TOP_K`, 增大阈值 |
| 提取数据太少 | 降低 `PAGE_FILTER_THRESHOLD` (0.2 → 0.1) |
| 字段不合理 | 检查 `prompts/zh/field_inference/_infer_fields.yaml` |
| 重复行太多 | 确认 `DEDUP_ENABLED=True` 且 `fields[0]` 是唯一标识 |
| embedding 报错 | 检查 `EMBEDDING_API_KEY` / 网络 |
| ChromaDB 损坏 | 删 `dialogue data/vector_store/` 重索引 |

---

## 8. 设计文档索引

- `extract/LITERATURE_INDEXER_DESIGN.md` — LiteratureIndexer 详细设计
- `extract/PLAN_2026-05-10-pdf-metadata-extraction.md` — 元数据提取规划
- `platform_init/test/phase*/DESIGN.md` — 各阶段测试设计
- `core/agent_tools.py:_resolve_pdf_path` — 文件名/标题/DOI 解析 7 级 fallback
- 项目根: `RAG_EXTRACTION_ENHANCEMENT_DESIGN.md` — 整体 RAG 设计

---

## 9. 关键类/函数速查

```python
# 主入口
from extract.extraction_engine import ExtractionEngine
engine = ExtractionEngine(task_manager, session_path=..., temporal_dir=...)
engine.process_pdf_library(task_id, task_description, fields)

# 字段推断
from core.field_inference import FieldInference
ok, fields = FieldInference().infer_fields(task_description)

# Phase 1
from extract.page_indexer import PageIndexer
from extract.page_filter import PageFilter
indexer = PageIndexer(embedding_service, vector_store, sqlite_path, pdf_processor)
indexer.index_all_pdfs()  # 返回 (indexed_count, skipped_count)

filter = PageFilter(embedding_service, vector_store, threshold=0.3)
filter.set_task(task_description)
if filter.should_process(pdf_path, page_num): ...

# Phase 2
from extract.few_shot_retriever import FewShotRetriever
fs = FewShotRetriever(embedding_service, vector_store, sqlite_path)
examples = fs.retrieve_examples(task_description, fields, top_k=3)

# Phase 3 语义搜索
from extract.semantic_search import SemanticSearch
ss = SemanticSearch(embedding_service, vector_store, sqlite_path)
results = ss.search(query, top_k=10)
# → [{pdf_name, page_num, similarity, text_snippet}, ...]

# 文献注册表
from extract.literature_indexer import LiteratureIndexer
indexer = LiteratureIndexer(config)
indexer.index_library(callback=...)  # 批量索引
entry = indexer.get_entry_by_title(title)
entry = indexer.search_by_doi(doi)

# 质量检查 + 去重
from extract.quality_checker import QualityChecker
from extract.dedup import deduplicate_extraction_results
qc = QualityChecker.run_all_checks(data, fields, sparse_threshold=0.3)
data = deduplicate_extraction_results(data, fields, normalize="strip", merge_strategy="longest")
```

---

## 10. 演进路线

| 阶段 | 状态 | 内容 |
|------|------|------|
| Phase 0 | ✓ | 纯 LLM 处理所有页面 (基线) |
| Phase 1 | ✓ | 页面预筛选 (PageFilter, 减少 80% LLM 调用) |
| Phase 2 | ✓ | Few-Shot 检索 (历史提取结果注入 prompt) |
| Phase 3 | ✓ | 语义搜索 (用户搜索文献, 找到相关页面) |
| Dedup | ✓ | 按 fields[0] 合并重复 |
| **TODO** | | LLM 跨页感知去重 (在 prompt 中传已提取实体) |
| **TODO** | | Embedding 语义聚类去重 (识别同义实体) |
| **TODO** | | 本地 embedding 模型 (离线场景) |
| **TODO** | | pgvector 迁移 (大规模部署) |
