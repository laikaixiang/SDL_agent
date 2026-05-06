# RAG-enhanced Literature Extraction Design

Status: **Phase 1+2 implemented (2026-05-07), Phase 3 pending**
Date: 2026-05-06 (design) / 2026-05-07 (Phase 1+2 done)

## 1. Problem Statement

Current extraction pipeline sends every page of every PDF to the LLM (Vision or Text API):

```
For each PDF → For each page → extract content → LLM call → parse JSON → accumulate results
```

- Many pages are irrelevant (references, background, unrelated sections)
- Each LLM call costs time + tokens
- No reuse across repeated extractions on the same PDF library
- Solution: Database + RAG techniques to filter, cache, and enhance extraction

## 2. Goals

| Phase | Goal | Description |
|-------|------|-------------|
| Phase 1 | Page pre-filtering | Skip irrelevant pages before LLM call using multimodal embedding similarity | **DONE** |
| Phase 2 | Few-shot retrieval | Use historical extraction results as prompt examples to improve accuracy | **DONE** |
| Phase 3 | Semantic search | Full-text semantic search across all indexed documents |

All phases share common infrastructure (VectorStore + EmbeddingService) and are independently usable via config flags.

## 3. Architecture

```
                         ┌─────────────────────────┐
                         │    ExtractionEngine      │
                         │  (minimal changes)        │
                         └──────────┬──────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
          ┌────────────┐  ┌──────────────┐  ┌──────────────┐
          │ PageFilter │  │FewShotRetriever│ │ SemanticSearch│
          │ (Phase 1)  │  │  (Phase 2)    │  │  (Phase 3)   │
          └─────┬──────┘  └──────┬───────┘  └──────┬───────┘
                │                │                  │
                ▼                ▼                  ▼
          ┌────────────────────────────────────────────┐
          │           VectorStore (abstract interface)  │
          │  ┌──────────────┐   ┌──────────────────┐   │
          │  │  ChromaDB    │   │  pgvector (TODO)  │   │
          │  │  (current)   │   │  (future scale)   │   │
          │  └──────────────┘   └──────────────────┘   │
          └────────────────────────────────────────────┘
                │
                ▼
          ┌──────────────────────────────┐
          │   ExtractionCache (SQLite)    │
          │   Structured extraction history│
          └──────────────────────────────┘
```

### Data flow change (with Phase 1 pre-filtering)

```
Before: PDF page → extract text → LLM call → parse JSON → CSV
After:  PDF page → extract text → embedding → similarity check
                             ↓                    ↓
                        irrelevant (skip)    relevant → LLM call (+ few-shot) → JSON → CSV + store in SQLite
```

## 4. Embedding Service (Multimodal)

### 4.1 Abstract Interface

```python
# core/embedding_service.py

from abc import ABC, abstractmethod

class EmbeddingService(ABC):
    """Multimodal embedding abstraction.
    Current implementation: JinaEmbeddingService (API)
    TODO: LocalEmbeddingService (local model, e.g. jina-clip-v2)
    """

    @abstractmethod
    def embed_page(self, text: str, image_base64: str | None) -> list[float]:
        """Embed a single PDF page (text + optional image), return vector."""
        ...

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Text-only embedding (for task description queries)."""
        ...

    @abstractmethod
    def embed_batch(self, pages: list[dict]) -> list[list[float]]:
        """Batch embedding (for PDF ingestion, reduces API calls)."""
        ...
```

### 4.2 Jina AI API Implementation (Current)

```python
class JinaEmbeddingService(EmbeddingService):
    def __init__(self, api_key: str, model: str = "jina-clip-v2"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.jina.ai/v1/embeddings"

    def embed_page(self, text, image_base64):
        # Build multimodal input: {text: ..., image: base64...}
        # Call Jina API, return embedding vector
        ...

    def embed_text(self, text):
        # Text-only embedding via same API
        ...

    def embed_batch(self, pages):
        # Batch call for efficiency during PDF ingestion
        ...
```

### 4.3 Local Model Implementation (TODO)

```python
class LocalEmbeddingService(EmbeddingService):
    """Local model placeholder for future use.
    When implemented, loads model via transformers or ONNX.
    Set EMBEDDING_BACKEND="local" + LOCAL_EMBEDDING_MODEL in config.
    """
    ...
```

### 4.4 Factory Function

```python
def create_embedding_service() -> EmbeddingService:
    config = Config()
    if config.EMBEDDING_BACKEND == "jina":
        return JinaEmbeddingService(
            api_key=config.EMBEDDING_API_KEY,
            model=config.EMBEDDING_MODEL
        )
    elif config.EMBEDDING_BACKEND == "local":
        return LocalEmbeddingService(
            model_path=config.LOCAL_EMBEDDING_MODEL
        )
    raise ValueError(f"Unknown backend: {config.EMBEDDING_BACKEND}")
```

## 5. Vector Store

### 5.1 Abstract Interface

```python
# core/vector_store.py

from abc import ABC, abstractmethod

class VectorStore(ABC):
    """Vector storage abstraction.
    Phase 1-3: ChromaDB implementation.
    TODO: pgvector implementation for large-scale deployment.
    """

    @abstractmethod
    def add_embeddings(self, ids: list[str], embeddings: list[list[float]],
                       metadatas: list[dict]) -> None:
        """Add embeddings with metadata. Duplicate IDs are skipped."""
        ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 20,
               where: dict = None) -> list[dict]:
        """Search by embedding vector, return [{id, metadata, distance}, ...]."""
        ...

    @abstractmethod
    def exists(self, id: str) -> bool:
        """Check if an embedding already exists."""
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete embeddings by ID."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Total number of stored embeddings."""
        ...
```

### 5.2 ChromaDB Implementation

```python
class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: str, collection_name: str = "page_embeddings"):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_embeddings(self, ids, embeddings, metadatas):
        # Use upsert — same ID overwrites, prevents duplicates
        self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def exists(self, id):
        result = self.collection.get(ids=[id])
        return len(result['ids']) > 0

    # ... search, delete, count
```

### 5.3 pgvector Implementation (TODO)

```python
class PgvectorVectorStore(VectorStore):
    """Future: PostgreSQL + pgvector implementation.
    Config switch: VECTOR_STORE_BACKEND="pgvector"
    Requires: PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD
    """
    ...
```

## 6. PDF Page Indexing (One-time, Deduplicated)

### 6.1 Page Identifier

```python
import hashlib

def make_page_id(pdf_path: str, page_num: int) -> str:
    """Unique ID: md5(pdf_path)_p{page_num}"""
    path_hash = hashlib.md5(pdf_path.encode()).hexdigest()[:12]
    return f"{path_hash}_p{page_num}"
```

### 6.2 Content Hash (for change detection)

```python
def compute_content_hash(text: str, image_base64: str | None) -> str:
    """Detect if page content changed since last indexing."""
    content = text + (image_base64[:100] if image_base64 else "")
    return hashlib.sha256(content.encode()).hexdigest()
```

### 6.3 Indexing Flow

```
Trigger: PDF added to library, or first extraction run
  ↓
For each PDF → For each page:
  1. page_id = make_page_id(pdf_path, page_num)
  2. Check SQLite: page_id exists AND content_hash matches?
     → YES: skip (already indexed)
     → NO:  proceed
  3. Extract page content (text + image if has complex content)
  4. Compute multimodal embedding via EmbeddingService.embed_batch()
  5. Store in ChromaDB: id=page_id, embedding, metadata
  6. Store in SQLite: page_id, pdf_path, page_num, content_hash,
     embedding_model, created_at, has_image
```

### 6.4 SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS page_embeddings (
    page_id TEXT PRIMARY KEY,
    pdf_path TEXT NOT NULL,
    page_num INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    text_content TEXT,
    embedding_model TEXT,
    has_image INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_page_pdf_path ON page_embeddings(pdf_path);
```

## 7. Phase 1: Page Pre-Filtering

### 7.1 Flow

```
Task: "提取FAPbI3钝化剂参数"
  ↓
1. page_filter = PageFilter(vector_store, embedding_service, config)
2. query_embedding = embedding_service.embed_text(task_description)
3. For each page_id in candidate pages:
     page_embedding = get from ChromaDB (pre-indexed)
     similarity = cosine_similarity(query_embedding, page_embedding)
     if similarity >= PAGE_FILTER_THRESHOLD:
         send page to LLM for extraction
     else:
         skip (log reason)
```

### 7.2 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PAGE_FILTER_ENABLED` | `True` | Enable/disable pre-filtering |
| `PAGE_FILTER_THRESHOLD` | `0.3` | Cosine similarity threshold (conservative, tunable) |
| `PAGE_FILTER_TOP_K` | `20` | Max pages to send to LLM per task |

### 7.3 Integration in ExtractionEngine

```python
# core/extraction_engine.py — modification in _process_single_pdf()

def _process_single_pdf(self, pdf_path, file_idx, total_files, fields, ...):
    for page_num in range(num_pages):
        if self.task_manager.is_task_cancelled():
            break

        # NEW: Phase 1 pre-filter
        if self.page_filter:
            page_id = make_page_id(pdf_path, page_num)
            if not self.page_filter.should_process(page_id, task_description):
                self.task_manager.put_task_message(
                    "info", f"⏭️ 跳过第{page_num+1}页 (相似度低于阈值)"
                )
                continue

        self._process_single_page(...)
```

## 8. Phase 2: Few-Shot Retrieval (DONE)

### 8.1 Goal

When extracting from a new PDF, retrieve similar historical extractions as examples in the LLM prompt to improve accuracy and consistency.

### 8.2 SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS extraction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id TEXT,
    field_name TEXT NOT NULL,
    field_value TEXT NOT NULL,
    source_doc TEXT,
    task_description TEXT,
    confidence REAL,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (page_id) REFERENCES page_embeddings(page_id)
);

CREATE INDEX IF NOT EXISTS idx_history_field ON extraction_history(field_name);
CREATE INDEX IF NOT EXISTS idx_history_task ON extraction_history(task_description);
```

### 8.3 Retriever Interface

```python
class FewShotRetriever:
    """Retrieve similar historical extractions as few-shot examples."""

    def __init__(self, vector_store, sqlite_path):
        ...

    def retrieve_examples(self, task_description: str, fields: list[str],
                          top_k: int = 3) -> list[dict]:
        """
        1. Embed task_description
        2. Search vector_store for similar pages
        3. Query SQLite for extraction results from those pages
        4. Return top_k examples matching the requested fields
        """
        ...

    def save_extraction(self, page_id: str, extracted_data: dict,
                        task_description: str):
        """Save extraction result to SQLite for future few-shot use."""
        ...
```

### 8.4 Prompt Integration

```python
# In _process_with_text / _process_with_vision

examples = few_shot_retriever.retrieve_examples(task_description, fields, top_k=3)
if examples:
    few_shot_text = "参考历史提取示例：\n" + json.dumps(examples, ensure_ascii=False)
    sys_prompt = few_shot_text + "\n\n" + sys_prompt
```

## 9. Phase 3: Semantic Search (Outline)

### 9.1 Goal

User searches natural language across all indexed document pages, gets relevant chunks, then optionally triggers deep extraction on matched results.

### 9.2 API Route

```python
# app.py

@app.route('/api/semantic_search', methods=['POST'])
def semantic_search():
    """
    POST body: {query: "钙钛矿钝化剂效率对比", top_k: 10}
    Returns: [{page_id, pdf_path, page_num, text_snippet, similarity}, ...]
    """
    query_embedding = embedding_service.embed_text(query)
    results = vector_store.search(query_embedding, top_k=top_k)

    # Enrich with text snippets from SQLite
    return jsonify(enriched_results)
```

### 9.3 Frontend Integration

```
Search bar in UI → /api/semantic_search → display result cards
  ↓
Click result → show PDF page image + text
  ↓
"Extract from this selection" → triggers targeted extraction on matched pages only
```

## 10. Configuration

```python
# core/config.py — new entries

# ==================== Embedding Configuration ====================
EMBEDDING_BACKEND: str = "jina"          # "jina" | "local"
EMBEDDING_API_KEY: str = ""              # Jina AI API key
EMBEDDING_API_URL: str = "https://api.jina.ai/v1/embeddings"
EMBEDDING_MODEL: str = "jina-clip-v2"
EMBEDDING_DIM: int = 1024
LOCAL_EMBEDDING_MODEL: str = ""          # Local model path (future)

# ==================== Vector Store Configuration ====================
VECTOR_STORE_BACKEND: str = "chromadb"   # "chromadb" | "pgvector" (TODO)
CHROMADB_PERSIST_DIR: str = "dialogue data/vector_store"
# pgvector config (future):
# PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD

# ==================== Pre-filter Configuration ====================
PAGE_FILTER_ENABLED: bool = True
PAGE_FILTER_THRESHOLD: float = 0.3       # Cosine similarity threshold
PAGE_FILTER_TOP_K: int = 20              # Max pages per task

# ==================== Few-Shot Configuration ====================
FEW_SHOT_ENABLED: bool = False           # Phase 2 flag (off until implemented)
FEW_SHOT_TOP_K: int = 3                  # Number of examples

# ==================== Semantic Search Configuration ====================
SEMANTIC_SEARCH_ENABLED: bool = False    # Phase 3 flag
```

## 11. Files to Create / Modify

### Phase 1 Files (DONE)

| File | Description |
|------|-------------|
| `core/embedding_service.py` | EmbeddingService ABC + APIEmbeddingService (SiliconFlow/DeepSeek compat) + JinaEmbeddingService (multimodal) + LocalEmbeddingService(TODO) + factory |
| `core/vector_store.py` | VectorStore ABC + ChromaVectorStore + PgvectorVectorStore(TODO) |
| `core/page_indexer.py` | PDF page pre-indexing with dedup: make_page_id(), compute_content_hash(), SQLite metadata, incremental indexing |
| `core/page_filter.py` | Phase 1: query-time page relevance filtering via cosine similarity vs task embedding |

### Modified Files (DONE)

| File | Change |
|------|--------|
| `core/config.py` | Added 17 config keys (EMBEDDING_*, VECTOR_STORE_*, PAGE_FILTER_*, FEW_SHOT_*, SEMANTIC_SEARCH_*) |
| `core/extraction_engine.py` | Added _init_page_filter_services(), pre-indexing in process_pdf_library(), should_process() check per page in _process_single_pdf() |
| `requirements.txt` | Added chromadb |
| `CLAUDE.md` | Updated RAG section with Phase 1 implementation summary |
| `README.md` | Updated Section 13 with Phase 1 completion status |
| `rag_extraction_enhancement_design.md` | Updated status to Phase 1 done |

### Test Files (DONE)

| File | Description |
|------|-------------|
| `platform_init/test/phase1_page_filter/test_phase1.py` | 10 functional tests (CRUD, config, factory, integration, embedding API) |
| `platform_init/test/phase1_page_filter/test_model_comparison.py` | A/B model comparison (BGE-en-v1.5 vs Qwen3-VL-Embedding-8B) |

### Phase 2 Files (DONE)

| File | Description |
|------|-------------|
| `core/few_shot_retriever.py` | Phase 2: few-shot example retrieval from historical extraction results |

### Modified Files (Phase 2, DONE)

| File | Change |
|------|--------|
| `core/extraction_engine.py` | Added `few_shot_retriever` init, `_inject_few_shot_examples()`, `_save_to_extraction_history()`, threaded `task_description` to vision/text methods |
| `core/config.py` | Set `FEW_SHOT_ENABLED=True` (was False) |

### Future Files (Phase 3)

| File | Description |
|------|-------------|
| `core/semantic_search.py` | Phase 3: semantic search logic |
| `app.py` (modify) | Add `/api/semantic_search` route |

## 12. pgvector Migration Path (TODO)

When data scale exceeds ChromaDB limits (>500K vectors):

1. Implement `PgvectorVectorStore(VectorStore)` with same interface
2. Set `VECTOR_STORE_BACKEND = "pgvector"` in config
3. Add PG connection configs
4. Run one-time migration: `scripts/migrate_chromadb_to_pgvector.py`
5. Existing code in `PageFilter`/`FewShotRetriever`/`SemanticSearch` unchanged (all depend on VectorStore interface)
