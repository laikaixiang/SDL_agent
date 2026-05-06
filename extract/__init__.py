"""
文献提取包 (extract)
==================

包含 PDF 文献提取的完整功能模块：

Phase 1: 页面预筛选
  - embedding_service.py  — Embedding 服务抽象层 (API / Jina / Local)
  - vector_store.py       — 向量存储抽象层 (ChromaDB / pgvector)
  - page_indexer.py       — PDF 页面预索引 (增量 + 去重)
  - page_filter.py        — 页面预筛选 (余弦相似度过滤)

Phase 2: Few-Shot 检索
  - few_shot_retriever.py — 历史提取示例检索 (向量搜索 + SQLite)

核心引擎:
  - extraction_engine.py  — 提取引擎主逻辑 (LLM 交互、数据解析)
  - pdf_processor.py      — PDF 处理 (文本提取、图像转换)

注意：为避免与 core 包的循环导入，__init__.py 不做 eager import。
请从子模块直接导入，或使用 core/extract_manager.py 门面。
"""

__all__ = [
    'PDFProcessor',
    'ExtractionEngine',
    'EmbeddingService',
    'APIEmbeddingService',
    'JinaEmbeddingService',
    'LocalEmbeddingService',
    'create_embedding_service',
    'VectorStore',
    'ChromaVectorStore',
    'PgvectorVectorStore',
    'PageIndexer',
    'make_page_id',
    'compute_content_hash',
    'PageFilter',
    'FewShotRetriever',
]
