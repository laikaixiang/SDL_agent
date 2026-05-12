"""
提取模块门面 (core/extract_manager.py)
===================================

桥接 core 和 extract 包，对外提供统一的提取功能接口。

app.py 通过 core/__init__.py → extract_manager → extract/ 间接调用提取功能。
使用子模块直接导入（而非 from extract import ...）以避免循环导入。
"""

from extract.pdf_processor import PDFProcessor
from extract.extraction_engine import ExtractionEngine

from extract.embedding_service import (
    EmbeddingService,
    APIEmbeddingService,
    JinaEmbeddingService,
    LocalEmbeddingService,
    create_embedding_service,
)

from extract.vector_store import (
    VectorStore,
    ChromaVectorStore,
    PgvectorVectorStore,
)

from extract.page_indexer import (
    PageIndexer,
    make_page_id,
    compute_content_hash,
)

from extract.page_filter import PageFilter
from extract.few_shot_retriever import FewShotRetriever
from extract.algorithm_guide import AlgorithmGuide

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
    'AlgorithmGuide',
]
