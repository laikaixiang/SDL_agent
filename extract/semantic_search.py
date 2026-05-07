"""
Phase 3: 语义搜索模块
=====================

用户用自然语言搜索全文献库，返回匹配的页面及其文本片段。

核心流程：
1. 将查询文本转为 embedding 向量
2. 在 ChromaDB 中搜索最相似的页面
3. 从 SQLite 获取页面的文本摘要和文件路径
4. 返回按相似度排序的结果列表

依赖：
- EmbeddingService: 将查询文本转向量
- VectorStore: 向量相似度搜索（余弦距离）
- page_metadata.db: Phase 1 的页面元数据（text_content、pdf_path 等）
"""

import os
import sqlite3
from typing import Optional

from core.config import Config
from extract.embedding_service import EmbeddingService
from extract.vector_store import VectorStore


class SemanticSearch:
    """
    Phase 3 语义搜索引擎

    在已索引的 PDF 文献库中按语义搜索相关页面。

    使用方式：
        ss = SemanticSearch(embedding_service, vector_store, sqlite_path)
        results = ss.search("钙钛矿钝化剂效率对比", top_k=10)
        # → [{page_id, pdf_path, pdf_name, page_num, text_snippet, similarity}, ...]
    """

    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore,
                 sqlite_path: str):
        """
        初始化语义搜索引擎

        Args:
            embedding_service: Embedding 服务实例
            vector_store: 向量存储实例
            sqlite_path: page_metadata.db 的路径（Phase 1 PageIndexer 创建）
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.sqlite_path = sqlite_path

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        按语义搜索全文献库

        Args:
            query: 自然语言查询文本（中文或英文）
            top_k: 返回的最大结果数

        Returns:
            结果列表，每项包含：
            - page_id: 页面唯一标识符
            - pdf_path: PDF 文件路径
            - pdf_name: PDF 文件名（不含目录）
            - page_num: 页码（从 0 开始）
            - text_snippet: 页面文本片段（前 300 字符）
            - similarity: 余弦相似度（0~1，越高越相关）
        """
        if self.vector_store.count() == 0:
            return []

        # 1. 将查询文本转为 embedding 向量
        query_embedding = self.embedding_service.embed_text(query)

        # 2. 向量搜索
        search_results = self.vector_store.search(query_embedding, top_k=top_k)

        if not search_results:
            return []

        # 3. 从 SQLite 获取文本片段和文件路径
        page_ids = [r["id"] for r in search_results]
        distances = {r["id"]: r.get("distance", 0.0) for r in search_results}
        metadata_map = self._query_sqlite(page_ids)

        # 4. 组装结果
        results = []
        for r in search_results:
            pid = r["id"]
            meta = metadata_map.get(pid, {})
            similarity = 1.0 - distances.get(pid, 0.0)

            pdf_path = meta.get("pdf_path", "")
            text = meta.get("text_content", "") or ""
            snippet = text[:300]

            results.append({
                "page_id": pid,
                "pdf_path": pdf_path,
                "pdf_name": os.path.basename(pdf_path) if pdf_path else "",
                "page_num": meta.get("page_num", -1),
                "text_snippet": snippet,
                "similarity": round(similarity, 4),
            })

        return results

    def _query_sqlite(self, page_ids: list[str]) -> dict:
        """
        批量查询 page_metadata.db 获取页面元数据

        Args:
            page_ids: 页面唯一标识符列表

        Returns:
            {page_id: {pdf_path, page_num, text_content}} 的映射字典
        """
        metadata_map = {}
        if not page_ids:
            return metadata_map

        if not os.path.isfile(self.sqlite_path):
            return metadata_map

        try:
            with sqlite3.connect(self.sqlite_path) as conn:
                placeholders = ",".join("?" for _ in page_ids)
                rows = conn.execute(
                    f"""SELECT page_id, pdf_path, page_num, text_content
                        FROM page_embeddings
                        WHERE page_id IN ({placeholders})""",
                    page_ids
                ).fetchall()

                for row in rows:
                    metadata_map[row[0]] = {
                        "pdf_path": row[1] or "",
                        "page_num": row[2] if row[2] is not None else -1,
                        "text_content": row[3] or "",
                    }
        except Exception:
            pass

        return metadata_map

    def get_total_pages(self) -> int:
        """
        获取已索引的总页面数

        Returns:
            ChromaDB 中的向量总数
        """
        return self.vector_store.count()
