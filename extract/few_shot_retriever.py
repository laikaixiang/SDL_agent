"""
Phase 2: Few-Shot 示例检索模块
==============================

从历史提取结果中检索与当前任务最相似的提取记录，作为 Few-Shot 示例
注入 LLM prompt，提高提取准确性和一致性。

核心流程：
1. save_extraction(): 每次 LLM 提取完成后，将结果存入 SQLite
2. retrieve_examples(): 提取开始前，用任务描述 embedding 搜索向量库，
   找到相似页面 → 查 SQLite 获取历史提取结果 → 返回作为 prompt 示例

SQLite 表结构：
  extraction_history: 存储每次提取的完整记录（JSON 格式），
  按 page_id 关联到已索引的页面，支持按任务描述检索。

检索策略：
  - 先用向量搜索找相似页面（语义匹配）
  - 再从 SQLite 查这些页面的历史提取结果
  - 搜索 top_k * 3 个页面以留足缓冲（部分页面可能没有提取记录）
  - 返回最多 top_k 条不重复的示例
"""

import json
import os
import sqlite3
from typing import Optional

from core.config import Config
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .page_indexer import make_page_id


class FewShotRetriever:
    """
    Phase 2 Few-Shot 示例检索器

    在 LLM 提取前，从历史提取记录中检索与当前任务最相关的示例，
    注入到 system prompt 中，帮助 LLM 理解提取格式和内容期望。

    使用方式：
        retriever = FewShotRetriever(embedding_service, vector_store, sqlite_path)

        # 提取完成后保存
        retriever.save_extraction(page_id, extracted_dict, task_description, source_doc)

        # 下次提取前检索示例
        examples = retriever.retrieve_examples(task_description, fields, top_k=3)
    """

    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore,
                 sqlite_path: str):
        """
        初始化 Few-Shot 检索器

        Args:
            embedding_service: Embedding 服务实例，用于将任务描述转为向量
            vector_store: 向量存储实例，用于搜索相似页面
            sqlite_path: SQLite 数据库文件路径（建议放在 CHROMADB_PERSIST_DIR 下）
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.sqlite_path = sqlite_path
        self._init_db()

    def _init_db(self):
        """
        初始化 SQLite 数据库

        创建 extraction_history 表（如果不存在）。
        每条记录对应一次提取事件的一个字段-值对，
        便于按字段名筛选和按页面聚合。
        """
        os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS extraction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    page_id TEXT NOT NULL,
                    source_doc TEXT,
                    task_description TEXT NOT NULL,
                    extracted_json TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_page_id
                ON extraction_history(page_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_task
                ON extraction_history(task_description)
            """)

    def save_extraction(self, page_id: str, extracted_data: dict,
                        task_description: str, source_doc: str = ""):
        """
        保存一次提取结果到历史数据库

        在 LLM 成功提取一个页面的数据后调用，将完整的提取结果
        以 JSON 格式存储，供后续 Few-Shot 检索使用。

        Args:
            page_id: 页面唯一标识符（通过 make_page_id 生成）
            extracted_data: 提取到的数据字典，如 {"passivation_agent": "PEAI", ...}
            task_description: 本次提取任务描述
            source_doc: 来源文档名（不含路径和扩展名）
        """
        extracted_json = json.dumps(extracted_data, ensure_ascii=False)
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """INSERT INTO extraction_history
                   (page_id, source_doc, task_description, extracted_json)
                   VALUES (?, ?, ?, ?)""",
                (page_id, source_doc, task_description, extracted_json)
            )

    def retrieve_examples(self, task_description: str, fields: list[str],
                          top_k: int = 3) -> list[dict]:
        """
        检索与当前任务最相关的历史提取示例

        检索步骤：
        1. 将任务描述转为 embedding 向量
        2. 在向量库中搜索语义最相似的页面（搜索 top_k * 3 以留足缓冲）
        3. 对每个相似页面，在 SQLite 中查找历史提取记录
        4. 提取 JSON 数据并按页面去重（同一页面只取最新一条）
        5. 返回最多 top_k 条示例

        Args:
            task_description: 当前提取任务描述
            fields: 当前提取字段列表（用于参考，暂不做字段级筛选）
            top_k: 返回的最大示例数

        Returns:
            示例列表，每个元素为提取数据的字典，如 [
                {"passivation_agent": "PEAI", "concentration": "5 mg/mL", ...},
                ...
            ]
            如果没有找到任何历史记录，返回空列表
        """
        if self.vector_store.count() == 0:
            return []

        # 1. 将任务描述转为向量
        query_embedding = self.embedding_service.embed_text(task_description)

        # 2. 搜索最相似的页面（多搜一些以应对部分页面无提取记录的情况）
        search_k = top_k * 3
        results = self.vector_store.search(query_embedding, top_k=search_k)

        if not results:
            return []

        # 3. 提取相似页面的 page_id 列表
        similar_page_ids = [r["id"] for r in results]

        # 4. 在 SQLite 中查找这些页面的历史提取记录
        examples = []
        seen_pages = set()

        with sqlite3.connect(self.sqlite_path) as conn:
            for page_id in similar_page_ids:
                if len(examples) >= top_k:
                    break
                if page_id in seen_pages:
                    continue
                seen_pages.add(page_id)

                row = conn.execute(
                    """SELECT extracted_json, source_doc FROM extraction_history
                       WHERE page_id = ?
                       ORDER BY id DESC LIMIT 1""",
                    (page_id,)
                ).fetchone()

                if row:
                    try:
                        extracted = json.loads(row[0])
                        # 清理内部字段（_source_doc 等不应该出现在示例中）
                        extracted = {k: v for k, v in extracted.items()
                                     if not k.startswith('_')}
                        if extracted:
                            examples.append(extracted)
                    except json.JSONDecodeError:
                        continue

        return examples

    def count(self) -> int:
        """
        获取历史提取记录总数

        Returns:
            已存储的提取记录数
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM extraction_history").fetchone()
            return row[0] if row else 0
