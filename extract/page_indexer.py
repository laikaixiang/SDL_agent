"""
PDF 页面预索引模块
=================

在提取任务开始前，将 PDF 文献库中所有页面预先转换为 embedding 向量并存储，
实现一次索引、多次复用的效果。

核心流程：
1. 遍历 PDF 文件夹中的所有 PDF 文件
2. 对每页提取文本（必选）和截图（可选），生成页面唯一 ID
3. 计算内容 hash，跳过未变更的已索引页面（增量索引）
4. 调用 EmbeddingService 生成多模态向量
5. 存入 VectorStore（ChromaDB）+ SQLite 元数据库

页面 ID 规则：
  page_id = md5(pdf_path 的前12位) + "_p" + 页码
  示例：对于 "D:/papers/perovskite.pdf" 的第 3 页 → "a1b2c3d4e5f6_p3"

去重逻辑：
  - SQLite 中按 page_id 存储 content_hash
  - 索引前先比较 content_hash，一致则跳过（相同内容不重复索引）
  - 内容变更则重新生成 embedding 并覆盖旧记录

SQLite 元数据表结构：
  page_id TEXT PRIMARY KEY     -- 页面唯一 ID
  pdf_path TEXT NOT NULL       -- PDF 文件路径
  page_num INTEGER NOT NULL    -- 页码（从 0 开始）
  content_hash TEXT NOT NULL   -- SHA256 内容摘要，用于变更检测
  text_content TEXT            -- 页面文本内容（存储以便 Phase 2/3 使用）
  embedding_model TEXT         -- 使用的 embedding 模型名
  has_image INTEGER DEFAULT 0  -- 是否包含截图（1=有，0=无）
  created_at TEXT              -- 索引时间
"""

import hashlib
import os
import sqlite3
from typing import Optional

from core.config import Config
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .pdf_processor import PDFProcessor


def make_page_id(pdf_path: str, page_num: int) -> str:
    """
    生成页面唯一标识符

    使用 MD5(path) 的前 12 位 + 页码，保证不同文件的不同页面有唯一 ID，
    同时保持可读性和可追溯性。

    Args:
        pdf_path: PDF 文件的绝对路径
        page_num: 页码（从 0 开始）

    Returns:
        格式为 "{12位hash}_p{页码}" 的唯一标识符
    """
    path_hash = hashlib.md5(pdf_path.encode()).hexdigest()[:12]
    return f"{path_hash}_p{page_num}"


def compute_content_hash(text: str, image_base64: Optional[str]) -> str:
    """
    计算页面内容的 SHA256 哈希值

    用于检测页面内容是否自上次索引后发生了变化。
    如果文本相同且图片前 100 字符（头部信息）相同，则认为内容未变。

    Args:
        text: 页面文本内容
        image_base64: 页面截图的 base64 编码，为 None 表示纯文本页面

    Returns:
        SHA256 十六进制字符串
    """
    content = text + (image_base64[:100] if image_base64 else "")
    return hashlib.sha256(content.encode()).hexdigest()


class PageIndexer:
    """
    PDF 页面预索引器

    负责将 PDF 文献库中的所有页面一次性转换为 embedding 向量并存储。
    支持增量索引（只处理新增或变更的页面）和幂等操作（重复运行安全）。

    使用方式：
        indexer = PageIndexer(embedding_service, vector_store, sqlite_path)
        indexed, skipped = indexer.index_all_pdfs()

    典型集成位置：ExtractionEngine.process_pdf_library() 开始时调用一次。
    """

    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore,
                 sqlite_path: str, pdf_processor: Optional[PDFProcessor] = None):
        """
        初始化页面索引器

        Args:
            embedding_service: Embedding 服务实例（如 JinaEmbeddingService）
            vector_store: 向量存储实例（如 ChromaVectorStore）
            sqlite_path: SQLite 元数据库文件路径（建议放在 CHROMADB_PERSIST_DIR 下）
            pdf_processor: PDF 处理器实例，不传则自动创建
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.sqlite_path = sqlite_path
        self.pdf_processor = pdf_processor or PDFProcessor()
        self._init_db()

    def _init_db(self):
        """
        初始化 SQLite 元数据库

        创建 page_embeddings 表和索引（如果不存在）。
        表结构与设计文档 6.4 节定义一致。
        """
        os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS page_embeddings (
                    page_id TEXT PRIMARY KEY,
                    pdf_path TEXT NOT NULL,
                    page_num INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    text_content TEXT,
                    embedding_model TEXT,
                    has_image INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_page_pdf_path
                ON page_embeddings(pdf_path)
            """)

    def index_pdf(self, pdf_path: str) -> tuple[int, int]:
        """
        索引单个 PDF 文件的所有页面

        对 PDF 的每一页：
        1. 提取文本和图片
        2. 生成 page_id 和 content_hash
        3. 检查是否需要更新（compare content_hash）
        4. 调用 embedding API 生成向量
        5. 存入向量库和 SQLite

        如果某页面已在索引中且内容未变更，则跳过该页以节省 API 调用。

        Args:
            pdf_path: PDF 文件的绝对路径

        Returns:
            (indexed, skipped) 元组：本次新增索引的页数和跳过的页数
        """
        pdf_info = self.pdf_processor.get_pdf_info(pdf_path)
        if not pdf_info:
            return 0, 0

        num_pages = pdf_info['num_pages']
        indexed = 0
        skipped = 0

        for page_num in range(num_pages):
            page_id = make_page_id(pdf_path, page_num)

            # 提取页面内容：文本 + 图片
            markdown_text = self.pdf_processor.extract_text_from_page(pdf_path, page_num)
            text = markdown_text or ""
            img_base64 = self.pdf_processor.pdf_page_to_image(pdf_path, page_num)

            # 计算内容 hash，判断是否需要重新索引
            content_hash = compute_content_hash(text, img_base64)
            if self._is_current(page_id, content_hash):
                skipped += 1
                continue

            # 调用 embedding API 生成向量
            embedding = self.embedding_service.embed_page(text, img_base64)

            # 存入向量库（upsert 语义，自动去重）
            metadata = {
                "pdf_path": pdf_path,
                "page_num": page_num,
                "content_hash": content_hash
            }
            self.vector_store.add_embeddings(
                ids=[page_id],
                embeddings=[embedding],
                metadatas=[metadata]
            )

            # 存入 SQLite 元数据库
            model_name = getattr(self.embedding_service, 'model', 'unknown')
            self._upsert_sqlite(page_id, pdf_path, page_num, content_hash,
                               text, model_name, bool(img_base64))

            indexed += 1

        return indexed, skipped

    def _is_current(self, page_id: str, content_hash: str) -> bool:
        """
        检查页面是否已索引且内容未变更

        仅比较 content_hash，如果一致说明页面内容与上次索引时相同，
        无需重新调用 embedding API。

        Args:
            page_id: 页面唯一标识符
            content_hash: 当前计算的内容 hash

        Returns:
            True 表示页面已索引且内容未变，应跳过
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            row = conn.execute(
                "SELECT content_hash FROM page_embeddings WHERE page_id = ?",
                (page_id,)
            ).fetchone()
        return row is not None and row[0] == content_hash

    def _upsert_sqlite(self, page_id: str, pdf_path: str, page_num: int,
                       content_hash: str, text: str, model: str, has_image: bool):
        """
        向 SQLite 中插入或更新页面元数据记录

        使用 INSERT OR REPLACE 实现 upsert：如果 page_id 已存在则更新所有字段，
        否则插入新行。

        Args:
            page_id: 页面唯一 ID
            pdf_path: PDF 文件路径
            page_num: 页码
            content_hash: 内容 hash
            text: 页面文本内容（存储备查）
            model: embedding 模型名称
            has_image: 是否包含截图
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO page_embeddings
                (page_id, pdf_path, page_num, content_hash, text_content, embedding_model, has_image, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (page_id, pdf_path, page_num, content_hash, text, model, int(has_image)))

    def index_all_pdfs(self, pdf_folder: str = None) -> tuple[int, int]:
        """
        索引 PDF 文件夹中的所有 PDF 文件

        这是主要的对外接口，通常在 process_pdf_library() 开始时调用一次。
        遍历配置的 PDF 文件夹，对每个 PDF 调用 index_pdf()。

        Args:
            pdf_folder: PDF 文件夹路径，为 None 则使用 Config.PDF_FOLDER

        Returns:
            (total_indexed, total_skipped) 元组
        """
        if pdf_folder is None:
            pdf_folder = Config().PDF_FOLDER

        pdf_files = self.pdf_processor.list_pdf_files(pdf_folder)
        total_indexed = 0
        total_skipped = 0

        for pdf_path in pdf_files:
            indexed, skipped = self.index_pdf(pdf_path)
            total_indexed += indexed
            total_skipped += skipped

        return total_indexed, total_skipped
