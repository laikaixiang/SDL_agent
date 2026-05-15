"""
文献库索引器
管理 SQLite 注册表，编排批量提取流程，处理去重和文件重命名
支持基于第1页embedding的语义搜索 + 单篇文件提取
"""

import os
import re
import sqlite3
import hashlib
import json
from typing import Optional, Callable

from utils.pdf_metadata_extractor import PDFMetadataExtractor, PDFMetadata
from utils.batch_processor import BatchProcessor
from core.config import Config
from extract.embedding_service import create_embedding_service
from extract.vector_store import ChromaVectorStore
from extract.page_filter import PageFilter, make_page_id
from extract.page_indexer import PageIndexer
from extract.pdf_processor import PDFProcessor


class LiteratureIndexer:
    """
    文献库索引器

    职责：
    - SQLite注册表的创建、查询、增删改
    - 编排批量提取流程：mtime检查 → 提取 → 去重 → 重命名 → 写入
    - 增量更新：mtime未变的文件直接跳过
    """

    def __init__(self, config: Optional[Config] = None):
        """
        初始化索引器

        Args:
            config: 配置对象，为None则使用默认Config
        """
        self.config = config or Config()
        self.db_path = self.config.LITERATURE_REGISTRY_DB_PATH
        self.extractor = PDFMetadataExtractor(self.config)
        self.batch_processor = BatchProcessor(
            max_workers=self.config.BATCH_MAX_WORKERS,
            retry_attempts=self.config.METADATA_RETRY_ATTEMPTS
        )
        # 确保数据库和表存在
        self._init_db()

    # ---- 数据库初始化 ----

    def _init_db(self):
        """初始化SQLite注册表"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS literature_registry (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                sanitized_title TEXT,
                authors TEXT,
                abstract_summary TEXT,
                innovation_points TEXT,
                key_image_page INTEGER,
                key_image_x1 REAL, key_image_y1 REAL,
                key_image_x2 REAL, key_image_y2 REAL,
                key_image_desc TEXT,
                doi TEXT,
                arxiv_id TEXT,
                published_date TEXT,
                journal TEXT,
                current_filename TEXT NOT NULL,
                file_hash TEXT,
                file_mtime REAL,
                extraction_status TEXT DEFAULT 'pending',
                error_message TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_literature_title ON literature_registry(title)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_literature_doi ON literature_registry(doi)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_literature_status ON literature_registry(extraction_status)"
        )
        conn.commit()
        conn.close()

    # ---- 文件哈希和时间戳 ----

    @staticmethod
    def compute_file_hash(pdf_path: str) -> str:
        """计算文件的SHA256哈希值"""
        sha256 = hashlib.sha256()
        with open(pdf_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def compute_file_mtime(pdf_path: str) -> float:
        """获取文件最后修改时间"""
        return os.path.getmtime(pdf_path)

    # ---- 注册表 CRUD ----

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def lookup_by_id(self, unique_id: str) -> Optional[dict]:
        """按唯一ID查询注册表记录"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM literature_registry WHERE id = ?", (unique_id,)
        ).fetchone()
        conn.close()
        if row:
            result = dict(row)
            if result.get('innovation_points'):
                try:
                    result['innovation_points'] = json.loads(result['innovation_points'])
                except json.JSONDecodeError:
                    result['innovation_points'] = []
            return result
        return None

    def lookup_by_title(self, title: str) -> Optional[dict]:
        """按标题查询注册表记录（用于去重验证）"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM literature_registry WHERE title = ?", (title,)
        ).fetchone()
        conn.close()
        if row:
            result = dict(row)
            if result.get('innovation_points'):
                try:
                    result['innovation_points'] = json.loads(result['innovation_points'])
                except json.JSONDecodeError:
                    result['innovation_points'] = []
            return result
        return None

    def delete_record(self, unique_id: str) -> bool:
        """按唯一ID删除注册表记录"""
        conn = self._get_conn()
        conn.execute("DELETE FROM literature_registry WHERE id = ?", (unique_id,))
        conn.commit()
        conn.close()
        return True

    def upsert_record(self, unique_id: str, metadata: PDFMetadata,
                      current_filename: str, file_hash: str, file_mtime: float):
        """插入或更新注册表记录"""
        conn = self._get_conn()
        innovation_json = json.dumps(metadata.innovation_points, ensure_ascii=False)

        ki = metadata.key_image
        conn.execute("""
            INSERT OR REPLACE INTO literature_registry
            (id, title, sanitized_title, authors, abstract_summary, innovation_points,
             key_image_page, key_image_x1, key_image_y1, key_image_x2, key_image_y2,
             key_image_desc, doi, arxiv_id, published_date, journal,
             current_filename, file_hash, file_mtime,
             extraction_status, error_message, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            unique_id,
            metadata.title,
            PDFMetadataExtractor.sanitize_title_for_filename(metadata.title),
            metadata.authors,
            metadata.abstract_summary,
            innovation_json,
            ki.page if ki else None,
            ki.x1 if ki else None, ki.y1 if ki else None,
            ki.x2 if ki else None, ki.y2 if ki else None,
            ki.description if ki else None,
            metadata.doi,
            metadata.arxiv_id,
            metadata.published_date,
            metadata.journal,
            current_filename,
            file_hash,
            file_mtime,
            'done',
            None
        ))
        conn.commit()
        conn.close()

    # ---- 核心索引流程 ----

    def index_all(self, on_progress: Optional[Callable] = None) -> dict:
        """
        索引 PDF_TARGET 下所有PDF文件

        流程：
        1. 扫描所有PDF
        2. 对每个PDF：mtime检查 → 跳过 or 提取
        3. 提取：LLM提取 → 去重 → 重命名 → 写入注册表

        Returns:
            {total, skipped, extracted, failed, errors}
        """
        pdf_folder = self.config.PDF_FOLDER
        if not os.path.isdir(pdf_folder):
            print(f"PDF目录不存在: {pdf_folder}")
            return {
                "total": 0, "skipped": 0, "extracted": 0, "failed": 0,
                "errors": [{"file": "", "error": f"目录不存在: {pdf_folder}"}]
            }

        pdf_files = [
            os.path.join(pdf_folder, f)
            for f in os.listdir(pdf_folder)
            if f.lower().endswith('.pdf')
        ]

        print(f"扫描到 {len(pdf_files)} 个PDF文件，开始索引...")

        return self.batch_processor.process_all(
            pdf_paths=pdf_files,
            process_one=self._process_single,
            on_progress=on_progress
        )

    def _process_single(self, pdf_path: str) -> dict:
        """
        处理单篇PDF的完整流程

        Returns:
            {"status": "done"|"skipped"|"failed", ...}
        """
        try:
            filename = os.path.basename(pdf_path)

            # 步骤1：计算文件哈希和修改时间
            file_hash = self.compute_file_hash(pdf_path)
            file_mtime = self.compute_file_mtime(pdf_path)

            # 步骤2：生成唯一ID
            unique_id = self.extractor.generate_unique_id(pdf_path)

            # 步骤3：检查注册表中是否存在且未修改
            existing = self.lookup_by_id(unique_id)
            if existing and existing.get('file_mtime') == file_mtime:
                print(f"跳过（未修改）: {filename}")
                return {"status": "skipped", "reason": "文件未修改"}

            # 步骤4：提取元数据
            print(f"提取中: {filename}")
            metadata = self.extractor.extract_metadata(pdf_path)

            # 步骤5：用标题查重（防止不同DOI但同内容）
            title_match = self.lookup_by_title(metadata.title)
            if title_match and title_match['id'] != unique_id:
                print(f"发现标题重复记录，删除旧版本: {title_match['id']}")
                self.delete_record(title_match['id'])

            # 如果当前ID已有旧记录（mtime不同），删除
            if existing:
                print(f"文件已修改，更新记录: {filename}")
                self.delete_record(unique_id)

            # 步骤6：重命名源文件为论文标题
            sanitized_title = PDFMetadataExtractor.sanitize_title_for_filename(metadata.title)
            new_filename = f"{sanitized_title}.pdf"
            new_path = os.path.join(os.path.dirname(pdf_path), new_filename)

            # 避免文件名冲突
            if new_path != pdf_path:
                counter = 1
                while os.path.exists(new_path):
                    name_part = sanitized_title[:70]
                    new_filename = f"{name_part}_{counter}.pdf"
                    new_path = os.path.join(os.path.dirname(pdf_path), new_filename)
                    counter += 1
                os.rename(pdf_path, new_path)
                print(f"文件已重命名: {filename} → {new_filename}")
                current_filename = new_filename
            else:
                current_filename = filename

            # 步骤7：写入注册表
            self.upsert_record(unique_id, metadata, current_filename, file_hash, file_mtime)

            print(f"提取完成: {current_filename}")
            return {
                "status": "done", "id": unique_id, "title": metadata.title,
                "filename": current_filename
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"处理失败 [{pdf_path}]: {e}")
            return {"status": "failed", "error": str(e)}

    # ---- 强制重索引 ----

    def reindex_force(self, pdf_path: Optional[str] = None) -> dict:
        """
        强制重新索引（忽略mtime检查）

        Args:
            pdf_path: 指定单个PDF路径，为None则重新索引全部
        """
        if pdf_path:
            unique_id = self.extractor.generate_unique_id(pdf_path)
            existing = self.lookup_by_id(unique_id)
            if existing:
                self.delete_record(unique_id)
                print(f"已删除旧记录: {unique_id}")

            result = self._process_single(pdf_path)
            return {
                "total": 1, "skipped": 0,
                "extracted": 1 if result["status"] == "done" else 0,
                "failed": 1 if result["status"] == "failed" else 0,
                "errors": [result] if result["status"] == "failed" else []
            }
        else:
            # 全部强制重索引：删除所有记录后重新索引
            conn = self._get_conn()
            conn.execute("DELETE FROM literature_registry")
            conn.commit()
            conn.close()
            print("已清空注册表，开始全量重索引...")
            return self.index_all()

    # ---- 查询接口 ----

    def query_registry(self, status: Optional[str] = None,
                       page: int = 1, limit: int = 20) -> dict:
        """
        分页查询注册表

        Args:
            status: 过滤状态 (done/failed/pending)，None表示全部
            page: 页码（从1开始）
            limit: 每页条数

        Returns:
            {entries: [...], total: 总数, page: 当前页, limit: 每页条数}
        """
        conn = self._get_conn()
        offset = (page - 1) * limit

        if status:
            count_row = conn.execute(
                "SELECT COUNT(*) FROM literature_registry WHERE extraction_status = ?",
                (status,)
            ).fetchone()
            rows = conn.execute(
                """SELECT id, title, authors, abstract_summary, innovation_points,
                   key_image_desc, doi, current_filename, extraction_status,
                   created_at, updated_at
                   FROM literature_registry WHERE extraction_status = ?
                   ORDER BY updated_at DESC LIMIT ? OFFSET ?""",
                (status, limit, offset)
            ).fetchall()
        else:
            count_row = conn.execute(
                "SELECT COUNT(*) FROM literature_registry"
            ).fetchone()
            rows = conn.execute(
                """SELECT id, title, authors, abstract_summary, innovation_points,
                   key_image_desc, doi, current_filename, extraction_status,
                   created_at, updated_at
                   FROM literature_registry
                   ORDER BY updated_at DESC LIMIT ? OFFSET ?""",
                (limit, offset)
            ).fetchall()

        total = count_row[0]
        entries = []
        for row in rows:
            entry = dict(row)
            if entry.get('innovation_points'):
                try:
                    entry['innovation_points'] = json.loads(entry['innovation_points'])
                except json.JSONDecodeError:
                    entry['innovation_points'] = []
            entries.append(entry)

        conn.close()
        return {"entries": entries, "total": total, "page": page, "limit": limit}

    def get_detail(self, unique_id: str) -> Optional[dict]:
        """按ID查询单篇文献完整详情"""
        record = self.lookup_by_id(unique_id)
        if not record:
            return None

        # 重建 ImageBBox 信息
        if record.get('key_image_page'):
            record['key_image'] = {
                "page": record['key_image_page'],
                "x1": record['key_image_x1'],
                "y1": record['key_image_y1'],
                "x2": record['key_image_x2'],
                "y2": record['key_image_y2'],
                "description": record.get('key_image_desc', '')
            }
        else:
            record['key_image'] = None

        return record

    # ---- 搜索与单篇提取 ----

    def _init_embedding_services(self):
        """延迟初始化embedding服务和向量存储，并确保页面索引是最新的"""
        if not hasattr(self, '_embedding_service') or self._embedding_service is None:
            try:
                self._embedding_service = create_embedding_service()
                self._vector_store = ChromaVectorStore(
                    persist_dir=self.config.CHROMADB_PERSIST_DIR
                )
                self._pdf_processor = PDFProcessor()

                # 初始化页面索引器并更新索引（确保pdf_path与磁盘一致）
                chroma_dir = self.config.CHROMADB_PERSIST_DIR
                sqlite_path = os.path.join(chroma_dir, "page_metadata.db")
                self._page_indexer = PageIndexer(
                    embedding_service=self._embedding_service,
                    vector_store=self._vector_store,
                    sqlite_path=sqlite_path,
                    pdf_processor=self._pdf_processor
                )
                indexed, skipped = self._page_indexer.index_all_pdfs()
                print(f"页面索引更新完成: 新索引{indexed}页, 跳过{skipped}页")
            except Exception as e:
                print(f"embedding服务初始化失败: {e}")
                self._embedding_service = None
                self._vector_store = None
                self._pdf_processor = None
                self._page_indexer = None

    def search_literature(self, query: str, top_k: int = 20) -> dict:
        """
        语义搜索文献库：仅搜索每篇PDF第1页的embedding + 标题相似度加权

        流程：
        1. 将查询文本嵌入为向量
        2. 在ChromaDB中搜索 page_num=0 的页面（每篇文献的第1页）
        3. 关联 literature_registry 获取标题、摘要
        4. 标题关键词匹配加分
        5. 按综合分数排序，返回预览图

        Args:
            query: 搜索查询（自然语言）
            top_k: 返回结果数

        Returns:
            {results: [{id, title, score, preview_image, abstract_summary, ...}],
             total_matches: 总匹配数}
        """
        self._init_embedding_services()
        if not self._embedding_service or not self._vector_store:
            return {"results": [], "total_matches": 0,
                    "error": "embedding服务未初始化，请检查配置"}

        try:
            # 步骤1：嵌入查询
            query_vec = self._embedding_service.embed_text(query)

            # 步骤2：在ChromaDB中搜索，限定 page_num=0（仅第1页）
            raw_results = self._vector_store.search(
                query_vec, top_k=top_k, where={"page_num": 0}
            )

            if not raw_results:
                return {"results": [], "total_matches": 0}

            # 步骤3：关联注册表获取标题，计算综合分数
            enriched = []
            seen_files = set()  # 去重（可能有旧索引残留）
            for item in raw_results:
                pdf_path = item['metadata'].get('pdf_path', '')
                pdf_name = os.path.basename(pdf_path)

                # 过滤：文件不存在则跳过（重命名后的残留记录）
                if not os.path.exists(pdf_path):
                    continue

                # 去重：同一文件只保留一条
                if pdf_path in seen_files:
                    continue
                seen_files.add(pdf_path)

                distance = item['distance']
                embedding_sim = 1.0 - distance  # 余弦距离转相似度

                # 从注册表查标题（按文件名 + 模糊路径匹配）
                registry_entry = self._find_by_filename(pdf_name)
                title = registry_entry['title'] if registry_entry else pdf_name
                abstract = registry_entry['abstract_summary'] if registry_entry else ''
                unique_id = registry_entry['id'] if registry_entry else ''

                # 标题关键词匹配加分
                title_sim = self._keyword_match_score(query, title)

                # 综合分数：embedding 0.7 + 标题 0.3
                combined_score = 0.7 * embedding_sim + 0.3 * title_sim

                enriched.append({
                    'id': unique_id,
                    'title': title,
                    'abstract_summary': abstract[:200] if abstract else '',
                    'pdf_path': pdf_path,
                    'pdf_name': pdf_name,
                    'embedding_score': round(embedding_sim, 4),
                    'title_match_score': round(title_sim, 4),
                    'combined_score': round(combined_score, 4),
                })

            # 按综合分数降序排列
            enriched.sort(key=lambda x: x['combined_score'], reverse=True)

            # 步骤4：为top结果生成第1页预览图
            results = []
            for entry in enriched[:top_k]:
                preview = None
                pdf_path = entry['pdf_path']
                if self._pdf_processor:
                    preview = self._pdf_processor.pdf_page_to_image(pdf_path, 0)

                results.append({
                    'id': entry['id'],
                    'title': entry['title'],
                    'abstract_summary': entry['abstract_summary'],
                    'pdf_path': pdf_path,
                    'pdf_name': entry['pdf_name'],
                    'score': entry['combined_score'],
                    'embedding_score': entry['embedding_score'],
                    'title_match_score': entry['title_match_score'],
                    'preview_image': preview,  # base64 JPEG
                })

            return {
                "results": results,
                "total_matches": len(raw_results),
                "query": query,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"results": [], "total_matches": 0, "error": str(e)}

    def _find_by_filename(self, filename: str) -> Optional[dict]:
        """通过文件名在注册表中查找记录"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM literature_registry WHERE current_filename = ?",
            (filename,)
        ).fetchone()
        conn.close()
        if row:
            result = dict(row)
            if result.get('innovation_points'):
                try:
                    result['innovation_points'] = json.loads(result['innovation_points'])
                except json.JSONDecodeError:
                    result['innovation_points'] = []
            return result
        return None

    @staticmethod
    def _keyword_match_score(query: str, title: str) -> float:
        """计算查询关键词在标题中的匹配分数"""
        if not query or not title:
            return 0.5  # 无查询时给中性分
        query_lower = query.lower()
        title_lower = title.lower()

        # 提取查询中的关键词（中英文分詞）
        keywords = []
        # 英文单词
        eng_words = re.findall(r'[a-zA-Z]{2,}', query_lower)
        keywords.extend(eng_words)
        # 中文字符（2字以上组合）
        chinese_chars = re.findall(r'[一-鿿]{2,}', query)
        keywords.extend(chinese_chars)

        if not keywords:
            return 0.5

        # 计算匹配比例
        matched = sum(1 for kw in keywords if kw in title_lower)
        return matched / len(keywords)

    def extract_single(self, pdf_path: str, task: str) -> dict:
        """
        单篇文献提取：对该PDF的所有页面进行相关性筛选，只提取相关页面

        流程：
        1. 初始化embedding服务
        2. 创建PageFilter，设置任务描述
        3. 遍历PDF所有页面，相关性筛选
        4. 对通过筛选的页面调LLM提取

        Args:
            pdf_path: PDF文件路径
            task: 提取任务描述（如"提取FAPbI3钝化剂的前驱体比例和退火温度"）

        Returns:
            {status, pdf_path, total_pages, relevant_pages, extraction_results, ...}
        """
        self._init_embedding_services()
        if not self._embedding_service or not self._vector_store:
            return {"status": "error", "message": "embedding服务未初始化"}

        # 路径解析：相对路径自动补全PDF_TARGET前缀
        if not os.path.isabs(pdf_path):
            pdf_path = os.path.join(self.config.PDF_FOLDER, os.path.basename(pdf_path))
        if not os.path.exists(pdf_path):
            return {"status": "error", "message": f"文件不存在: {pdf_path}"}

        try:
            import fitz
            from core.llm_client import LLMClient

            # 步骤1：初始化PageFilter
            page_filter = PageFilter(
                embedding_service=self._embedding_service,
                vector_store=self._vector_store,
                threshold=self.config.PAGE_FILTER_THRESHOLD
            )
            page_filter.set_task(task)

            # 步骤2：遍历PDF所有页面
            doc = fitz.open(pdf_path)
            num_pages = len(doc)

            relevant_pages = []
            skipped_pages = []
            for page_num in range(num_pages):
                if page_filter.should_process(pdf_path, page_num):
                    relevant_pages.append(page_num)
                else:
                    skipped_pages.append(page_num)
            doc.close()

            print(f"单篇提取 [{os.path.basename(pdf_path)}]: "
                  f"总{num_pages}页, 相关{len(relevant_pages)}页, "
                  f"跳过{len(skipped_pages)}页")

            if not relevant_pages:
                return {
                    "status": "done",
                    "pdf_path": pdf_path,
                    "total_pages": num_pages,
                    "relevant_pages": 0,
                    "skipped_pages": len(skipped_pages),
                    "extraction_results": [],
                    "message": "没有页面与任务相关"
                }

            # 步骤3：对相关页面调LLM提取
            llm_client = LLMClient(
                api_key=self.config.VL_API_KEY,
                api_url=self.config.VL_API_URL,
            )
            pdf_processor = PDFProcessor()
            extraction_results = []

            for page_num in relevant_pages:
                # 获取页面文本和截图
                doc = fitz.open(pdf_path)
                page = doc[page_num]
                page_text = page.get_text()[:3000]

                # 用现有工具渲染页面截图
                page_image = pdf_processor.pdf_page_to_image(pdf_path, page_num)
                doc.close()

                # 构建提取消息
                content_parts = []
                if page_image:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{page_image}"}
                    })
                content_parts.append({
                    "type": "text",
                    "text": self._build_single_extraction_prompt(task, page_text)
                })

                messages = [{"role": "user", "content": content_parts}]

                result = llm_client.call_api(
                    model=self.config.METADATA_EXTRACTION_MODEL,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=None,
                    timeout=self.config.METADATA_EXTRACTION_TIMEOUT
                )

                if result:
                    content = result['choices'][0]['message']['content']
                    content = re.sub(r'```json\n|\n```|```', '', content).strip()
                    try:
                        parsed = json.loads(content)
                        parsed['_source_page'] = page_num + 1  # 1-based
                        extraction_results.append(parsed)
                    except json.JSONDecodeError:
                        extraction_results.append({
                            '_source_page': page_num + 1,
                            '_raw_content': content[:500]
                        })

            # 步骤4：更新注册表状态
            filename = os.path.basename(pdf_path)
            registry_entry = self._find_by_filename(filename)
            if registry_entry:
                conn = self._get_conn()
                conn.execute(
                    "UPDATE literature_registry SET extraction_status = 'done', "
                    "updated_at = datetime('now') WHERE id = ?",
                    (registry_entry['id'],)
                )
                conn.commit()
                conn.close()

            return {
                "status": "done",
                "pdf_path": pdf_path,
                "pdf_name": filename,
                "total_pages": num_pages,
                "relevant_pages": len(relevant_pages),
                "skipped_pages": len(skipped_pages),
                "extraction_results": extraction_results,
                "task": task,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    @staticmethod
    def _build_single_extraction_prompt(task: str, page_text: str) -> str:
        """构建单篇提取的LLM提示词"""
        prompt = f"""你是学术文献数据提取专家。请根据以下论文页面内容，提取与任务相关的结构化数据。

【提取任务】
{task}

【页面文本内容】
{page_text[:2500]}

请以JSON格式返回提取到的数据，字段名使用英文，值使用原文语言：
{{{{
  "entries": [
    {{"field1": "value1", "field2": "value2", ...}}
  ]
}}}}
如果页面中没有与任务相关的数据，返回 {{{{"entries": []}}}}。
仅返回JSON，不要添加其他内容。"""
        return prompt
