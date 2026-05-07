"""
Phase 3 语义搜索 功能测试
=========================

测试覆盖:
1. SemanticSearch 初始化
2. search() Mock 服务 — 可控向量搜索验证
3. 空状态处理 — 无索引页面 / SQLite 不存在
4. 结果结构验证 — 所有字段存在且类型正确
5. 相似度计算 — 余弦距离→相似度转换
6. text_snippet 截断 — 300 字符限制
7. get_total_pages — 与 vector_store 一致
8. Flask API 路由 — /api/semantic_search + /api/page_image
9. 优雅降级 — 服务未初始化时的 503 响应
10. 真实 API 集成测试 — 需要 EMBEDDING_API_KEY

运行方式:
    cd D:/PycharmProjects/SDL_agent
    python platform_init/test/phase3_semantic_search/test_phase3.py
"""

import sys
import io
import os
import json
import shutil
import tempfile
import sqlite3

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from core.config import Config
from extract.embedding_service import EmbeddingService
from extract.vector_store import ChromaVectorStore
from extract.page_indexer import make_page_id
from extract.semantic_search import SemanticSearch


# ============================================================================
# Mock Embedding Service
# ============================================================================

class MockEmbeddingService(EmbeddingService):
    """
    模拟 Embedding 服务

    策略：返回的向量前两个元素编码"语义分组"。
    - 包含 "perovskite"/"钙钛矿"/"passivation" → group=(0.95, 0.05)
    - 包含 "reference"/"参考文献" → group=(0.10, 0.90)
    - 包含 "solar"/"电池" → group=(0.80, 0.20)
    - 其他 → group=(0.50, 0.50)
    """

    DIM = 128

    def _group(self, text: str) -> tuple:
        t = text.lower()
        if any(k in t for k in ["perovskite", "钙钛矿", "passivation", "钝化"]):
            return (0.95, 0.05)
        if any(k in t for k in ["reference", "参考文献", "bibliography"]):
            return (0.10, 0.90)
        if any(k in t for k in ["solar", "电池", "efficiency", "效率"]):
            return (0.80, 0.20)
        return (0.50, 0.50)

    def _make_vec(self, text: str) -> list[float]:
        a, b = self._group(text)
        vec = [0.0] * self.DIM
        vec[0] = a
        vec[1] = b
        return vec

    def embed_text(self, text: str) -> list[float]:
        return self._make_vec(text)

    def embed_page(self, text: str, image_base64=None) -> list[float]:
        return self._make_vec(text)

    def embed_batch(self, pages: list[dict]) -> list[list[float]]:
        return [self._make_vec(p.get("text", "")) for p in pages]


# ============================================================================
# 辅助函数
# ============================================================================

def setup_test_env(populate_sqlite: bool = False):
    """创建测试环境"""
    test_dir = tempfile.mkdtemp(prefix="phase3_test_")
    vector_dir = os.path.join(test_dir, "chromadb")
    os.makedirs(vector_dir, exist_ok=True)

    embedding = MockEmbeddingService()
    vector_store = ChromaVectorStore(persist_dir=vector_dir)
    sqlite_path = os.path.join(test_dir, "page_metadata.db")
    ss = SemanticSearch(embedding, vector_store, sqlite_path)

    if populate_sqlite:
        _create_test_sqlite(sqlite_path)

    return test_dir, embedding, vector_store, ss, sqlite_path


def _create_test_sqlite(sqlite_path: str):
    """创建测试用 SQLite 数据库，含模拟页面元数据"""
    with sqlite3.connect(sqlite_path) as conn:
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
        for i in range(5):
            conn.execute(
                """INSERT OR REPLACE INTO page_embeddings
                   (page_id, pdf_path, page_num, content_hash, text_content, embedding_model)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    f"test_p{i}",
                    f"/data/paper_{i}.pdf",
                    i,
                    f"hash_{i}",
                    f"This is page {i} of paper {i}. " * 10,
                    "mock_model",
                )
            )


def cleanup_test_env(test_dir):
    """清理测试临时目录"""
    shutil.rmtree(test_dir, ignore_errors=True)


# ============================================================================
# 测试 1: 初始化
# ============================================================================

def test_init():
    """验证 SemanticSearch 正确保存构造参数"""
    print("\n=== Test 1: 初始化 ===")
    test_dir, embedding, vector_store, ss, sqlite_path = setup_test_env()
    try:
        assert ss.embedding_service is embedding, "embedding_service 应引用传入的实例"
        assert ss.vector_store is vector_store, "vector_store 应引用传入的实例"
        assert ss.sqlite_path == sqlite_path, "sqlite_path 应匹配"
        print("  PASS: 初始化正确保存所有依赖")
    finally:
        cleanup_test_env(test_dir)


# ============================================================================
# 测试 2: 空向量库搜索
# ============================================================================

def test_search_empty_store():
    """验证向量库为空时返回空列表"""
    print("\n=== Test 2: 空向量库 ===")
    test_dir, embedding, vector_store, ss, _ = setup_test_env()
    try:
        results = ss.search("perovskite solar cell", top_k=10)
        assert results == [], f"空向量库应返回 []，实际: {results}"
        assert ss.get_total_pages() == 0
        print("  PASS: 空向量库正确返回空列表")
    finally:
        cleanup_test_env(test_dir)


# ============================================================================
# 测试 3: Mock 向量搜索 + SQLite 富化
# ============================================================================

def test_search_with_mock():
    """验证完整搜索流程：embed → vector search → SQLite enrich"""
    print("\n=== Test 3: Mock 搜索 + SQLite 富化 ===")
    test_dir, embedding, vector_store, ss, sqlite_path = setup_test_env(populate_sqlite=True)
    try:
        # 索引页面到向量库
        pages = [
            ("p_perov", "perovskite passivation with PEAI improves efficiency"),
            ("p_solar", "solar cell efficiency reaches 25 percent"),
            ("p_ref", "references and bibliography"),
            ("p_perov2", "FAPbI3 perovskite solar cells passivation study"),
        ]
        for pid, text in pages:
            emb = embedding.embed_text(text)
            vector_store.add_embeddings(
                ids=[pid], embeddings=[emb],
                metadatas=[{"pdf_path": f"/data/{pid}.pdf", "page_num": 0}]
            )

        # 同时需要在 SQLite 中有这些 page_id 的记录
        with sqlite3.connect(sqlite_path) as conn:
            for pid, text in pages:
                conn.execute(
                    """INSERT OR REPLACE INTO page_embeddings
                       (page_id, pdf_path, page_num, content_hash, text_content)
                       VALUES (?, ?, 0, ?, ?)""",
                    (pid, f"/data/{pid}.pdf", f"hash_{pid}", text)
                )

        # 搜索 perovskite 相关
        results = ss.search("perovskite passivation", top_k=3)
        assert len(results) >= 2, f"应至少返回 2 条 perovskite 相关结果，实际: {len(results)}"

        # 验证结果结构
        for r in results:
            for key in ["page_id", "pdf_path", "pdf_name", "page_num", "text_snippet", "similarity"]:
                assert key in r, f"结果缺少字段: {key}"
            assert isinstance(r["similarity"], float), f"similarity 应为 float: {type(r['similarity'])}"
            assert 0 <= r["similarity"] <= 1, f"similarity 应在 [0,1]: {r['similarity']}"
            assert len(r["text_snippet"]) <= 300, f"text_snippet 应 <= 300 字符: {len(r['text_snippet'])}"

        # perovskite 页面应排在 reference 页面之前
        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True), f"相似度应降序排列: {sims}"

        # 第一个结果应该与 perovskite 相关（不是 reference）
        top_pid = results[0]["page_id"]
        assert "ref" not in top_pid.lower(), f"顶部结果不应是 reference 页面: {top_pid}"

        assert ss.get_total_pages() == 4, f"应有 4 个索引页面，实际: {ss.get_total_pages()}"

        print(f"  PASS: {len(results)} 条结果，首位 sim={results[0]['similarity']:.4f}")
    finally:
        cleanup_test_env(test_dir)


# ============================================================================
# 测试 4: SQLite 文件不存在时优雅处理
# ============================================================================

def test_missing_sqlite():
    """验证 SQLite 不存在时不崩溃，返回结果中 metadata 为空"""
    print("\n=== Test 4: SQLite 缺失 ===")
    test_dir, embedding, vector_store, ss, _ = setup_test_env(populate_sqlite=False)
    try:
        # 索引一个页面到向量库（但不创建 SQLite）
        emb = embedding.embed_text("perovskite solar cell")
        vector_store.add_embeddings(
            ids=["p1"], embeddings=[emb],
            metadatas=[{"pdf_path": "/data/test.pdf", "page_num": 0}]
        )

        # 搜索 — SQLite 不存在不应崩溃
        results = ss.search("perovskite", top_k=1)
        assert len(results) == 1, f"应返回 1 条结果，实际: {len(results)}"
        # 没有 SQLite 时 metadata 为空
        assert results[0]["pdf_path"] == "", "无 SQLite 时 pdf_path 应为空字符串"
        assert results[0]["text_snippet"] == "", "无 SQLite 时 text_snippet 应为空字符串"
        print("  PASS: SQLite 缺失时优雅降级，不崩溃")
    finally:
        cleanup_test_env(test_dir)


# ============================================================================
# 测试 5: text_snippet 截断
# ============================================================================

def test_text_snippet_truncation():
    """验证文本片段正确截断到 300 字符"""
    print("\n=== Test 5: text_snippet 截断 ===")
    test_dir, embedding, vector_store, ss, sqlite_path = setup_test_env()
    try:
        long_text = "A" * 500
        with sqlite3.connect(sqlite_path) as conn:
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
            conn.execute(
                """INSERT OR REPLACE INTO page_embeddings
                   (page_id, pdf_path, page_num, content_hash, text_content)
                   VALUES (?, ?, 0, ?, ?)""",
                ("p_long", "/data/long.pdf", "hash_long", long_text)
            )

        emb = embedding.embed_text("test query")
        vector_store.add_embeddings(
            ids=["p_long"], embeddings=[emb],
            metadatas=[{"pdf_path": "/data/long.pdf"}]
        )

        results = ss.search("test query", top_k=1)
        assert len(results) == 1
        snippet = results[0]["text_snippet"]
        assert len(snippet) == 300, f"text_snippet 应 <= 300，实际: {len(snippet)}"
        assert snippet == "A" * 300, f"text_snippet 应为前 300 个 'A'，实际: {snippet[:50]}..."
        print(f"  PASS: text_snippet 正确截断到 300 字符")
    finally:
        cleanup_test_env(test_dir)


# ============================================================================
# 测试 6: 中英文混合查询
# ============================================================================

def test_chinese_query():
    """验证中文查询和英文查询都能正常工作（Mock 层面验证向量搜索逻辑）"""
    print("\n=== Test 6: 中英文查询 ===")
    test_dir, embedding, vector_store, ss, sqlite_path = setup_test_env(populate_sqlite=True)
    try:
        # 索引中英文混合页面
        pages_data = [
            ("p_en", "perovskite passivation using PEAI improves PCE"),
            ("p_cn", "钙钛矿太阳能电池的钝化处理提升转换效率"),
        ]
        for pid, text in pages_data:
            emb = embedding.embed_text(text)
            vector_store.add_embeddings(
                ids=[pid], embeddings=[emb],
                metadatas=[{"pdf_path": f"/data/{pid}.pdf", "page_num": 0}]
            )
            with sqlite3.connect(sqlite_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO page_embeddings
                       (page_id, pdf_path, page_num, content_hash, text_content)
                       VALUES (?, ?, 0, ?, ?)""",
                    (pid, f"/data/{pid}.pdf", f"hash_{pid}", text)
                )

        # 英文查询
        en_results = ss.search("perovskite passivation", top_k=2)
        assert len(en_results) >= 1

        # 中文查询 — 因为 Mock 也识别 "钙钛矿" 关键词
        cn_results = ss.search("钙钛矿钝化", top_k=2)
        assert len(cn_results) >= 1

        # 两者第一位应该都是 p_en（perovskite 关键词），因为 Mock 对 "perovskite" 和 "钙钛矿" 的 group 相同
        # 但距离取决于具体向量 … 不做严格断言，只确保不崩溃

        print(f"  PASS: 英文查询 {len(en_results)} 条，中文查询 {len(cn_results)} 条")
    finally:
        cleanup_test_env(test_dir)


# ============================================================================
# 测试 7: _query_sqlite 空输入
# ============================================================================

def test_query_sqlite_empty():
    """验证 _query_sqlite 对空列表返回空字典"""
    print("\n=== Test 7: _query_sqlite 空列表 ===")
    test_dir, _, _, ss, _ = setup_test_env()
    try:
        result = ss._query_sqlite([])
        assert result == {}, f"空列表应返回 {{}}，实际: {result}"
        print("  PASS: 空列表返回空字典")
    finally:
        cleanup_test_env(test_dir)


# ============================================================================
# 测试 8: get_total_pages
# ============================================================================

def test_get_total_pages():
    """验证 get_total_pages 返回正确的已索引页面数"""
    print("\n=== Test 8: get_total_pages ===")
    test_dir, embedding, vector_store, ss, _ = setup_test_env()
    try:
        assert ss.get_total_pages() == 0, "初始值应为 0"

        for i in range(3):
            emb = embedding.embed_text(f"page {i}")
            vector_store.add_embeddings(
                ids=[f"p{i}"], embeddings=[emb],
                metadatas=[{"page_num": i}]
            )
        assert ss.get_total_pages() == 3, f"应有 3 页，实际: {ss.get_total_pages()}"

        # Upsert 不增加
        emb = embedding.embed_text("page 0 updated")
        vector_store.add_embeddings(
            ids=["p0"], embeddings=[emb],
            metadatas=[{"page_num": 0}]
        )
        assert ss.get_total_pages() == 3, f"upsert 后仍应为 3，实际: {ss.get_total_pages()}"

        print(f"  PASS: get_total_pages = {ss.get_total_pages()}")
    finally:
        cleanup_test_env(test_dir)


# ============================================================================
# 测试 9: 结果排序
# ============================================================================

def test_result_order():
    """验证搜索结果严格按相似度降序排列"""
    print("\n=== Test 9: 结果排序 ===")
    test_dir, embedding, vector_store, ss, sqlite_path = setup_test_env(populate_sqlite=True)
    try:
        # 索引语义上从近到远的页面
        pages = [
            ("p_best", "perovskite passivation efficiency improvement"),
            ("p_good", "perovskite solar cell performance"),
            ("p_ok", "solar energy materials"),
            ("p_bad", "references and bibliography section"),
        ]
        for pid, text in pages:
            emb = embedding.embed_text(text)
            vector_store.add_embeddings(
                ids=[pid], embeddings=[emb],
                metadatas=[{"pdf_path": f"/data/{pid}.pdf"}]
            )
            with sqlite3.connect(sqlite_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO page_embeddings
                       (page_id, pdf_path, page_num, content_hash, text_content)
                       VALUES (?, ?, 0, ?, ?)""",
                    (pid, f"/data/{pid}.pdf", f"hash_{pid}", text)
                )

        results = ss.search("perovskite passivation", top_k=4)
        assert len(results) == 4
        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True), f"相似度应严格降序: {sims}"

        # 最佳匹配应该是 p_best
        assert results[0]["page_id"] == "p_best", \
            f"首位应为 p_best，实际: {results[0]['page_id']}"
        # 最差匹配应该是 p_bad (reference)
        assert results[-1]["page_id"] == "p_bad", \
            f"末位应为 p_bad，实际: {results[-1]['page_id']}"

        print(f"  PASS: 排序正确 {[r['page_id'] for r in results]}")
    finally:
        cleanup_test_env(test_dir)


# ============================================================================
# 测试 10: Flask API 路由测试
# ============================================================================

def test_flask_api_routes():
    """使用 Flask test client 测试 API 路由"""
    print("\n=== Test 10: Flask API 路由 ===")
    c = Config()
    if not c.EMBEDDING_API_KEY:
        print("  SKIP: EMBEDDING_API_KEY 未配置，无法测试 Flask 路由")
        return

    from app import app as flask_app

    with flask_app.test_client() as client:
        # 10a: 空 query 应返回 400
        resp = client.post('/api/semantic_search',
                           data=json.dumps({"query": "", "top_k": 5}),
                           content_type='application/json')
        data = resp.get_json()
        assert resp.status_code == 400, f"空 query 应返回 400，实际: {resp.status_code}"
        assert data["success"] is False
        print("  10a PASS: 空 query → 400")

        # 10b: 正常搜索
        resp = client.post('/api/semantic_search',
                           data=json.dumps({"query": "perovskite passivation", "top_k": 3}),
                           content_type='application/json')
        data = resp.get_json()
        assert resp.status_code == 200, f"正常请求应返回 200，实际: {resp.status_code}"
        assert data["success"] is True, f"success 应为 True: {data}"
        assert "query" in data
        assert "total_pages_indexed" in data
        assert "result_count" in data
        assert "results" in data
        assert isinstance(data["results"], list)
        for r in data["results"]:
            for key in ["page_id", "pdf_path", "pdf_name", "page_num", "text_snippet", "similarity"]:
                assert key in r, f"结果缺少字段: {key} (keys: {list(r.keys())})"
        print(f"  10b PASS: 正常搜索 → {data['result_count']} 条结果 (共 {data['total_pages_indexed']} 页)")

        # 10c: top_k 超出范围
        resp = client.post('/api/semantic_search',
                           data=json.dumps({"query": "test", "top_k": 999}),
                           content_type='application/json')
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["result_count"] <= 10, f"top_k 超范围应截断到 10: {data['result_count']}"
        print("  10c PASS: top_k 超范围 → 截断到 10")

        # 10d: 缺少 body
        resp = client.post('/api/semantic_search',
                           data=None,
                           content_type='application/json')
        data = resp.get_json()
        assert resp.status_code == 400, f"无 query 应返回 400，实际: {resp.status_code}"
        print("  10d PASS: 缺少 query → 400")

    print("  PASS: Flask API 路由全部正常")


# ============================================================================
# 测试 11: page_image API 路由
# ============================================================================

def test_flask_page_image():
    """测试 /api/page_image 路由"""
    print("\n=== Test 11: /api/page_image ===")
    c = Config()
    if not c.EMBEDDING_API_KEY:
        print("  SKIP: EMBEDDING_API_KEY 未配置")
        return

    from app import app as flask_app

    with flask_app.test_client() as client:
        # 11a: 不存在的 PDF 应返回 404
        resp = client.post('/api/page_image',
                           data=json.dumps({
                               "pdf_path": "/nonexistent/file.pdf",
                               "page_num": 0
                           }),
                           content_type='application/json')
        assert resp.status_code == 404, f"不存在 PDF 应返回 404，实际: {resp.status_code}"
        print("  11a PASS: 不存在 PDF → 404")

        # 11b: 负页码应返回 400
        resp = client.post('/api/page_image',
                           data=json.dumps({
                               "pdf_path": "some/file.pdf",
                               "page_num": -1
                           }),
                           content_type='application/json')
        assert resp.status_code == 400, f"负页码应返回 400，实际: {resp.status_code}"
        print("  11b PASS: 负页码 → 400")

        # 11c: 获取实际存在的 PDF 页面
        pdf_folder = c.PDF_FOLDER
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        if pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_files[0])
            resp = client.post('/api/page_image',
                               data=json.dumps({
                                   "pdf_path": pdf_path,
                                   "page_num": 0
                               }),
                               content_type='application/json')
            data = resp.get_json()
            assert resp.status_code == 200, f"正常请求应返回 200，实际: {resp.status_code} {data}"
            assert data["success"] is True
            assert "image_base64" in data
            assert len(data["image_base64"]) > 100, "image_base64 不应为空"
            print(f"  11c PASS: 获取 PDF 第1页图片 ({len(data['image_base64'])} 字符)")
        else:
            print("  11c SKIP: PDF_TARGET 中无 PDF 文件")

    print("  PASS: /api/page_image 路由正常")


# ============================================================================
# 测试 12: 真实 API 集成测试
# ============================================================================

def test_real_api_search():
    """使用真实 Embedding API 进行端到端搜索测试"""
    print("\n=== Test 12: 真实 API 搜索 ===")
    c = Config()
    if not c.EMBEDDING_API_KEY:
        print("  SKIP: EMBEDDING_API_KEY 未配置")
        return

    from extract.embedding_service import create_embedding_service

    test_dir = tempfile.mkdtemp(prefix="phase3_real_")
    try:
        embedding = create_embedding_service()
        vector_dir = os.path.join(test_dir, "chromadb")
        os.makedirs(vector_dir, exist_ok=True)
        vector_store = ChromaVectorStore(persist_dir=vector_dir)
        sqlite_path = os.path.join(test_dir, "page_metadata.db")

        # 创建 SQLite 并填充测试数据
        with sqlite3.connect(sqlite_path) as conn:
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

        # 索引模拟页面
        pages = [
            ("rp1", "FAPbI3 perovskite solar cell passivation using PEAI treatment",
             "/data/paper_a.pdf", "The passivation of FAPbI3 perovskite using PEAI significantly improved PCE from 18.5% to 21.2%..."),
            ("rp2", "Effect of 4F-PEAI passivator on device stability",
             "/data/paper_b.pdf", "4F-PEAI treatment enhanced VOC from 1.05V to 1.15V under standard AM1.5G illumination..."),
            ("rp3", "References: perovskite solar cell review papers",
             "/data/paper_c.pdf", "1. Smith et al. Nature 2019 2. Wang et al. Science 2020 3. Li et al. Joule 2021..."),
        ]

        for pid, text, pdf_path, full_text in pages:
            emb = embedding.embed_text(text)
            vector_store.add_embeddings(
                ids=[pid], embeddings=[emb],
                metadatas=[{"pdf_path": pdf_path, "page_num": 0}]
            )
            with sqlite3.connect(sqlite_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO page_embeddings
                       (page_id, pdf_path, page_num, content_hash, text_content)
                       VALUES (?, ?, 0, ?, ?)""",
                    (pid, pdf_path, f"hash_{pid}", full_text)
                )

        ss = SemanticSearch(embedding, vector_store, sqlite_path)

        # 英文查询
        results = ss.search("perovskite passivator molecules", top_k=3)
        assert len(results) >= 2, f"应至少 2 条结果，实际: {len(results)}"

        # rp1 和 rp2（钝化剂相关）应排在 rp3（参考文献）前面
        top_pids = [r["page_id"] for r in results]
        if "rp3" in top_pids:
            ref_idx = top_pids.index("rp3")
            assert ref_idx == len(top_pids) - 1 or ref_idx > 0, \
                f"参考文献页面不应排在首位: {top_pids}"

        # 验证字段完整性
        for r in results:
            assert all(k in r for k in ["page_id", "pdf_path", "pdf_name", "similarity", "text_snippet"])
            assert 0 <= r["similarity"] <= 1

        print(f"  PASS: 真实 API 检索到 {len(results)} 条，首位: {results[0]['page_id']} sim={results[0]['similarity']:.4f}")
    finally:
        cleanup_test_env(test_dir)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("  Phase 3 语义搜索 功能测试")
    print("=" * 60)
    print(f"  Config: SEMANTIC_SEARCH_ENABLED={Config.SEMANTIC_SEARCH_ENABLED}")

    tests = [
        ("初始化", test_init),
        ("空向量库", test_search_empty_store),
        ("Mock 搜索 + SQLite 富化", test_search_with_mock),
        ("SQLite 缺失降级", test_missing_sqlite),
        ("text_snippet 截断", test_text_snippet_truncation),
        ("中英文查询", test_chinese_query),
        ("_query_sqlite 空列表", test_query_sqlite_empty),
        ("get_total_pages", test_get_total_pages),
        ("结果排序", test_result_order),
        ("Flask API 路由", test_flask_api_routes),
        ("/api/page_image 路由", test_flask_page_image),
        ("真实 API 搜索", test_real_api_search),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            if "SKIP" in str(e):
                skipped += 1
                print(f"  SKIP: {name} — {e}")
            else:
                failed += 1
                print(f"  FAIL: {name} — {e}")
                import traceback
                traceback.print_exc()

    print()
    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
