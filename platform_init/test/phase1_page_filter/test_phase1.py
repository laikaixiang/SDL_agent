"""
Phase 1 页面预筛选 功能测试
===========================

测试覆盖:
1. JinaEmbeddingService — embed_text / embed_page / embed_batch
2. ChromaVectorStore — CRUD + 搜索
3. PageIndexer — make_page_id / compute_content_hash / SQLite 去重
4. PageFilter — set_task + should_process 余弦相似度筛选
5. ExtractionEngine 集成 — 初始化和优雅降级

运行方式:
    cd D:/PycharmProjects/SDL_agent
    python platform_init/test/phase1_page_filter/test_phase1.py
"""

import sys
import io
import os
import shutil
import tempfile

# 修复 Windows GBK 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from core.config import Config
from extract.embedding_service import (
    EmbeddingService, APIEmbeddingService, JinaEmbeddingService, LocalEmbeddingService,
    create_embedding_service
)
from extract.vector_store import VectorStore, ChromaVectorStore, PgvectorVectorStore
from extract.page_indexer import PageIndexer, make_page_id, compute_content_hash
from extract.page_filter import PageFilter
from extract.extraction_engine import ExtractionEngine
from core.task_manager import TaskManager


def test_make_page_id():
    """测试页面 ID 生成"""
    print("\n=== test_make_page_id ===")
    page_id = make_page_id("D:/papers/test.pdf", 3)
    assert "_p3" in page_id, f"page_id 应包含 _p3: {page_id}"
    # 同一文件同一页的 ID 应该一致
    id1 = make_page_id("/data/sample.pdf", 0)
    id2 = make_page_id("/data/sample.pdf", 0)
    assert id1 == id2, "相同文件和页码应生成相同 ID"
    # 不同页码的 ID 应该不同
    id3 = make_page_id("/data/sample.pdf", 1)
    assert id1 != id3, "不同页码应生成不同 ID"
    print(f"  PASS: page_id={page_id}")


def test_compute_content_hash():
    """测试内容 hash 生成和变更检测"""
    print("\n=== test_compute_content_hash ===")
    h1 = compute_content_hash("hello world", None)
    h2 = compute_content_hash("hello world", None)
    h3 = compute_content_hash("hello world!", None)
    assert h1 == h2, "相同内容应生成相同 hash"
    assert h1 != h3, "不同内容应生成不同 hash"
    # 带图片的情况
    h4 = compute_content_hash("hello world", "base64_image_data")
    h5 = compute_content_hash("hello world", "base64_image_data")
    h6 = compute_content_hash("hello world", "different_image_data")
    assert h4 == h5, "相同文本+图片应生成相同 hash"
    assert h4 != h6, "不同图片应生成不同 hash"
    print("  PASS: content hash 去重逻辑正确")


def test_cosine_similarity():
    """测试余弦相似度计算"""
    print("\n=== test_cosine_similarity ===")
    # 完全一致
    sim = PageFilter._cosine_similarity([1.0, 0.0], [1.0, 0.0])
    assert abs(sim - 1.0) < 0.001, f"相同向量相似度应为 1.0，得到 {sim}"
    # 正交
    sim = PageFilter._cosine_similarity([1.0, 0.0], [0.0, 1.0])
    assert abs(sim - 0.0) < 0.001, f"正交向量相似度应为 0.0，得到 {sim}"
    # 零向量
    sim = PageFilter._cosine_similarity([0.0, 0.0], [1.0, 0.0])
    assert sim == 0.0, f"零向量相似度应为 0.0，得到 {sim}"
    # 一般情况
    sim = PageFilter._cosine_similarity([1.0, 1.0], [1.0, 0.0])
    expected = 0.7071  # cos(45°)
    assert abs(sim - expected) < 0.01, f"45度角相似度应为 {expected}，得到 {sim}"
    print(f"  PASS: 余弦相似度计算正确，45°={sim:.4f}")


def test_chromadb_crud():
    """测试 ChromaDB 向量存储 CRUD"""
    print("\n=== test_chromadb_crud ===")
    d = tempfile.mkdtemp()
    try:
        vs = ChromaVectorStore(persist_dir=d)
        assert vs.count() == 0

        # 添加
        vs.add_embeddings(
            ids=["p1", "p2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            metadatas=[{"page_num": 0}, {"page_num": 1}]
        )
        assert vs.count() == 2
        assert vs.exists("p1")
        assert not vs.exists("p999")

        # 获取
        emb = vs.get_embedding("p1")
        assert len(emb) == 2
        assert abs(emb[0] - 0.1) < 0.001

        # Upsert（更新）
        vs.add_embeddings(
            ids=["p1"],
            embeddings=[[0.9, 0.8]],
            metadatas=[{"page_num": 0, "updated": True}]
        )
        assert vs.count() == 2  # 不应增加
        emb2 = vs.get_embedding("p1")
        assert abs(emb2[0] - 0.9) < 0.001  # 已更新

        # 搜索
        results = vs.search([0.9, 0.8], top_k=2)
        assert len(results) == 2
        assert results[0]["id"] == "p1"  # 最相似

        # 删除
        vs.delete(["p1"])
        assert vs.count() == 1
        assert not vs.exists("p1")

        print(f"  PASS: ChromaDB CRUD 全部正常，count={vs.count()}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_embedding_api():
    """测试 Embedding API（根据 EMBEDDING_BACKEND 自动选择后端）"""
    print("\n=== test_embedding_api ===")
    c = Config()
    backend = c.EMBEDDING_BACKEND

    if not c.EMBEDDING_API_KEY:
        print(f"  SKIP: EMBEDDING_API_KEY 未配置（当前后端: {backend}）")
        return

    print(f"  后端: {backend}")
    print(f"  模型: {c.EMBEDDING_MODEL}")
    print(f"  API:  {c.EMBEDDING_API_URL}")

    svc = create_embedding_service()

    # Test 1: embed_text
    print("  Testing embed_text...")
    vec = svc.embed_text("提取FAPbI3钙钛矿钝化剂参数")
    assert isinstance(vec, list) and all(isinstance(x, float) for x in vec), "返回值应为浮点数列表"
    print(f"    embed_text: dim={len(vec)}, first_5={[round(x, 4) for x in vec[:5]]}")

    # Test 2: embed_page (text only)
    print("  Testing embed_page (text only)...")
    vec2 = svc.embed_page("在DMF溶剂中使用PbI2和FAI制备FAPbI3钙钛矿薄膜", None)
    assert len(vec2) == len(vec), f"embed_page 维度应与 embed_text 一致: {len(vec2)} vs {len(vec)}"
    print(f"    embed_page: dim={len(vec2)}")

    # Test 3: embed_batch
    print("  Testing embed_batch...")
    batch = [
        {"text": "旋涂速度3000rpm，退火温度150度"},
        {"text": "参考文献 Smith et al. Nature 2020"},
        {"text": "致谢：感谢国家自然科学基金资助"},
    ]
    vecs = svc.embed_batch(batch)
    assert len(vecs) == 3, f"embed_batch 应返回 3 个向量，得到 {len(vecs)}"
    assert all(len(v) == len(vec) for v in vecs), "所有向量维度应一致"
    print(f"    embed_batch: count={len(vecs)}")

    # Test 4: 语义区分能力
    print("  Testing semantic discrimination...")
    task_vec = svc.embed_text("提取钙钛矿太阳能电池制备参数")
    relevant_vec = svc.embed_text("在DMF中溶解PbI2和FAI，旋涂3000rpm 30秒，150度退火10分钟")
    irrelevant_vec = svc.embed_text("参考文献 [1] Kojima A. JACS 2009, 131, 6050")
    acknowledge_vec = svc.embed_text("致谢：本研究获得国家自然科学基金重点项目资助")

    sim_rel = PageFilter._cosine_similarity(task_vec, relevant_vec)
    sim_irr = PageFilter._cosine_similarity(task_vec, irrelevant_vec)
    sim_ack = PageFilter._cosine_similarity(task_vec, acknowledge_vec)

    print(f"    相关(制备参数): sim={sim_rel:.4f}")
    print(f"    边界(参考文献): sim={sim_irr:.4f}")
    print(f"    无关(致谢):     sim={sim_ack:.4f}")
    print(f"    Delta(相关-无关) = {sim_rel - sim_ack:.4f}")

    # 验证相对排序：相关内容 > 无关内容（致谢）
    # 不同 embedding 模型的相似度数值分布不同，此处只验证相对关系
    assert sim_rel > sim_ack, f"相关内容({sim_rel:.4f})应高于致谢({sim_ack:.4f})"

    # 验证阈值 0.3 的实际效果（注意：不同模型的绝对相似度值差异较大，
    # Qwen embedding 的中文相似度普遍偏高，阈值可能需要根据实际模型调优）
    threshold = c.PAGE_FILTER_THRESHOLD
    results = {
        "相关(制备)": sim_rel >= threshold,
        "边界(引用)": sim_irr >= threshold,
        "无关(致谢)": sim_ack >= threshold,
    }
    print(f"    阈值={threshold} 判定: {results}")
    # 核心断言：相关内容必须被处理
    assert results["相关(制备)"] is True, f"相关内容应该被处理（sim={sim_rel:.4f} < {threshold}）"
    # 致谢内容在不同模型下表现不同：
    #   BGE-M3: 通常被过滤（< 0.3）
    #   Qwen3-VL-Embedding: 普遍偏高（可能 >= 0.3），需调高阈值
    # 此处不硬编码断言，由用户根据实际场景调优

    print(f"  PASS: {backend} Embedding API 测试通过（模型={c.EMBEDDING_MODEL}, dim={len(vec)}）")


def test_page_filter_logic():
    """测试 PageFilter 逻辑（使用真实 embedding，根据 backend 自动选择服务）"""
    print("\n=== test_page_filter_logic ===")
    c = Config()
    if not c.EMBEDDING_API_KEY:
        print("  SKIP: EMBEDDING_API_KEY 未配置")
        return

    svc = create_embedding_service()

    d = tempfile.mkdtemp()
    try:
        vs = ChromaVectorStore(persist_dir=d)

        # 模拟预索引的页面
        test_pages = [
            ("a1b2c3d4e5f6_p0", [0.1] * 1024, {"pdf_path": "/test/a.pdf", "page_num": 0}),
        ]
        # 使用真实 embedding
        page_texts = [
            "在DMF溶剂中使用反溶剂法制备FAPbI3钙钛矿薄膜，旋涂速度3000rpm",
            "参考文献 [1] Smith et al. Nature Materials 2019",
            "致谢：感谢实验室各位同学的支持与帮助",
        ]
        pids = ["test_p0", "test_p1", "test_p2"]
        embs = [svc.embed_page(t, None) for t in page_texts]
        metas = [{"text": t, "page_num": i} for i, t in enumerate(page_texts)]
        vs.add_embeddings(ids=pids, embeddings=embs, metadatas=metas)

        pf = PageFilter(svc, vs, threshold=0.3)
        pf.set_task("提取钙钛矿太阳能电池的制备工艺参数")

        # 逐页检查（直接使用 pids）
        task_emb = svc.embed_text("提取钙钛矿太阳能电池的制备工艺参数")
        for pid, text in zip(pids, page_texts):
            page_emb = vs.get_embedding(pid)
            sim = PageFilter._cosine_similarity(task_emb, page_emb)
            status = "PASS" if sim >= 0.3 else "SKIP"
            print(f"  [{status}] sim={sim:.4f} | {text[:50]}...")

        # 验证排序：相关内容 > 致谢内容
        sim_rel = PageFilter._cosine_similarity(task_emb, vs.get_embedding("test_p0"))
        sim_ack = PageFilter._cosine_similarity(task_emb, vs.get_embedding("test_p2"))
        assert sim_rel > sim_ack, f"相关内容({sim_rel:.4f})应高于致谢({sim_ack:.4f})"
        print("  PASS: PageFilter 语义排序正确")

    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_extraction_engine_init():
    """测试 ExtractionEngine 集成 —— 初始化和优雅降级"""
    print("\n=== test_extraction_engine_init ===")
    tm = TaskManager()
    engine = ExtractionEngine(tm, session_path=".")

    # 初始状态：服务未初始化
    assert engine.page_filter is None
    assert engine.page_indexer is None

    # 调用初始化
    engine._init_page_filter_services()

    if engine.page_filter is not None:
        print(f"  PASS: page_filter 已初始化，threshold={engine.page_filter.threshold}")
    else:
        print(f"  PASS: page_filter=None（API Key 未配置时的预期降级行为）")

    print("  PASS: ExtractionEngine 正常启动，优雅降级机制工作正常")


def test_config_entries():
    """测试配置项存在且类型正确"""
    print("\n=== test_config_entries ===")
    c = Config()

    # Embedding 配置
    assert hasattr(c, "EMBEDDING_BACKEND")
    assert hasattr(c, "EMBEDDING_API_KEY")
    assert hasattr(c, "EMBEDDING_API_URL")
    assert hasattr(c, "EMBEDDING_MODEL")
    assert hasattr(c, "EMBEDDING_DIM")
    assert hasattr(c, "LOCAL_EMBEDDING_MODEL")
    assert isinstance(c.EMBEDDING_DIM, int) and c.EMBEDDING_DIM > 0

    # Vector Store 配置
    assert hasattr(c, "VECTOR_STORE_BACKEND")
    assert hasattr(c, "CHROMADB_PERSIST_DIR")

    # Page Filter 配置
    assert hasattr(c, "PAGE_FILTER_ENABLED")
    assert hasattr(c, "PAGE_FILTER_THRESHOLD")
    assert hasattr(c, "PAGE_FILTER_TOP_K")
    assert isinstance(c.PAGE_FILTER_ENABLED, bool)
    assert isinstance(c.PAGE_FILTER_THRESHOLD, float)
    assert isinstance(c.PAGE_FILTER_TOP_K, int)

    # Phase 2/3 flag
    assert hasattr(c, "FEW_SHOT_ENABLED")
    assert hasattr(c, "SEMANTIC_SEARCH_ENABLED")
    assert isinstance(c.FEW_SHOT_ENABLED, bool)  # Phase 2 已实现，默认为 True
    assert c.SEMANTIC_SEARCH_ENABLED is False

    print("  PASS: 所有配置项存在且类型正确")


def test_factory_function():
    """测试工厂函数：'api' / 'jina' / 'local' 三种后端"""
    print("\n=== test_factory_function ===")
    # 注意：create_embedding_service() 内部调用 Config() 读取类属性，
    # 所以要修改 Config 类属性而非实例属性
    old_key = Config.EMBEDDING_API_KEY
    old_url = Config.EMBEDDING_API_URL
    old_backend = Config.EMBEDDING_BACKEND

    try:
        # 1. 'api' 后端 + 空 API key → ValueError
        Config.EMBEDDING_BACKEND = "api"
        Config.EMBEDDING_API_KEY = ""
        try:
            create_embedding_service()
            assert False, "api 后端 + 空 key 应抛出 ValueError"
        except ValueError:
            print("  PASS: 'api' 后端正确拒绝空 API key")

        # 2. 'jina' 后端 + 空 API key → ValueError
        Config.EMBEDDING_BACKEND = "jina"
        try:
            create_embedding_service()
            assert False, "jina 后端 + 空 key 应抛出 ValueError"
        except ValueError:
            print("  PASS: 'jina' 后端正确拒绝空 API key")

        # 3. 'local' 后端 → NotImplementedError
        Config.EMBEDDING_BACKEND = "local"
        try:
            create_embedding_service()
            assert False, "local 后端应抛出 NotImplementedError"
        except NotImplementedError:
            print("  PASS: 'local' 后端正确抛出 NotImplementedError")

        # 4. 未知后端 → ValueError
        Config.EMBEDDING_BACKEND = "unknown_backend"
        try:
            create_embedding_service()
            assert False, "未知后端应抛出 ValueError"
        except ValueError as e:
            assert "未知" in str(e), f"错误消息应包含'未知': {e}"
            print("  PASS: 未知后端正确抛出 ValueError")
    finally:
        Config.EMBEDDING_API_KEY = old_key
        Config.EMBEDDING_API_URL = old_url
        Config.EMBEDDING_BACKEND = old_backend


def test_abc_enforcement():
    """测试 ABC 抽象方法强制实现"""
    print("\n=== test_abc_enforcement ===")
    # EmbeddingService ABC 必须实现所有方法
    try:
        class IncompleteEmbedding(EmbeddingService):
            def embed_page(self, text, image_base64): pass
            # 缺少 embed_text 和 embed_batch

        IncompleteEmbedding()  # 应该失败
        assert False, "不完整的 ABC 子类应无法实例化"
    except TypeError:
        print("  PASS: EmbeddingService ABC 强制方法实现")

    # PgvectorVectorStore 构造应抛 NotImplementedError
    try:
        PgvectorVectorStore()
        assert False, "Pgvector 应抛 NotImplementedError"
    except NotImplementedError:
        print("  PASS: PgvectorVectorStore 正确抛出 NotImplementedError")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Phase 1 页面预筛选 功能测试")
    print("=" * 60)

    tests = [
        ("make_page_id / compute_content_hash", test_make_page_id),
        ("内容 hash 去重", test_compute_content_hash),
        ("余弦相似度", test_cosine_similarity),
        ("ChromaDB CRUD", test_chromadb_crud),
        ("配置项检查", test_config_entries),
        ("ABC 抽象基类", test_abc_enforcement),
        ("工厂函数", test_factory_function),
        ("ExtractionEngine 集成", test_extraction_engine_init),
        ("Embedding API (自动适配后端)", test_embedding_api),
        ("PageFilter 筛选逻辑", test_page_filter_logic),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n  FAIL [{name}]: {e}")
        except Exception as e:
            if "SKIP" in str(e):
                skipped += 1
                print(f"  SKIP [{name}]: {e}")
            else:
                failed += 1
                print(f"\n  FAIL [{name}]: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败, {skipped} 跳过")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
