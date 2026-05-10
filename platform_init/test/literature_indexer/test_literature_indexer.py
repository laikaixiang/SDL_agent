"""
文献索引器集成测试
"""

import sys
import os
import json
import tempfile
import shutil

# 添加项目根目录到路径 (platform_init/test/literature_indexer → 根目录，需向上4级)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from extract.literature_indexer import LiteratureIndexer
from utils.pdf_metadata_extractor import PDFMetadataExtractor, PDFMetadata, ImageBBox
from core.config import Config


class TestPDFMetadataExtractor:
    """单元测试：单篇PDF元数据提取器"""

    def test_image_bbox_model(self):
        """测试 ImageBBox 数据模型"""
        bbox = ImageBBox(page=1, x1=10.0, y1=20.0, x2=100.0, y2=200.0, description="测试图片")
        assert bbox.page == 1
        assert bbox.x1 == 10.0
        assert bbox.description == "测试图片"

    def test_pdf_metadata_model(self):
        """测试 PDFMetadata 数据模型"""
        metadata = PDFMetadata(
            title="Test Paper",
            authors="Author A, Author B",
            abstract_summary="summary",
            innovation_points=["point1", "point2"],
            key_image=ImageBBox(page=1, x1=0, y1=0, x2=100, y2=100, description="Fig 1")
        )
        assert metadata.title == "Test Paper"
        assert len(metadata.innovation_points) == 2
        assert metadata.key_image is not None

    def test_sanitize_title_for_filename(self):
        """测试文件名清理"""
        ext = PDFMetadataExtractor()
        # 包含非法字符
        result = ext.sanitize_title_for_filename('Test: "Paper" <Title>?')
        assert ':' not in result
        assert '"' not in result
        assert '<' not in result
        assert '>' not in result
        assert '?' not in result

        # 超长标题截断
        long_title = "A" * 200
        result = ext.sanitize_title_for_filename(long_title, max_len=80)
        assert len(result) <= 80

    def test_sanitize_id(self):
        """测试ID清理"""
        result = PDFMetadataExtractor._sanitize_id("10.1002/adfm.202002366")
        assert result == "10.1002_adfm.202002366"

        result = PDFMetadataExtractor._sanitize_id("")
        assert result == "unknown_id"

    def test_generate_unique_id_from_path(self):
        """测试从文件路径生成唯一ID（无DOI的PDF）"""
        ext = PDFMetadataExtractor()
        uid = ext.generate_unique_id("dialogue data/PDF_TARGET/test.pdf")
        assert len(uid) == 12  # MD5前12位

    def test_extract_doi_from_real_pdf(self):
        """测试从真实PDF提取DOI"""
        pdf_dir = Config.PDF_FOLDER
        if not os.path.isdir(pdf_dir):
            print(f"跳过：PDF目录不存在 {pdf_dir}")
            return

        pdfs = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        if not pdfs:
            print("跳过：PDF目录为空")
            return

        ext = PDFMetadataExtractor()
        pdf_path = os.path.join(pdf_dir, pdfs[0])
        doi = ext.extract_doi_from_pdf(pdf_path)
        print(f"从 {pdfs[0]} 提取到的DOI: {doi}")


class TestLiteratureIndexer:
    """单元测试：文献库索引器"""

    def setup_method(self):
        """每个测试前创建临时索引器"""
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_registry.db")

        # 使用临时数据库路径
        import core.config
        self._original_db_path = core.config.Config.LITERATURE_REGISTRY_DB_PATH
        core.config.Config.LITERATURE_REGISTRY_DB_PATH = self.db_path
        self.indexer = LiteratureIndexer()

    def teardown_method(self):
        """清理临时文件"""
        import core.config
        core.config.Config.LITERATURE_REGISTRY_DB_PATH = self._original_db_path
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_init_creates_db(self):
        """测试初始化创建数据库文件"""
        assert os.path.exists(self.db_path)

    def test_init_creates_table(self):
        """测试初始化创建注册表"""
        conn = self.indexer._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='literature_registry'")
        assert cursor.fetchone() is not None
        conn.close()

    def test_upsert_and_lookup(self):
        """测试插入和查询"""
        metadata = PDFMetadata(
            title="测试论文标题",
            authors="张三, 李四",
            abstract_summary="测试摘要",
            innovation_points=["创新点A", "创新点B"],
            key_image=ImageBBox(page=1, x1=10, y1=20, x2=100, y2=200, description="架构图")
        )

        self.indexer.upsert_record(
            unique_id="test_id_001",
            metadata=metadata,
            current_filename="测试论文标题.pdf",
            file_hash="abc123",
            file_mtime=1234567890.0
        )

        # 查询验证
        record = self.indexer.lookup_by_id("test_id_001")
        assert record is not None
        assert record['title'] == "测试论文标题"
        assert record['authors'] == "张三, 李四"
        assert len(record['innovation_points']) == 2
        assert record['key_image_page'] == 1
        assert record['current_filename'] == "测试论文标题.pdf"

    def test_delete_record(self):
        """测试删除记录"""
        metadata = PDFMetadata(title="待删除论文")
        self.indexer.upsert_record(
            unique_id="test_id_del",
            metadata=metadata,
            current_filename="待删除论文.pdf",
            file_hash="hash",
            file_mtime=0.0
        )
        assert self.indexer.lookup_by_id("test_id_del") is not None

        self.indexer.delete_record("test_id_del")
        assert self.indexer.lookup_by_id("test_id_del") is None

    def test_query_registry_pagination(self):
        """测试分页查询"""
        # 清空之前测试可能残留的数据
        conn = self.indexer._get_conn()
        conn.execute("DELETE FROM literature_registry")
        conn.commit()
        conn.close()

        for i in range(3):
            metadata = PDFMetadata(title=f"论文{i}")
            self.indexer.upsert_record(
                unique_id=f"id_{i}",
                metadata=metadata,
                current_filename=f"论文{i}.pdf",
                file_hash=f"hash{i}",
                file_mtime=float(i)
            )

        result = self.indexer.query_registry(page=1, limit=2)
        assert result['total'] == 3
        assert len(result['entries']) == 2

        result = self.indexer.query_registry(page=2, limit=2)
        assert len(result['entries']) == 1

    def test_get_detail(self):
        """测试获取详情（含key_image重建）"""
        metadata = PDFMetadata(
            title="详情测试",
            key_image=ImageBBox(page=2, x1=5, y1=5, x2=50, y2=50, description="Fig 2")
        )
        self.indexer.upsert_record(
            unique_id="detail_test",
            metadata=metadata,
            current_filename="详情测试.pdf",
            file_hash="hash",
            file_mtime=1.0
        )

        detail = self.indexer.get_detail("detail_test")
        assert detail is not None
        assert detail['key_image'] is not None
        assert detail['key_image']['page'] == 2
        assert detail['key_image']['description'] == "Fig 2"


class TestBatchProcessor:
    """单元测试：并发批处理器"""

    def test_process_all_basic(self):
        """测试基本批处理功能"""
        from utils.batch_processor import BatchProcessor

        bp = BatchProcessor(max_workers=2, retry_attempts=0)

        def dummy_process(path):
            if 'skip' in path:
                return {"status": "skipped"}
            elif 'fail' in path:
                return {"status": "failed", "error": "模拟失败"}
            else:
                return {"status": "done"}

        paths = ["a.pdf", "b_skip.pdf", "c.pdf", "d_fail.pdf"]
        result = bp.process_all(paths, dummy_process)

        assert result['total'] == 4
        assert result['extracted'] == 2
        assert result['skipped'] == 1
        assert result['failed'] == 1
        assert len(result['errors']) == 1

    def test_retry_on_exception(self):
        """测试异常重试"""
        from utils.batch_processor import BatchProcessor

        bp = BatchProcessor(max_workers=1, retry_attempts=2)
        call_count = [0]

        def fail_then_succeed(path):
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("模拟临时失败")
            return {"status": "done"}

        result = bp.process_all(["test.pdf"], fail_then_succeed)
        assert result['extracted'] == 1
        assert call_count[0] == 3  # 2次失败 + 1次成功


if __name__ == '__main__':
    print("=" * 60)
    print("运行文献索引器测试套件")
    print("=" * 60)

    # --- PDFMetadataExtractor 测试 ---
    print("\n[PDFMetadataExtractor 测试]")
    tester = TestPDFMetadataExtractor()

    print("  测试 ImageBBox 模型...", end=" ")
    tester.test_image_bbox_model()
    print("OK")

    print("  测试 PDFMetadata 模型...", end=" ")
    tester.test_pdf_metadata_model()
    print("OK")

    print("  测试文件名清理...", end=" ")
    tester.test_sanitize_title_for_filename()
    print("OK")

    print("  测试ID清理...", end=" ")
    tester.test_sanitize_id()
    print("OK")

    print("  测试唯一ID生成...", end=" ")
    tester.test_generate_unique_id_from_path()
    print("OK")

    # --- LiteratureIndexer 测试 ---
    print("\n[LiteratureIndexer 测试]")
    ti = TestLiteratureIndexer()

    print("  初始化测试环境...", end=" ")
    ti.setup_method()
    print("OK")

    print("  测试建表和数据库...", end=" ")
    ti.test_init_creates_db()
    ti.test_init_creates_table()
    print("OK")

    print("  测试增删查...", end=" ")
    ti.test_upsert_and_lookup()
    print("OK")
    ti.test_delete_record()
    print("OK")

    print("  测试分页...", end=" ")
    ti.test_query_registry_pagination()
    print("OK")

    print("  测试详情...", end=" ")
    ti.test_get_detail()
    print("OK")

    print("  清理测试环境...", end=" ")
    ti.teardown_method()
    print("OK")

    # --- BatchProcessor 测试 ---
    print("\n[BatchProcessor 测试]")
    tbp = TestBatchProcessor()

    print("  测试基本批处理...", end=" ")
    tbp.test_process_all_basic()
    print("OK")

    print("  测试重试机制...", end=" ")
    tbp.test_retry_on_exception()
    print("OK")

    # --- 真实PDF测试 ---
    print("\n[真实PDF测试]")
    tester.test_extract_doi_from_real_pdf()

    print("\n" + "=" * 60)
    print("全部测试通过！")
    print("=" * 60)
