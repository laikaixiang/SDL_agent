"""
针对 PDF 工具路径解析的回归测试

覆盖三个 bug:
- preview_pdf_page: get_pdf_info 返回 None 时,info.get() 触发 NoneType 错误
- extract_from_pdf: 'PDF_TARGET/foo.pdf' 形式路径,会被双重拼接成
  'PDF_TARGET\\PDF_TARGET/foo.pdf'
- 用户传不存在的 PDF 时,应返回清晰的错误信息(列出可用的文件),
  而不是崩溃

运行: python platform_init/test/agent_system/test_pdf_path_resolution.py
"""
import sys, io, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
from unittest.mock import patch, MagicMock

from core.agent_tools import (
    _resolve_pdf_path,
    _list_available_pdfs,
    _preview_pdf_page_func,
    _extract_from_pdf_func,
)
from core.config import Config


# =============================================================================
# _resolve_pdf_path 路径解析测试
# =============================================================================

def test_resolve_absolute_path():
    """绝对路径应原样返回"""
    print("\n=== test_resolve_absolute_path ===")
    cfg = Config()
    assert _resolve_pdf_path("/abs/path/foo.pdf", cfg) == "/abs/path/foo.pdf"
    assert _resolve_pdf_path("C:\\Users\\foo.pdf", cfg) == "C:\\Users\\foo.pdf"
    print("PASS")


def test_resolve_prefix_strip():
    """'PDF_TARGET/foo.pdf' 形式应去掉 PDF_TARGET/ 前缀,避免双重拼接"""
    print("\n=== test_resolve_prefix_strip ===")
    cfg = Config()
    # 关键 bug: 不修复时会得到 'dialogue data/PDF_TARGET\\PDF_TARGET/foo.pdf'
    resolved = _resolve_pdf_path("PDF_TARGET/10.1002-anie.202005211.pdf", cfg)
    # 必须以单层 PDF_FOLDER 结尾,而非双层
    norm_resolved = os.path.normpath(resolved)
    norm_folder = os.path.normpath(cfg.PDF_FOLDER)
    assert norm_resolved.startswith(norm_folder + os.sep), \
        f"Should start with PDF_FOLDER, got: {resolved} (norm={norm_resolved})"
    norm_resolved_str = norm_resolved.replace(os.sep, "/")
    assert "PDF_TARGET/PDF_TARGET" not in norm_resolved_str, \
        f"Double-prefix bug present: {resolved}"
    assert norm_resolved_str.endswith("10.1002-anie.202005211.pdf")
    print(f"  resolved to: {resolved}")
    print("PASS")


def test_resolve_bare_filename():
    """仅文件名应拼到 PDF_FOLDER 下"""
    print("\n=== test_resolve_bare_filename ===")
    cfg = Config()
    resolved = _resolve_pdf_path("foo.pdf", cfg)
    expected = os.path.normpath(os.path.join(cfg.PDF_FOLDER, "foo.pdf"))
    assert os.path.normpath(resolved) == expected, f"got: {resolved}, expected: {expected}"
    print("PASS")


def test_resolve_registry_fallback():
    """literature_registry 兜底:PDF 不在原路径但注册表里有 current_filename"""
    print("\n=== test_resolve_registry_fallback ===")
    cfg = Config()
    with patch("os.path.isfile", side_effect=lambda p: "registry_hit" in p or p.endswith("literature_registry.db")):
        with patch("sqlite3.connect") as fake_sqlite:
            fake_conn = MagicMock()
            fake_conn.__enter__.return_value.execute.return_value.fetchone.return_value = (
                "registry_hit.pdf",
            )
            fake_sqlite.return_value = fake_conn
            # 输入文件名与 current_filename 不同,触发兜底
            resolved = _resolve_pdf_path("different_name.pdf", cfg)
            # 应该返回 PDF_FOLDER/registry_hit.pdf
            expected = os.path.normpath(os.path.join(cfg.PDF_FOLDER, "registry_hit.pdf"))
            assert os.path.normpath(resolved) == expected, \
                f"got: {resolved}, expected: {expected}"
    print("PASS")


def test_resolve_empty_path():
    """空路径应原样返回"""
    print("\n=== test_resolve_empty_path ===")
    cfg = Config()
    assert _resolve_pdf_path("", cfg) == ""
    print("PASS")


def test_resolve_windows_prefix_strip():
    """Windows 反斜杠前缀也应正确剥离"""
    print("\n=== test_resolve_windows_prefix_strip ===")
    cfg = Config()
    resolved = _resolve_pdf_path("PDF_TARGET\\foo.pdf", cfg)
    norm_resolved = os.path.normpath(resolved).replace(os.sep, "/")
    norm_folder = os.path.normpath(cfg.PDF_FOLDER).replace(os.sep, "/")
    assert "PDF_TARGET/PDF_TARGET" not in norm_resolved
    assert norm_resolved.startswith(norm_folder + "/"), \
        f"Should start with {norm_folder}, got: {resolved}"
    print(f"  resolved to: {resolved}")
    print("PASS")


# =============================================================================
# _list_available_pdfs 测试
# =============================================================================

def test_list_pdfs_with_files(tmp_dir=None):
    """有文件时列出前 10 个"""
    print("\n=== test_list_pdfs_with_files ===")
    cfg = Config()
    available = _list_available_pdfs(cfg)
    assert "可用的文件" not in available
    # 实际 PDF_FOLDER 至少有 1 个文件(看 lit 索引)
    assert "(无文件)" not in available
    assert "(无法列举)" not in available
    # 至少 1 个 .pdf
    assert ".pdf" in available
    print(f"  {available[:200]}")
    print("PASS")


def test_list_pdfs_empty_folder():
    """空文件夹返回 '(无文件)'"""
    print("\n=== test_list_pdfs_empty_folder ===")
    cfg = Config()
    with patch("os.listdir", return_value=[]):
        result = _list_available_pdfs(cfg)
    assert result == "(无文件)", f"got: {result!r}"
    print("PASS")


def test_list_pdfs_oserror():
    """os.listdir 抛错时返回 '(无法列举)'"""
    print("\n=== test_list_pdfs_oserror ===")
    cfg = Config()
    with patch("os.listdir", side_effect=OSError("perm denied")):
        result = _list_available_pdfs(cfg)
    assert result == "(无法列举)", f"got: {result!r}"
    print("PASS")


# =============================================================================
# _preview_pdf_page_func 错误信息测试
# =============================================================================

def test_preview_pdf_missing_file():
    """PDF 不存在时应返回清晰错误(不再 NoneType)"""
    print("\n=== test_preview_pdf_missing_file ===")
    out = _preview_pdf_page_func({
        "pdf_path": "definitely_does_not_exist_12345.pdf",
        "page_num": 1,
    })
    print(f"  output: {out[:200]}")
    assert "'NoneType' object has no attribute" not in out, \
        f"REGRESSION: still has NoneType error"
    assert "PDF 文件不存在" in out
    assert "可用的文件" in out
    print("PASS")


def test_preview_pdf_get_info_returns_none():
    """get_pdf_info 返回 None 时(无 NoneType 错误)"""
    print("\n=== test_preview_pdf_get_info_returns_none ===")
    # 找一个真实存在的 PDF
    cfg = Config()
    real_pdf = None
    for f in os.listdir(cfg.PDF_FOLDER):
        if f.lower().endswith(".pdf"):
            real_pdf = f
            break
    if not real_pdf:
        print("SKIP: no PDFs in PDF_FOLDER")
        return
    with patch("core.extract_manager.PDFProcessor") as MockProcessor:
        mock_instance = MockProcessor.return_value
        mock_instance.get_pdf_info = MagicMock(return_value=None)
        out = _preview_pdf_page_func({"pdf_path": real_pdf, "page_num": 1})
    print(f"  output: {out[:200]}")
    assert "'NoneType'" not in out, f"NoneType error: {out}"
    assert "无法读取 PDF" in out, f"Expected clear error, got: {out[:200]}"
    print("PASS")


# =============================================================================
# _extract_from_pdf_func 路径测试
# =============================================================================

def test_extract_pdf_with_double_prefix():
    """用户传 'PDF_TARGET/foo.pdf' 时不应双重拼接"""
    print("\n=== test_extract_pdf_with_double_prefix ===")
    out_str = _extract_from_pdf_func({
        "pdf_path": "PDF_TARGET/10.1002-anie.202005211.pdf",
        "task_description": "提取实验参数",
    })
    out = json.loads(out_str)
    print(f"  error: {out.get('error', '(none)')[:200]}")
    assert "error" in out
    err = out["error"]
    # 不应出现双层 PDF_TARGET/PDF_TARGET (无论正反斜杠)
    norm_err = err.replace(os.sep, "/").replace("\\", "/")
    assert "PDF_TARGET/PDF_TARGET" not in norm_err
    # 应列出可用文件
    assert "可用的文件" in err
    print("PASS")


# =============================================================================
if __name__ == "__main__":
    passed = failed = 0
    tests = [
        test_resolve_absolute_path,
        test_resolve_prefix_strip,
        test_resolve_bare_filename,
        test_resolve_registry_fallback,
        test_resolve_empty_path,
        test_resolve_windows_prefix_strip,
        test_list_pdfs_with_files,
        test_list_pdfs_empty_folder,
        test_list_pdfs_oserror,
        test_preview_pdf_missing_file,
        test_preview_pdf_get_info_returns_none,
        test_extract_pdf_with_double_prefix,
    ]
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback
            print(f"\n[FAIL] {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
    sys.exit(1 if failed else 0)
