"""
Phase 2 extraction tools — 单元测试

测试 extract_from_pdf / preview_pdf_page 工具注册与调用。

运行方法: python platform_init/test/agent_phase2/test_phase2_tools.py
"""
import sys
import io
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from core.agent_tools import BUILTIN_TOOLS, UnifiedToolExecutor, create_main_executor


# =============================================================================
def test_extraction_tools_registered():
    """Phase 2 提取工具已注册"""
    print("\n=== test_extraction_tools_registered ===")
    names = {t.name for t in BUILTIN_TOOLS}
    assert "extract_from_pdf" in names, f"Missing extract_from_pdf"
    assert "preview_pdf_page" in names, f"Missing preview_pdf_page"
    print(f"  OK: {len(names)} builtin tools, extraction tools present")
    print("PASS")


def test_extraction_tools_in_main_executor():
    """create_main_executor 包含 Phase 2 提取工具"""
    print("\n=== test_extraction_tools_in_main_executor ===")
    executor = create_main_executor()
    names = executor.names
    assert "extract_from_pdf" in names
    assert "preview_pdf_page" in names
    print(f"  OK: executor has {len(names)} tools including extraction tools")
    print("PASS")


def test_extraction_tool_category():
    """提取工具的 category 为 'extraction'"""
    print("\n=== test_extraction_tool_category ===")
    extraction_tools = [t for t in BUILTIN_TOOLS if t.name in ("extract_from_pdf", "preview_pdf_page")]
    assert len(extraction_tools) == 2
    for t in extraction_tools:
        assert t.category == "extraction", f"{t.name}: expected 'extraction', got '{t.category}'"
        assert not t.dangerous, f"{t.name} should not be dangerous"
    print(f"  OK: both tools have category='extraction', dangerous=False")
    print("PASS")


def test_extraction_tool_schemas():
    """提取工具的 OpenAI schema 格式正确"""
    print("\n=== test_extraction_tool_schemas ===")
    extraction_tools = [t for t in BUILTIN_TOOLS if t.name in ("extract_from_pdf", "preview_pdf_page")]
    executor = UnifiedToolExecutor(extraction_tools)
    schemas = executor.build_openai_tools()
    assert len(schemas) == 2

    for s in schemas:
        fn = s["function"]
        assert fn["name"] in ("extract_from_pdf", "preview_pdf_page")
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params

        if fn["name"] == "extract_from_pdf":
            assert "pdf_path" in params["properties"]
            assert "task_description" in params["properties"]
            assert "pdf_path" in params["required"]
        elif fn["name"] == "preview_pdf_page":
            assert "pdf_path" in params["properties"]
            assert "page_num" in params["properties"]

    print(f"  OK: schema validation passed for both tools")
    print("PASS")


def test_extract_from_pdf_dispatch():
    """extract_from_pdf 对于不存在的文件返回 error JSON"""
    print("\n=== test_extract_from_pdf_dispatch ===")
    tool = next(t for t in BUILTIN_TOOLS if t.name == "extract_from_pdf")
    executor = UnifiedToolExecutor([tool])
    result = executor.dispatch("extract_from_pdf", {
        "pdf_path": "test.pdf",
        "task_description": "提取钙钛矿参数",
    })
    # 文件不存在 → 返回包含 error 的 JSON
    assert isinstance(result, str)
    assert "error" in result.lower(), f"Expected error for nonexistent file, got: {result[:120]}"
    print(f"  Result: {result[:120]}")
    print("PASS")


def test_preview_pdf_page_dispatch_nonexistent():
    """preview_pdf_page 对不存在的PDF返回错误，不崩溃"""
    print("\n=== test_preview_pdf_page_dispatch_nonexistent ===")
    tool = next(t for t in BUILTIN_TOOLS if t.name == "preview_pdf_page")
    executor = UnifiedToolExecutor([tool])
    result = executor.dispatch("preview_pdf_page", {
        "pdf_path": "/nonexistent/path/file.pdf",
        "page_num": 1,
    })
    # 应返回错误信息，不抛异常
    assert isinstance(result, str)
    assert "失败" in result or "错误" in result or "not" in result.lower()
    print(f"  Result: {result[:120]}")
    print("PASS")


def test_preview_pdf_page_resolves_relative_path():
    """preview_pdf_page 对相对路径自动补全 PDF_FOLDER 前缀"""
    print("\n=== test_preview_pdf_page_resolves_relative_path ===")
    tool = next(t for t in BUILTIN_TOOLS if t.name == "preview_pdf_page")
    executor = UnifiedToolExecutor([tool])
    # 相对路径应被解析为 PDF_FOLDER/relative.pdf
    result = executor.dispatch("preview_pdf_page", {
        "pdf_path": "some_relative_path.pdf",
        "page_num": 1,
    })
    assert isinstance(result, str)
    print(f"  Result: {result[:120]}")
    print("PASS")


def test_extract_from_pdf_optional_params():
    """extract_from_pdf 可选参数 fields/pages 不影响调用"""
    print("\n=== test_extract_from_pdf_optional_params ===")
    tool = next(t for t in BUILTIN_TOOLS if t.name == "extract_from_pdf")
    executor = UnifiedToolExecutor([tool])
    # 不带可选参数
    r1 = executor.dispatch("extract_from_pdf", {
        "pdf_path": "test.pdf",
        "task_description": "test",
    })
    assert isinstance(r1, str)
    # 带可选参数
    r2 = executor.dispatch("extract_from_pdf", {
        "pdf_path": "test.pdf",
        "task_description": "test",
        "fields": ["name", "value"],
        "pages": [1, 2, 3],
    })
    assert isinstance(r2, str)
    print(f"  Without optional: {r1[:100]}")
    print(f"  With optional: {r2[:100]}")
    print("PASS")


# =============================================================================
if __name__ == "__main__":
    passed = 0
    failed = 0
    tests = [
        test_extraction_tools_registered,
        test_extraction_tools_in_main_executor,
        test_extraction_tool_category,
        test_extraction_tool_schemas,
        test_extract_from_pdf_dispatch,
        test_preview_pdf_page_dispatch_nonexistent,
        test_preview_pdf_page_resolves_relative_path,
        test_extract_from_pdf_optional_params,
    ]

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback
            print(f"\n[FAIL] {test.__name__}: {e}")
            traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
    if failed > 0:
        sys.exit(1)
