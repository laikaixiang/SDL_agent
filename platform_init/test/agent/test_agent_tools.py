"""
Agent tools — 单元测试

运行方法: python platform_init/test/agent/test_agent_tools.py
"""
import sys
import io
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from core.agent_tools import (
    AgentTool,
    UnifiedToolExecutor,
    scan_hardware_tools,
    scan_software_algorithms,
    create_main_executor,
    _params_to_json_schema,
    _param_to_json_schema,
    BUILTIN_TOOLS,
)


# =============================================================================
def test_import():
    """验证所有核心符号可导入"""
    print("\n=== test_import ===")
    assert AgentTool is not None
    assert UnifiedToolExecutor is not None
    assert callable(scan_hardware_tools)
    assert callable(scan_software_algorithms)
    assert callable(create_main_executor)
    print("PASS")


def test_builtin_tools_present():
    """BUILTIN_TOOLS 至少包含 ask_user, search_literature"""
    print("\n=== test_builtin_tools_present ===")
    names = {t.name for t in BUILTIN_TOOLS}
    assert "ask_user" in names, f"Missing ask_user in {names}"
    assert "search_literature" in names, f"Missing search_literature in {names}"
    assert "design_experiment" in names, f"Missing design_experiment in {names}"
    assert "generate_algorithm" in names, f"Missing generate_algorithm in {names}"
    print(f"  Found {len(BUILTIN_TOOLS)} builtin tools: {names}")
    print("PASS")


def test_agent_tool_dataclass():
    """AgentTool dataclass 字段正确"""
    print("\n=== test_agent_tool_dataclass ===")
    t = AgentTool(
        name="test",
        description="a test tool",
        parameters={"type": "object", "properties": {}},
        required=[],
        func=lambda args: "ok",
        category="builtin",
        dangerous=False,
    )
    assert t.name == "test"
    assert t.category == "builtin"
    assert t.dangerous is False
    assert t.func({}) == "ok"
    print("PASS")


def test_params_to_json_schema():
    """_params_to_json_schema 正确转换 registry 格式"""
    print("\n=== test_params_to_json_schema ===")
    params = {
        "speed": {"type": "int", "description": "rpm", "required": True, "default": 3000},
        "name": {"type": "str", "description": "label"},
    }
    schema = _params_to_json_schema(params)
    assert schema["properties"]["speed"]["type"] == "integer"
    assert schema["properties"]["speed"]["default"] == 3000
    assert "speed" in schema.get("required", [])
    assert schema["properties"]["name"]["type"] == "string"
    assert "name" not in schema.get("required", [])
    print("PASS")


def test_scan_hardware_tools():
    """scan_hardware_tools 返回已注册的硬件工具"""
    print("\n=== test_scan_hardware_tools ===")
    tools = scan_hardware_tools()
    names = {t.name for t in tools}
    # 至少包含 spin_coating
    assert "spin_coating" in names, f"Missing spin_coating in {names}"
    # 硬件工具应标记为 dangerous
    for t in tools:
        assert t.dangerous is True, f"{t.name} should be dangerous=True"
        assert t.category == "hardware"
    print(f"  Found {len(tools)} hardware tools: {names}")
    print("PASS")


def test_scan_software_algorithms():
    """scan_software_algorithms 返回已注册的软件算法"""
    print("\n=== test_scan_software_algorithms ===")
    tools = scan_software_algorithms()
    names = {t.name for t in tools}
    assert "data_statistics" in names, f"Missing data_statistics in {names}"
    assert "data_normalization" in names, f"Missing data_normalization in {names}"
    for t in tools:
        assert t.category == "software"
    print(f"  Found {len(tools)} software algorithms: {names}")
    print("PASS")


def test_executor_build_openai_tools():
    """UnifiedToolExecutor.build_openai_tools() 返回正确格式"""
    print("\n=== test_executor_build_openai_tools ===")
    t = AgentTool(
        name="test_tool",
        description="desc",
        parameters={"type": "object", "properties": {"x": {"type": "integer", "description": "value"}}, "required": ["x"]},
        required=["x"],
        func=lambda args: str(args.get("x", 0)),
        category="builtin",
    )
    executor = UnifiedToolExecutor([t])
    schemas = executor.build_openai_tools()
    assert len(schemas) == 1
    func = schemas[0]["function"]
    assert func["name"] == "test_tool"
    assert func["description"] == "desc"
    assert "x" in func["parameters"]["properties"]
    print("PASS")


def test_executor_dispatch():
    """UnifiedToolExecutor.dispatch() 正确调用工具函数"""
    print("\n=== test_executor_dispatch ===")
    t = AgentTool(
        name="add",
        description="add two numbers",
        parameters={"type": "object", "properties": {}},
        required=[],
        func=lambda args: str(int(args.get("a", 0)) + int(args.get("b", 0))),
        category="builtin",
    )
    executor = UnifiedToolExecutor([t])
    result = executor.dispatch("add", {"a": 3, "b": 4})
    assert result == "7", f"Expected '7', got {result}"
    print("PASS")


def test_executor_dispatch_unknown():
    """dispatch 未知工具返回错误字符串"""
    print("\n=== test_executor_dispatch_unknown ===")
    executor = UnifiedToolExecutor([])
    result = executor.dispatch("nonexistent", {})
    assert "错误" in result or "未找到" in result
    print("PASS")


def test_executor_get():
    """UnifiedToolExecutor.get() 按名查找"""
    print("\n=== test_executor_get ===")
    t = AgentTool(name="a", description="", parameters={}, required=[], func=lambda a: "", category="builtin")
    executor = UnifiedToolExecutor([t])
    assert executor.get("a") is not None
    assert executor.get("b") is None
    print("PASS")


def test_executor_is_hardware_tool():
    """is_hardware_tool 正确区分硬件和软件工具"""
    print("\n=== test_executor_is_hardware_tool ===")
    hw = AgentTool(name="hw", description="", parameters={}, required=[], func=lambda a: "", category="hardware", dangerous=True)
    sw = AgentTool(name="sw", description="", parameters={}, required=[], func=lambda a: "", category="software")
    executor = UnifiedToolExecutor([hw, sw])
    assert executor.is_hardware_tool("hw") is True
    assert executor.is_hardware_tool("sw") is False
    assert executor.is_hardware_tool("nonexistent") is False
    print("PASS")


def test_executor_names_property():
    """executor.names 返回所有已注册工具名"""
    print("\n=== test_executor_names_property ===")
    t1 = AgentTool(name="a", description="", parameters={}, required=[], func=lambda a: "", category="builtin")
    t2 = AgentTool(name="b", description="", parameters={}, required=[], func=lambda a: "", category="builtin")
    executor = UnifiedToolExecutor([t1, t2])
    names = executor.names
    assert "a" in names
    assert "b" in names
    assert len(names) == 2
    print("PASS")


def test_create_main_executor():
    """create_main_executor 返回可用的 executor"""
    print("\n=== test_create_main_executor ===")
    executor = create_main_executor()
    assert executor is not None
    names = executor.names
    assert len(names) >= 4, f"Expected at least 4 tools, got {len(names)}"
    # 关键业务工具必须存在
    assert "ask_user" in names
    assert "search_literature" in names
    assert "spin_coating" in names
    print(f"  Total tools: {len(names)}")
    schemas = executor.build_openai_tools()
    assert len(schemas) == len(names)
    print("PASS")


def test_ask_user_is_noop():
    """ask_user 工具返回占位符（由 AgentLoop 拦截）"""
    print("\n=== test_ask_user_is_noop ===")
    for t in BUILTIN_TOOLS:
        if t.name == "ask_user":
            result = t.func({"question": "test"})
            assert result == "__ASK_USER_PENDING__", f"Expected placeholder, got {result}"
            break
    else:
        assert False, "ask_user not found in BUILTIN_TOOLS"
    print("PASS")


def test_extraction_tools():
    """Phase 2: extract_from_pdf 和 preview_pdf_page 已注册"""
    print("\n=== test_extraction_tools ===")
    names = {t.name for t in BUILTIN_TOOLS}
    assert "extract_from_pdf" in names, f"Missing extract_from_pdf in {names}"
    assert "preview_pdf_page" in names, f"Missing preview_pdf_page in {names}"
    print(f"  Extraction tools present: extract_from_pdf, preview_pdf_page")
    print("PASS")


def test_extraction_tool_category():
    """Phase 2 提取工具的 category 为 'extraction'"""
    print("\n=== test_extraction_tool_category ===")
    for t in BUILTIN_TOOLS:
        if t.name in ("extract_from_pdf", "preview_pdf_page"):
            assert t.category == "extraction", f"{t.name} category should be 'extraction', got '{t.category}'"
    print("PASS")


# =============================================================================
if __name__ == "__main__":
    passed = 0
    failed = 0
    tests = [
        test_import,
        test_builtin_tools_present,
        test_agent_tool_dataclass,
        test_params_to_json_schema,
        test_scan_hardware_tools,
        test_scan_software_algorithms,
        test_executor_build_openai_tools,
        test_executor_dispatch,
        test_executor_dispatch_unknown,
        test_executor_get,
        test_executor_is_hardware_tool,
        test_executor_names_property,
        test_create_main_executor,
        test_ask_user_is_noop,
        test_extraction_tools,
        test_extraction_tool_category,
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
