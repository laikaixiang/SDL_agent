"""
Agent 工具注册 — 系统测试

运行方法: python platform_init/test/agent_system/test_tool_registry.py
"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from core.agent_tools import (
    AgentTool, UnifiedToolExecutor, BUILTIN_TOOLS,
    scan_hardware_tools, scan_software_algorithms, create_main_executor,
    _params_to_json_schema,
)
from hardware import ToolRegistry

executor = create_main_executor()
HW_NAMES = {t.name for t in scan_hardware_tools()}
SW_NAMES = {t.name for t in scan_software_algorithms()}
ALL_NAMES = set(executor.names)


# =============================================================================
def test_total_tool_count():
    """总计 >= 22 个工具（6 builtin + 10 hardware + 6 software）"""
    print("\n=== test_total_tool_count ===")
    assert len(ALL_NAMES) >= 22, f"Expected >=22, got {len(ALL_NAMES)}"
    print(f"  Total: {len(ALL_NAMES)} tools")
    print("PASS")


def test_builtin_tools_complete():
    """6 个 builtin 工具全部注册"""
    print("\n=== test_builtin_tools_complete ===")
    required = {"ask_user", "search_literature", "design_experiment",
                "generate_algorithm", "extract_from_pdf", "preview_pdf_page"}
    names = {t.name for t in BUILTIN_TOOLS}
    assert required <= names, f"Missing: {required - names}"
    print(f"  Builtin: {sorted(names)}")
    print("PASS")


def test_hardware_tools_complete():
    """10 个硬件工具 + 全部标记为 dangerous"""
    print("\n=== test_hardware_tools_complete ===")
    assert "spin_coating" in HW_NAMES
    assert "set_temperature" in HW_NAMES
    assert "collect_spectrum" in HW_NAMES
    assert len(HW_NAMES) >= 10, f"Expected >=10, got {len(HW_NAMES)}"
    for t in scan_hardware_tools():
        assert t.dangerous, f"{t.name} should be dangerous"
        assert t.category == "hardware"
    print(f"  Hardware: {sorted(HW_NAMES)} (all dangerous)")
    print("PASS")


def test_software_algos_complete():
    """6 个软件算法可被扫描"""
    print("\n=== test_software_algos_complete ===")
    assert "data_statistics" in SW_NAMES
    assert "data_normalization" in SW_NAMES
    assert "spectrum_analysis" in SW_NAMES
    assert "bayesian_optimization" in SW_NAMES
    for t in scan_software_algorithms():
        assert t.category == "software"
    print(f"  Software: {sorted(SW_NAMES)}")
    print("PASS")


def test_all_tools_have_valid_schema():
    """所有 22 个工具的 OpenAI schema 格式正确"""
    print("\n=== test_all_tools_have_valid_schema ===")
    schemas = executor.build_openai_tools()
    assert len(schemas) == len(ALL_NAMES)
    for s in schemas:
        fn = s["function"]
        assert "type" not in s or s["type"] == "function"
        assert "name" in fn, f"Missing name in {fn}"
        assert "description" in fn, f"Missing description in {fn}"
        assert "parameters" in fn, f"Missing parameters in {fn}"
        params = fn["parameters"]
        assert params.get("type") == "object", f"{fn['name']}: params.type is not object"
        assert "properties" in params, f"{fn['name']}: missing properties"
    print(f"  All {len(schemas)} tools have valid OpenAI schema")
    print("PASS")


def test_dispatch_valid_tool():
    """已知工具 dispatch 正常工作"""
    print("\n=== test_dispatch_valid_tool ===")
    r = executor.dispatch("search_literature", {"query": "test"})
    assert isinstance(r, str) and len(r) > 0
    print(f"  search_literature result: {r[:80]}...")
    print("PASS")


def test_dispatch_unknown_graceful():
    """dispatch 不存在的工具返回错误不崩溃"""
    print("\n=== test_dispatch_unknown_graceful ===")
    r = executor.dispatch("nonexistent_tool_xyz", {})
    assert isinstance(r, str)
    assert "错误" in r or "未找到" in r
    print(f"  Unknown tool: {r[:80]}")
    print("PASS")


def test_build_openai_tools_is_deterministic():
    """连续两次 build_openai_tools() 返回相同结果"""
    print("\n=== test_build_openai_tools_is_deterministic ===")
    s1 = executor.build_openai_tools()
    s2 = executor.build_openai_tools()
    assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)
    print("PASS")


def test_builtin_category_mapping():
    """builtin 工具 category 正确"""
    print("\n=== test_builtin_category_mapping ===")
    expected = {
        "ask_user": "builtin", "search_literature": "builtin",
        "design_experiment": "builtin", "generate_algorithm": "builtin",
        "extract_from_pdf": "extraction", "preview_pdf_page": "extraction",
    }
    for t in BUILTIN_TOOLS:
        assert t.category == expected.get(t.name, "builtin"), \
            f"{t.name}: expected {expected.get(t.name)}, got {t.category}"
    print("PASS")


def test_dangerous_flag_correct():
    """只有 hardware 工具标记为 dangerous"""
    print("\n=== test_dangerous_flag_correct ===")
    for t in BUILTIN_TOOLS:
        assert not t.dangerous, f"builtin {t.name} should not be dangerous"
    for t in scan_hardware_tools():
        assert t.dangerous, f"hardware {t.name} should be dangerous"
    for t in scan_software_algorithms():
        assert not t.dangerous, f"software {t.name} should not be dangerous"
    print("PASS")


def test_schema_required_fields():
    """schema.required 中的字段必须在 properties 中"""
    print("\n=== test_schema_required_fields ===")
    schemas = executor.build_openai_tools()
    for s in schemas:
        fn = s["function"]
        required = fn["parameters"].get("required", [])
        properties = fn["parameters"].get("properties", {})
        for field in required:
            assert field in properties, \
                f"{fn['name']}: required field '{field}' not in properties"
    print(f"  All schemas: required fields exist in properties")
    print("PASS")


# =============================================================================
if __name__ == "__main__":
    passed = failed = 0
    tests = [
        test_total_tool_count, test_builtin_tools_complete,
        test_hardware_tools_complete, test_software_algos_complete,
        test_all_tools_have_valid_schema, test_dispatch_valid_tool,
        test_dispatch_unknown_graceful, test_build_openai_tools_is_deterministic,
        test_builtin_category_mapping, test_dangerous_flag_correct,
        test_schema_required_fields,
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
