"""
StreamAdapter SSE 协议 — 系统测试

运行方法: python platform_init/test/agent_system/test_stream_protocol.py
"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from utils.stream_adapter import (
    StreamAdapter, make_event,
    THINKING_START, THINKING_DELTA, THINKING_END,
    TEXT_START, TEXT_DELTA, TEXT_END,
    TOOL_CALL_START, TOOL_CALL_ARGS, TOOL_CALL_END,
    DONE, ERROR,
)


def sse(delta: dict) -> str:
    payload = {"choices": [{"delta": delta}]}
    return f"data: {json.dumps(payload)}"


# =============================================================================
def test_thinking_then_text():
    """标准流程: reasoning → thinking_start/end → text_start/end → done"""
    print("\n=== test_thinking_then_text ===")
    a = StreamAdapter()
    events = list(a.adapt([
        sse({"reasoning_content": "hmm"}), sse({"reasoning_content": "..."}),
        sse({"content": "Hello"}), sse({"content": " world"}),
    ]))
    types = [e["type"] for e in events]
    assert THINKING_START in types
    assert THINKING_END in types
    assert TEXT_START in types
    assert TEXT_END in types
    assert DONE in types
    # 顺序: thinking_end 在 text_start 之前
    assert types.index(THINKING_END) < types.index(TEXT_START)
    te = [e for e in events if e["type"] == TEXT_END][0]
    assert te["text"] == "Hello world"
    print(f"  Types: {types}")
    print("PASS")


def test_text_only():
    """无 reasoning 模型直接输出文本"""
    print("\n=== test_text_only ===")
    a = StreamAdapter()
    events = list(a.adapt([sse({"content": "direct"})]))
    types = [e["type"] for e in events]
    assert THINKING_START not in types
    assert TEXT_START in types
    assert TEXT_END in types
    print("PASS")


def test_tool_call_full_cycle():
    """完整 tool_call: START → ARGS* → END"""
    print("\n=== test_tool_call_full_cycle ===")
    a = StreamAdapter()
    events = list(a.adapt([
        sse({"tool_calls": [{"index": 0, "id": "c1", "type": "function",
              "function": {"name": "search", "arguments": ""}}]}),
        sse({"tool_calls": [{"index": 0, "function": {"arguments": "{"}}]}),
        sse({"tool_calls": [{"index": 0, "function": {"arguments": '"q"'}}]}),
        sse({"tool_calls": [{"index": 0, "function": {"arguments": ': '}}]}),
        sse({"tool_calls": [{"index": 0, "function": {"arguments": '"x"}'}}]}),
    ]))
    types = [e["type"] for e in events]
    assert TOOL_CALL_START in types
    assert TOOL_CALL_ARGS in types
    assert TOOL_CALL_END in types
    end = [e for e in events if e["type"] == TOOL_CALL_END][0]
    assert end["name"] == "search"
    assert end["arguments"] == {"q": "x"}
    assert "call_id" in end
    print(f"  Parsed args: {end['arguments']}")
    print("PASS")


def test_multiple_tools_different_indices():
    """不同 index 的两个工具同时到达"""
    print("\n=== test_multiple_tools_different_indices ===")
    a = StreamAdapter()
    events = list(a.adapt([
        sse({"tool_calls": [
            {"index": 0, "id": "c1", "type": "function", "function": {"name": "a", "arguments": '{"x":1}'}},
            {"index": 1, "id": "c2", "type": "function", "function": {"name": "b", "arguments": '{"y":2}'}},
        ]}),
    ]))
    starts = [e for e in events if e["type"] == TOOL_CALL_START]
    ends = [e for e in events if e["type"] == TOOL_CALL_END]
    assert len(starts) == 2
    assert len(ends) == 2
    names = {e["name"] for e in ends}
    assert names == {"a", "b"}
    print(f"  Both tools parsed: {names}")
    print("PASS")


def test_thinking_flushed_before_tool_call():
    """tool_call 到达时先 flush thinking"""
    print("\n=== test_thinking_flushed_before_tool_call ===")
    a = StreamAdapter()
    events = list(a.adapt([
        sse({"reasoning_content": "think"}),
        sse({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "t", "arguments": "{}"}}]}),
    ]))
    types = [e["type"] for e in events]
    te_idx = types.index(THINKING_END)
    ts_idx = types.index(TOOL_CALL_START)
    assert te_idx < ts_idx, "Thinking must end before tool_call starts"
    print("PASS")


def test_text_flushed_before_tool_call():
    """tool_call 到达时先 flush 正在输出的文本"""
    print("\n=== test_text_flushed_before_tool_call ===")
    a = StreamAdapter()
    events = list(a.adapt([
        sse({"content": "partial"}),
        sse({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "t", "arguments": "{}"}}]}),
    ]))
    types = [e["type"] for e in events]
    assert TEXT_END in types
    te_idx = types.index(TEXT_END)
    ts_idx = types.index(TOOL_CALL_START)
    assert te_idx < ts_idx, "Text must end before tool_call starts"
    print("PASS")


def test_state_reset_across_streams():
    """第二次调用 adapt() 时状态清零"""
    print("\n=== test_state_reset_across_streams ===")
    a = StreamAdapter()
    list(a.adapt([sse({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "x", "arguments": "{}"}}]})]))
    assert len(a.get_pending_tool_calls()) == 1
    # 第二次流
    list(a.adapt([sse({"content": "clean"})]))
    pending = a.get_pending_tool_calls()
    no_started = all(not s.get("started") for s in pending)
    assert len(pending) == 0 or no_started, f"Stale state: {pending}"
    print("PASS")


def test_null_function_safe():
    """function: null 不崩溃"""
    print("\n=== test_null_function_safe ===")
    a = StreamAdapter()
    events = list(a.adapt([
        sse({"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": None}]}),
        sse({"tool_calls": [{"index": 0, "function": {"name": "ok", "arguments": "{}"}}]}),
    ]))
    types = [e["type"] for e in events]
    assert ERROR not in types
    assert TOOL_CALL_END in types
    print("PASS")


def test_non_dict_entries_safe():
    """tool_calls 列表含 None → 安全跳过"""
    print("\n=== test_non_dict_entries_safe ===")
    a = StreamAdapter()
    events = list(a.adapt([
        sse({"tool_calls": [None, "string", 123]}),
        sse({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "valid", "arguments": "{}"}}]}),
    ]))
    assert ERROR not in [e["type"] for e in events]
    print("PASS")


def test_json_parse_fallback():
    """非法 JSON args_buf → _raw 回退"""
    print("\n=== test_json_parse_fallback ===")
    a = StreamAdapter()
    events = list(a.adapt([
        sse({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "bad", "arguments": "not { valid"}}]}),
    ]))
    end = [e for e in events if e["type"] == TOOL_CALL_END][0]
    assert "_raw" in end["arguments"]
    print(f"  Fallback: {end['arguments']}")
    print("PASS")


def test_make_event_utility():
    """make_event 工具函数"""
    print("\n=== test_make_event_utility ===")
    e1 = make_event("custom", "hello", key="val")
    assert e1["type"] == "custom"
    assert e1["text"] == "hello"
    assert e1["key"] == "val"
    e2 = make_event("no_text")
    assert "text" not in e2
    print("PASS")


# =============================================================================
if __name__ == "__main__":
    passed = failed = 0
    tests = [
        test_thinking_then_text, test_text_only,
        test_tool_call_full_cycle, test_multiple_tools_different_indices,
        test_thinking_flushed_before_tool_call, test_text_flushed_before_tool_call,
        test_state_reset_across_streams, test_null_function_safe,
        test_non_dict_entries_safe, test_json_parse_fallback,
        test_make_event_utility,
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
