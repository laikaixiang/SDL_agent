"""
StreamAdapter tool_call delta — 单元测试

运行方法: python platform_init/test/agent/test_stream_adapter.py
"""
import sys
import io
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from utils.stream_adapter import (
    StreamAdapter,
    TOOL_CALL_START,
    TOOL_CALL_ARGS,
    TOOL_CALL_END,
    THINKING_START,
    THINKING_END,
    TEXT_START,
    TEXT_DELTA,
    TEXT_END,
    DONE,
    ERROR,
)


# ---- SSE 模拟辅助 ----
def sse_line(delta: dict) -> str:
    payload = {"choices": [{"delta": delta}]}
    return f"data: {json.dumps(payload)}"


# =============================================================================
def test_simple_tool_call_accumulation():
    """单工具单次调用：正确累积 name + arguments"""
    print("\n=== test_simple_tool_call_accumulation ===")
    adapter = StreamAdapter()
    # 模拟真实 DeepSeek 流式格式：首chunk带name+空arguments，后续chunk逐步拼接
    events = list(adapter.adapt([
        sse_line({"tool_calls": [
            {"index": 0, "id": "call_1", "type": "function",
             "function": {"name": "search", "arguments": ""}}
        ]}),
        sse_line({"tool_calls": [
            {"index": 0, "function": {"arguments": "{"}}
        ]}),
        sse_line({"tool_calls": [
            {"index": 0, "function": {"arguments": '"q"'}}
        ]}),
        sse_line({"tool_calls": [
            {"index": 0, "function": {"arguments": ": "}}
        ]}),
        sse_line({"tool_calls": [
            {"index": 0, "function": {"arguments": '"test"}'}}
        ]}),
    ]))

    types = [e["type"] for e in events]
    assert TOOL_CALL_START in types, f"Missing TOOL_CALL_START in {types}"
    assert TOOL_CALL_ARGS in types, f"Missing TOOL_CALL_ARGS in {types}"
    assert TOOL_CALL_END in types, f"Missing TOOL_CALL_END in {types}"

    end_events = [e for e in events if e["type"] == TOOL_CALL_END]
    assert len(end_events) == 1
    assert end_events[0]["name"] == "search"
    assert end_events[0]["call_id"] == "call_1"
    assert end_events[0]["arguments"] == {"q": "test"}
    print("PASS")


def test_tool_call_with_thinking():
    """先thinking再tool_call：thinking正常结束再发tool_call"""
    print("\n=== test_tool_call_with_thinking ===")
    adapter = StreamAdapter()
    events = list(adapter.adapt([
        sse_line({"reasoning_content": "I need to"}),
        sse_line({"reasoning_content": " search"}),
        sse_line({"tool_calls": [
            {"index": 0, "id": "c2", "function": {"name": "query", "arguments": '{"x":1}'}}
        ]}),
    ]))

    types = [e["type"] for e in events]
    # thinking 必须先结束
    think_end_idx = types.index(THINKING_END)
    tool_start_idx = types.index(TOOL_CALL_START)
    assert think_end_idx < tool_start_idx, "Thinking must end before tool_call starts"
    print("PASS")


def test_multiple_tool_calls_in_one_chunk():
    """同一chunk包含两个tool_call（不同index）"""
    print("\n=== test_multiple_tool_calls_in_one_chunk ===")
    adapter = StreamAdapter()
    events = list(adapter.adapt([
        sse_line({"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "query", "arguments": '{"a":1}'}},
            {"index": 1, "id": "c2", "function": {"name": "fetch", "arguments": '{"b":2}'}},
        ]}),
    ]))

    starts = [e for e in events if e["type"] == TOOL_CALL_START]
    ends = [e for e in events if e["type"] == TOOL_CALL_END]
    assert len(starts) == 2, f"Expected 2 starts, got {len(starts)}"
    assert len(ends) == 2, f"Expected 2 ends, got {len(ends)}"
    names = {e["name"] for e in ends}
    assert names == {"query", "fetch"}
    print("PASS")


def test_tool_call_from_empty_stream():
    """无tool_calls时get_pending_tool_calls返回空列表"""
    print("\n=== test_tool_call_from_empty_stream ===")
    adapter = StreamAdapter()
    list(adapter.adapt([
        sse_line({"content": "Hello world"})
    ]))
    pending = adapter.get_pending_tool_calls()
    assert pending == [], f"Expected empty, got {pending}"
    print("PASS")


def test_state_reset_between_adapt_calls():
    """第二次adapt调用后状态被重置，不残留上次的tool_calls"""
    print("\n=== test_state_reset_between_adapt_calls ===")
    adapter = StreamAdapter()
    # 第一次流有 tool_call
    list(adapter.adapt([
        sse_line({"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "x", "arguments": '{}'}}
        ]}),
    ]))
    assert len(adapter.get_pending_tool_calls()) == 1

    # 第二次流是纯文本
    list(adapter.adapt([
        sse_line({"content": "Hello"})
    ]))
    pending = adapter.get_pending_tool_calls()
    assert len(pending) == 0 or all(not s.get("started") for s in pending), \
        f"Expected no started slots after reset, got {pending}"
    # 缓冲区也应被重置
    assert adapter._thinking_buf == ""
    assert adapter._text_buf == ""
    print("PASS")


def test_tool_call_args_ordering():
    """TOOL_CALL_ARGS不会在TOOL_CALL_START之前发出"""
    print("\n=== test_tool_call_args_ordering ===")
    adapter = StreamAdapter()
    # 模拟异常场景：arguments先于name到达
    events = list(adapter.adapt([
        sse_line({"tool_calls": [
            {"index": 0, "function": {"arguments": '{"x"'}}
        ]}),
        sse_line({"tool_calls": [
            {"index": 0, "id": "c3", "function": {"name": "late", "arguments": '":1}'}}
        ]}),
    ]))

    start_idx = None
    first_args_idx = None
    for i, e in enumerate(events):
        if e["type"] == TOOL_CALL_START and not start_idx:
            start_idx = i
        if e["type"] == TOOL_CALL_ARGS and first_args_idx is None:
            first_args_idx = i
    if first_args_idx is not None and start_idx is not None:
        assert start_idx < first_args_idx, \
            f"TOOL_CALL_START (idx={start_idx}) must precede TOOL_CALL_ARGS (idx={first_args_idx})"
    elif first_args_idx is not None and start_idx is None:
        # ARGS without START is a bug
        assert False, "TOOL_CALL_ARGS emitted without TOOL_CALL_START"
    print("PASS")


def test_tool_call_null_function():
    """function字段为null时不崩溃"""
    print("\n=== test_tool_call_null_function ===")
    adapter = StreamAdapter()
    events = list(adapter.adapt([
        sse_line({"tool_calls": [
            {"index": 0, "id": "c4", "type": "function", "function": None}
        ]}),
    ]))
    # 不应产生 TOOL_CALL_START（因为没有name）
    types = [e["type"] for e in events]
    assert TOOL_CALL_START not in types, f"Should not emit START for null function"
    assert ERROR not in types, f"Should not crash: {types}"
    print("PASS")


def test_tool_call_non_dict_entries():
    """tool_calls列表含非dict元素时安全跳过"""
    print("\n=== test_tool_call_non_dict_entries ===")
    adapter = StreamAdapter()
    events = list(adapter.adapt([
        sse_line({"tool_calls": [None]}),
        sse_line({"tool_calls": [{"index": 0, "id": "c5", "function": {"name": "ok", "arguments": "{}"}}]}),
    ]))
    ends = [e for e in events if e["type"] == TOOL_CALL_END]
    assert len(ends) == 1
    assert ends[0]["name"] == "ok"
    print("PASS")


def test_json_parse_fallback():
    """args_buf不是合法JSON时fallback为_raw包装"""
    print("\n=== test_json_parse_fallback ===")
    adapter = StreamAdapter()
    events = list(adapter.adapt([
        sse_line({"tool_calls": [
            {"index": 0, "id": "c6", "function": {"name": "bad", "arguments": "not valid json {"}}
        ]}),
    ]))
    ends = [e for e in events if e["type"] == TOOL_CALL_END]
    assert len(ends) == 1
    assert "_raw" in ends[0]["arguments"], f"Expected _raw fallback, got {ends[0]['arguments']}"
    print("PASS")


def test_existing_events_still_work():
    """纯文本流：thinking/text事件不受影响"""
    print("\n=== test_existing_events_still_work ===")
    adapter = StreamAdapter()
    events = list(adapter.adapt([
        sse_line({"content": "Hello"}),
        sse_line({"content": " world"}),
    ]))
    types = [e["type"] for e in events]
    assert TEXT_START in types
    assert TEXT_DELTA in types
    assert TEXT_END in types
    assert DONE in types
    # text_end 应该是完整文本
    text_ends = [e for e in events if e["type"] == TEXT_END]
    assert len(text_ends) == 1
    assert text_ends[0].get("text") == "Hello world"
    print("PASS")


# =============================================================================
if __name__ == "__main__":
    passed = 0
    failed = 0
    tests = [
        test_simple_tool_call_accumulation,
        test_tool_call_with_thinking,
        test_multiple_tool_calls_in_one_chunk,
        test_tool_call_from_empty_stream,
        test_state_reset_between_adapt_calls,
        test_tool_call_args_ordering,
        test_tool_call_null_function,
        test_tool_call_non_dict_entries,
        test_json_parse_fallback,
        test_existing_events_still_work,
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
