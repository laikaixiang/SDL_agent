"""
AgentLoop 生命周期 — 系统测试（Mock LLM）

运行方法: python platform_init/test/agent_system/test_agent_lifecycle.py
"""
import sys, io, os, json, queue, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from core.agent_tools import AgentTool, UnifiedToolExecutor
from core.agent_loop import AgentLoop, AgentTurn


# ---- Mock LLM ----
class MockLLM:
    def __init__(self, responses: list[list[str]]):
        self._responses = responses
        self._call_count = 0
        self._args = []

    def stream_raw(self, model, messages, tools=None, extra_body=None):
        self._args.append({"model": model, "messages": messages, "tools": tools})
        if self._call_count < len(self._responses):
            lines = self._responses[self._call_count]
            self._call_count += 1
            yield from lines
        else:
            yield from []


def sse(delta: dict) -> str:
    payload = {"choices": [{"delta": delta}]}
    return f"data: {json.dumps(payload)}"


def text(s: str) -> str:       return sse({"content": s})
def reasoning(s: str) -> str:  return sse({"reasoning_content": s})


def tc(index, id_="", name="", args=""):
    fn = {}
    if name: fn["name"] = name
    if args: fn["arguments"] = args
    obj = {"index": index}
    if id_: obj["id"] = id_
    if name: obj["type"] = "function"
    obj["function"] = fn
    return sse({"tool_calls": [obj]})


def tool(name, response="ok"):
    return AgentTool(name=name, description=f"tool {name}",
                     parameters={"type": "object", "properties": {}},
                     required=[], func=lambda a, r=response: r, category="builtin")


def make_loop(llm, tools, max_turns=5):
    return AgentLoop(llm=llm, executor=UnifiedToolExecutor(tools), model="test", max_turns=max_turns)


# =============================================================================
def test_text_reply():
    """单轮文本回答"""
    print("\n=== test_text_reply ===")
    loop = make_loop(MockLLM([[text("Hello world")]]), [])
    r = loop.run([{"role": "user", "content": "hi"}])
    assert r["error"] is None
    assert r["final_message"]["role"] == "assistant"
    assert "Hello" in r["final_message"]["content"]
    assert len(r["tool_turns"]) == 0
    print("PASS")


def test_single_tool_call():
    """一轮 tool call → 执行 → 继续 → 文本"""
    print("\n=== test_single_tool_call ===")
    llm = MockLLM([
        [tc(0, "c1", "search", '"{\\"q\\":\\"x\\"}"')],
        [text("found 3 results")],
    ])
    loop = make_loop(llm, [tool("search", "result: 3 docs")])
    r = loop.run([{"role": "user", "content": "search x"}])
    assert r["error"] is None
    assert len(r["tool_turns"]) == 1
    assert r["tool_turns"][0].tool_name == "search"
    assert r["tool_turns"][0].status == "success"
    print("PASS")


def test_multi_tool_multi_round():
    """多轮 tool call: 3轮每轮1个工具"""
    print("\n=== test_multi_tool_multi_round ===")
    llm = MockLLM([
        [tc(0, "c1", "a", '{}')],
        [tc(0, "c2", "b", '{}')],
        [tc(0, "c3", "c", '{}')],
        [text("done")],
    ])
    loop = make_loop(llm, [tool("a"), tool("b"), tool("c")], max_turns=10)
    r = loop.run([{"role": "user", "content": "go"}])
    assert r["error"] is None
    assert len(r["tool_turns"]) == 3
    names = [t.tool_name for t in r["tool_turns"]]
    assert names == ["a", "b", "c"]
    print("PASS")


def test_max_turns_limit():
    """超过 max_turns 返回 error"""
    print("\n=== test_max_turns_limit ===")
    rounds = [[tc(0, f"c{i}", "loop", '{}')] for i in range(10)]
    llm = MockLLM(rounds)
    loop = make_loop(llm, [tool("loop")], max_turns=3)
    r = loop.run([{"role": "user", "content": "loop forever"}])
    assert r["error"] is not None
    assert "最大" in r["error"] or "max" in r["error"].lower()
    print("PASS")


def test_ask_user_interrupt():
    """ask_user → queue.get() → 继续"""
    print("\n=== test_ask_user_interrupt ===")
    llm = MockLLM([
        [tc(0, "c1", "ask_user", '{"question":"confirm?"}')],
        [text("confirmed")],
    ])
    q = queue.Queue()
    q.put("yes")  # 预先放入答案
    loop = make_loop(llm, [tool("ask_user", "__ASK_USER_PENDING__")])
    events = []
    r = loop.run([{"role": "user", "content": "do"}],
                 event_callback=lambda e: events.append(e),
                 ask_user_queue=q)
    assert r["error"] is None
    assert len(r["tool_turns"]) == 1
    assert r["tool_turns"][0].result == "yes"
    # 验证 agent_question 事件被发送
    q_events = [e for e in events if e["type"] == "agent_question"]
    assert len(q_events) == 1
    print("PASS")


def test_ask_user_no_queue():
    """无 ask_user_queue 时 dispatch 正常走"""
    print("\n=== test_ask_user_no_queue ===")
    llm = MockLLM([
        [tc(0, "c1", "ask_user", '{"question":"q"}')],
        [text("handled")],
    ])
    loop = make_loop(llm, [tool("ask_user", "__ASK_USER_PENDING__")])
    r = loop.run([{"role": "user", "content": "test"}])
    assert r["error"] is None
    assert r["tool_turns"][0].result == "__ASK_USER_PENDING__"
    print("PASS")


def test_tool_error_handling():
    """工具执行异常 → status=error，继续循环"""
    print("\n=== test_tool_error_handling ===")
    def fail(args):
        raise RuntimeError("boom!")
    bad_tool = AgentTool(name="bad", description="", parameters={"type": "object", "properties": {}},
                         required=[], func=fail, category="builtin")
    llm = MockLLM([
        [tc(0, "c1", "bad", '{}')],
        [text("recovered")],
    ])
    loop = make_loop(llm, [bad_tool])
    r = loop.run([{"role": "user", "content": "go"}])
    assert r["error"] is None
    assert len(r["tool_turns"]) == 1
    assert r["tool_turns"][0].status == "error"
    assert "错误" in r["tool_turns"][0].result
    print("PASS")


def test_reasoning_events():
    """reasoning_content → thinking_start/delta/end 事件"""
    print("\n=== test_reasoning_events ===")
    llm = MockLLM([[reasoning("I "), reasoning("think"), text("answer")]])
    loop = make_loop(llm, [])
    events = []
    r = loop.run([{"role": "user", "content": "q"}],
                 event_callback=lambda e: events.append(e))
    types = [e["type"] for e in events]
    assert "thinking_start" in types
    assert "thinking_delta" in types
    assert "thinking_end" in types
    assert "text_delta" in types
    print(f"  Event types: {types}")
    print("PASS")


def test_event_callback_all_types():
    """event_callback 覆盖所有事件类型"""
    print("\n=== test_event_callback_all_types ===")
    llm = MockLLM([
        [reasoning("think"), tc(0, "c1", "search", '"{\\"q\\":\\"x\\"}"')],
        [text("done")],
    ])
    loop = make_loop(llm, [tool("search", "ok")])
    events = []
    r = loop.run([{"role": "user", "content": "go"}],
                 event_callback=lambda e: events.append(e))
    types = {e["type"] for e in events}
    required = {"thinking_start", "thinking_delta", "thinking_end",
                "tool_call_start", "tool_call_args", "tool_call_end",
                "tool_result"}
    missing = required - types
    assert not missing, f"Missing event types: {missing}"
    print(f"  All {len(required)} event types present: {sorted(required)}")
    print("PASS")


def test_agent_turn_dataclass():
    """AgentTurn 字段验证"""
    print("\n=== test_agent_turn_dataclass ===")
    t = AgentTurn(tool_name="x", arguments={"a": 1}, result="ok", status="success")
    assert t.tool_name == "x"
    assert t.status == "success"
    e = AgentTurn(tool_name="e", arguments={}, result="fail", status="error")
    assert e.status == "error"
    print("PASS")


def test_tool_result_summary_in_messages():
    """每次 tool call 后 messages 中包含 tool result"""
    print("\n=== test_tool_result_summary_in_messages ===")
    llm = MockLLM([
        [tc(0, "c1", "search", '"{\\"q\\":\\"x\\"}"')],
        [text("done")],
    ])
    loop = make_loop(llm, [tool("search", "found 5 items")])
    msgs = [{"role": "user", "content": "go"}]
    r = loop.run(msgs)
    assert r["error"] is None
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "found 5 items" in tool_msgs[0]["content"]
    print("PASS")


# =============================================================================
if __name__ == "__main__":
    passed = failed = 0
    tests = [test_text_reply, test_single_tool_call, test_multi_tool_multi_round,
             test_max_turns_limit, test_ask_user_interrupt, test_ask_user_no_queue,
             test_tool_error_handling, test_reasoning_events,
             test_event_callback_all_types, test_agent_turn_dataclass,
             test_tool_result_summary_in_messages]
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
