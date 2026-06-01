"""
AgentLoop — 单元测试（使用 Mock LLM）

运行方法: python platform_init/test/agent/test_agent_loop.py
"""
import sys
import io
import os
import json
import queue

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from core.agent_tools import AgentTool, UnifiedToolExecutor
from core.agent_loop import AgentLoop, AgentOrchestrator, SubAgent, AgentTurn


# =============================================================================
# Mock LLM — 返回预设的 SSE 行序列
# =============================================================================
class MockLLM:
    """模拟 LLM，返回预制的 SSE 行"""
    def __init__(self, responses: list[list[str]]):
        """
        responses: 每轮调用的 SSE 行列表的列表。
        [[round1_lines], [round2_lines], ...]
        """
        self._responses = responses
        self._call_count = 0
        self._args = []  # 记录每次调用的参数

    def stream_raw(self, model, messages, tools=None, extra_body=None):
        self._args.append({"model": model, "messages": messages, "tools": tools})
        if self._call_count < len(self._responses):
            lines = self._responses[self._call_count]
            self._call_count += 1
            yield from lines
        else:
            # 默认：返回纯文本
            yield from []


# ---- SSE 行构造辅助 ----
def sse(delta: dict) -> str:
    payload = {"choices": [{"delta": delta}]}
    return f"data: {json.dumps(payload)}"


def text_delta(content: str) -> str:
    return sse({"content": content})


def tool_call_delta(index: int, id_: str = "", name: str = "", arguments: str = "") -> str:
    fn = {}
    if name:
        fn["name"] = name
    if arguments:
        fn["arguments"] = arguments
    tc = {"index": index}
    if id_:
        tc["id"] = id_
    if name:
        tc["type"] = "function"
    tc["function"] = fn
    return sse({"tool_calls": [tc]})


def reasoning_delta(text: str) -> str:
    return sse({"reasoning_content": text})


# =============================================================================
def make_tool(name: str, response: str = "ok") -> AgentTool:
    return AgentTool(
        name=name,
        description=f"Tool {name}",
        parameters={"type": "object", "properties": {}},
        required=[],
        func=lambda args, r=response: r,
        category="builtin",
    )


# =============================================================================
def test_agent_loop_text_only():
    """纯文本回答：不调工具，直接返回final_message"""
    print("\n=== test_agent_loop_text_only ===")
    llm = MockLLM([
        [text_delta("Hello"), text_delta(" world")]
    ])
    executor = UnifiedToolExecutor([])
    loop = AgentLoop(llm=llm, executor=executor, model="test", max_turns=3)

    messages = [{"role": "user", "content": "hi"}]
    result = loop.run(messages=messages)

    assert result["error"] is None, f"Unexpected error: {result['error']}"
    assert result["final_message"] is not None
    assert result["final_message"]["role"] == "assistant"
    assert len(result["tool_turns"]) == 0
    print("PASS")


def test_agent_loop_single_tool_call():
    """一轮tool call：LLM调工具→执行→继续→返回文本"""
    print("\n=== test_agent_loop_single_tool_call ===")
    llm = MockLLM([
        # 第1轮：调工具
        [
            tool_call_delta(0, id_="c1", name="search", arguments='{"q"'),
            tool_call_delta(0, arguments='":"test"}'),
        ],
        # 第2轮：返回文本
        [text_delta("Found 3 results")]
    ])
    executor = UnifiedToolExecutor([make_tool("search", "result: 3 docs")])
    loop = AgentLoop(llm=llm, executor=executor, model="test", max_turns=5)

    messages = [{"role": "user", "content": "search for test"}]
    result = loop.run(messages=messages)

    assert result["error"] is None, f"Unexpected error: {result['error']}"
    assert len(result["tool_turns"]) == 1
    assert result["tool_turns"][0].tool_name == "search"
    assert result["tool_turns"][0].status == "success"
    assert "3 docs" in result["tool_turns"][0].result
    print("PASS")


def test_agent_loop_multi_tool_calls():
    """多轮tool call：LLM调两个工具→每轮都执行"""
    print("\n=== test_agent_loop_multi_tool_calls ===")
    llm = MockLLM([
        # 第1轮：调 search
        [tool_call_delta(0, id_="c1", name="search", arguments='{"q":"x"}')],
        # 第2轮：调 fetch
        [tool_call_delta(0, id_="c2", name="fetch", arguments='{"id":1}')],
        # 第3轮：文本
        [text_delta("done")],
    ])
    executor = UnifiedToolExecutor([
        make_tool("search", "search_ok"),
        make_tool("fetch", "fetch_ok"),
    ])
    loop = AgentLoop(llm=llm, executor=executor, model="test", max_turns=5)

    messages = [{"role": "user", "content": "go"}]
    result = loop.run(messages=messages)

    assert result["error"] is None
    assert len(result["tool_turns"]) == 2
    assert result["tool_turns"][0].tool_name == "search"
    assert result["tool_turns"][1].tool_name == "fetch"
    print("PASS")


def test_agent_loop_max_turns():
    """达到max_turns上限：返回error"""
    print("\n=== test_agent_loop_max_turns ===")
    # 每轮都调工具（会无限循环）
    responses = [[tool_call_delta(0, id_=f"c{i}", name="loop", arguments='{}')] for i in range(5)]
    llm = MockLLM(responses)
    executor = UnifiedToolExecutor([make_tool("loop", "looping")])
    loop = AgentLoop(llm=llm, executor=executor, model="test", max_turns=3)

    messages = [{"role": "user", "content": "loop"}]
    result = loop.run(messages=messages)

    assert result["error"] is not None, "Should have error from max_turns"
    assert "最大" in result["error"] or "max" in result["error"].lower()
    print("PASS")


def test_agent_loop_ask_user():
    """ask_user 触发暂停：queue中预先放入答案，直接取回继续"""
    print("\n=== test_agent_loop_ask_user ===")
    llm = MockLLM([
        [tool_call_delta(0, id_="c1", name="ask_user", arguments='{"question":"confirm?"}')],
        [text_delta("confirmed")],
    ])
    executor = UnifiedToolExecutor([make_tool("ask_user", "__ASK_USER_PENDING__")])
    loop = AgentLoop(llm=llm, executor=executor, model="test", max_turns=5)

    # 预先放入答案，AgentLoop 直接取到
    ask_queue = queue.Queue()
    ask_queue.put("yes, proceed")

    events = []
    messages = [{"role": "user", "content": "do it"}]
    result = loop.run(messages=messages, event_callback=lambda e: events.append(e), ask_user_queue=ask_queue)

    assert result["error"] is None
    assert len(result["tool_turns"]) == 1
    assert result["tool_turns"][0].tool_name == "ask_user"
    assert result["tool_turns"][0].result == "yes, proceed"
    print("PASS")


def test_agent_loop_ask_user_no_queue():
    """无ask_user_queue时，dispatch返回__ASK_USER_PENDING__占位符"""
    print("\n=== test_agent_loop_ask_user_no_queue ===")
    llm = MockLLM([
        [tool_call_delta(0, id_="c1", name="ask_user", arguments='{"question":"wait?"}')],
        [text_delta("handled")],
    ])
    executor = UnifiedToolExecutor([make_tool("ask_user", "__ASK_USER_PENDING__")])
    loop = AgentLoop(llm=llm, executor=executor, model="test", max_turns=5)

    messages = [{"role": "user", "content": "test"}]
    # 不传ask_user_queue走dispatch正常路径
    result = loop.run(messages=messages)
    assert result["error"] is None
    assert len(result["tool_turns"]) == 1
    # 无queue时dispatch返回占位符字符串
    assert result["tool_turns"][0].result == "__ASK_USER_PENDING__"
    print("PASS")


def test_agent_loop_tool_error():
    """工具执行异常时记录error状态，不中断循环"""
    print("\n=== test_agent_loop_tool_error ===")
    def failing_tool(args):
        raise RuntimeError("boom!")

    llm = MockLLM([
        [tool_call_delta(0, id_="c1", name="bad_tool", arguments='{}')],
        [text_delta("handled error")],
    ])
    tool = AgentTool(
        name="bad_tool", description="",
        parameters={"type": "object", "properties": {}}, required=[],
        func=failing_tool, category="builtin",
    )
    executor = UnifiedToolExecutor([tool])
    loop = AgentLoop(llm=llm, executor=executor, model="test", max_turns=5)

    messages = [{"role": "user", "content": "go"}]
    result = loop.run(messages=messages)

    assert result["error"] is None
    assert len(result["tool_turns"]) == 1
    assert result["tool_turns"][0].status == "error", f"Expected error status, got {result['tool_turns'][0].status}"
    assert "错误" in result["tool_turns"][0].result
    print("PASS")


def test_agent_loop_reasoning_visible():
    """reasoning_content被正确捕获为thinking事件"""
    print("\n=== test_agent_loop_reasoning_visible ===")
    llm = MockLLM([
        [reasoning_delta("I think"), reasoning_delta(" this"), text_delta("answer")],
    ])
    executor = UnifiedToolExecutor([])
    loop = AgentLoop(llm=llm, executor=executor, model="test", max_turns=3)

    events = []
    messages = [{"role": "user", "content": "q"}]
    result = loop.run(messages=messages, event_callback=lambda e: events.append(e))

    types = [e["type"] for e in events]
    assert "thinking_start" in types, f"Missing thinking_start in {types}"
    assert "thinking_delta" in types, f"Missing thinking_delta in {types}"
    assert "thinking_end" in types, f"Missing thinking_end in {types}"
    print("PASS")


def test_agent_turn_dataclass():
    """AgentTurn dataclass 字段正确"""
    print("\n=== test_agent_turn_dataclass ===")
    turn = AgentTurn(tool_name="test", arguments={"a": 1}, result="ok", status="success")
    assert turn.tool_name == "test"
    assert turn.status == "success"
    assert turn.result == "ok"
    print("PASS")


def test_agent_orchestrator_list_templates():
    """AgentOrchestrator.list_templates() 返回YAML模板列表"""
    print("\n=== test_agent_orchestrator_list_templates ===")
    orch = AgentOrchestrator()
    templates = orch.list_templates()
    assert "literature_searcher" in templates, f"Missing literature_searcher in {templates}"
    assert "experiment_designer" in templates
    assert "summarizer" in templates
    assert "data_analyst" in templates
    print(f"  Templates: {templates}")
    print("PASS")


def test_agent_orchestrator_spawn_nonexistent():
    """spawn不存在的模板返回error"""
    print("\n=== test_agent_orchestrator_spawn_nonexistent ===")
    orch = AgentOrchestrator()
    result = orch.spawn("nonexistent_template", "do something")
    assert result.get("error") is not None, f"Expected error, got {result}"
    print("PASS")


# =============================================================================
if __name__ == "__main__":
    passed = 0
    failed = 0
    tests = [
        test_agent_turn_dataclass,
        test_agent_loop_text_only,
        test_agent_loop_single_tool_call,
        test_agent_loop_multi_tool_calls,
        test_agent_loop_max_turns,
        test_agent_loop_ask_user,
        test_agent_loop_ask_user_no_queue,
        test_agent_loop_tool_error,
        test_agent_loop_reasoning_visible,
        test_agent_orchestrator_list_templates,
        test_agent_orchestrator_spawn_nonexistent,
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
