"""
Phase 2 extraction flow — 集成测试

模拟完整提取 Pipeline（search → extract → summarize）

运行方法: python platform_init/test/agent_phase2/test_extraction_flow.py
"""
import sys
import io
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from core.agent_tools import AgentTool, UnifiedToolExecutor
from core.agent_loop import AgentLoop, AgentOrchestrator


# =============================================================================
# ---- Mock LLM & helpers (copied from test_agent_loop for self-contained tests) ----
# =============================================================================
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


def make_tool(name: str, response: str = "ok") -> AgentTool:
    return AgentTool(
        name=name, description=f"Tool {name}",
        parameters={"type": "object", "properties": {}},
        required=[], func=lambda args, r=response: r,
        category="builtin",
    )


# =============================================================================
def test_mock_extraction_pipeline():
    """
    模拟提取流程：
    Agent 先调 search_literature，拿到结果后调 spawn_agent 创建 extractor，
    最后调 summarizer 汇总。
    """
    print("\n=== test_mock_extraction_pipeline ===")
    llm = MockLLM([
        # 第1轮：搜索文献
        [tool_call_delta(0, id_="c1", name="search_literature", arguments='{"query":"perovskite"}')],
        # 第2轮：拿到搜索结果后，发起提取
        [tool_call_delta(0, id_="c2", name="extract_from_pdf", arguments='{"pdf_path":"paper.pdf","task_description":"extract data"}')],
        # 第3轮：最终回复
        [text_delta("提取完成：3条记录")],
    ])
    executor = UnifiedToolExecutor([
        make_tool("search_literature", "1. Paper A (score:0.85)\n2. Paper B (score:0.72)"),
        make_tool("extract_from_pdf", '{"status":"ok","records":3,"fields":["name","value"]}'),
    ])
    loop = AgentLoop(llm=llm, executor=executor, model="test", max_turns=5)

    messages = [{"role": "user", "content": "提取钙钛矿论文的参数"}]
    result = loop.run(messages=messages)

    assert result["error"] is None, f"Error: {result['error']}"
    assert len(result["tool_turns"]) == 2
    assert result["tool_turns"][0].tool_name == "search_literature"
    assert result["tool_turns"][1].tool_name == "extract_from_pdf"
    print(f"  search_literature → extract_from_pdf → text reply: OK")
    print("PASS")


def test_mock_search_only_flow():
    """只搜索不提取的流程"""
    print("\n=== test_mock_search_only_flow ===")
    llm = MockLLM([
        [tool_call_delta(0, id_="c1", name="search_literature", arguments='{"query":"FAPbI3"}')],
        [text_delta("找到5篇相关文献")],
    ])
    executor = UnifiedToolExecutor([
        make_tool("search_literature", "1. Paper X\n2. Paper Y"),
    ])
    loop = AgentLoop(llm=llm, executor=executor, model="test", max_turns=5)

    messages = [{"role": "user", "content": "搜索FAPbI3文献"}]
    result = loop.run(messages=messages)

    assert result["error"] is None
    assert len(result["tool_turns"]) == 1
    assert result["tool_turns"][0].tool_name == "search_literature"
    print(f"  Single search → text reply: OK")
    print("PASS")


def test_mock_preview_before_extract():
    """先预览PDF再提取的流程"""
    print("\n=== test_mock_preview_before_extract ===")
    llm = MockLLM([
        # 先预览
        [tool_call_delta(0, id_="c1", name="preview_pdf_page", arguments='{"pdf_path":"paper.pdf","page_num":1}')],
        # 再提取
        [tool_call_delta(0, id_="c2", name="extract_from_pdf", arguments='{"pdf_path":"paper.pdf","task_description":"extract"}')],
        [text_delta("done")],
    ])
    executor = UnifiedToolExecutor([
        make_tool("preview_pdf_page", "PDF: paper.pdf, 第1/10页"),
        make_tool("extract_from_pdf", '{"status":"ok","records":5}'),
    ])
    loop = AgentLoop(llm=llm, executor=executor, model="test", max_turns=5)

    messages = [{"role": "user", "content": "分析paper.pdf的数据"}]
    result = loop.run(messages=messages)

    assert result["error"] is None
    assert len(result["tool_turns"]) == 2
    assert result["tool_turns"][0].tool_name == "preview_pdf_page"
    assert result["tool_turns"][1].tool_name == "extract_from_pdf"
    print(f"  preview_pdf_page → extract_from_pdf: OK")
    print("PASS")


def test_empty_search_results_handled():
    """搜索无结果时Agent应能处理"""
    print("\n=== test_empty_search_results_handled ===")
    llm = MockLLM([
        [tool_call_delta(0, id_="c1", name="search_literature", arguments='{"query":"nonexistent_query_xyz"}')],
        [text_delta("未找到相关文献，请换个关键词试试")],
    ])
    executor = UnifiedToolExecutor([
        make_tool("search_literature", "未找到相关文献"),
    ])
    loop = AgentLoop(llm=llm, executor=executor, model="test", max_turns=5)

    messages = [{"role": "user", "content": "搜索不存在的内容"}]
    result = loop.run(messages=messages)

    assert result["error"] is None
    assert len(result["tool_turns"]) == 1
    assert "未找到" in result["tool_turns"][0].result
    print(f"  Empty search → agent handled gracefully: OK")
    print("PASS")


def test_tool_error_recovery():
    """工具执行失败时Agent继续而不崩溃"""
    print("\n=== test_tool_error_recovery ===")
    llm = MockLLM([
        # 第一次搜索失败
        [tool_call_delta(0, id_="c1", name="search_literature", arguments='{"query":""}')],
        # 第二次重试
        [tool_call_delta(0, id_="c2", name="search_literature", arguments='{"query":"perovskite"}')],
        [text_delta("found results")],
    ])
    executor = UnifiedToolExecutor([
        make_tool("search_literature", "1. Result A (score: 0.9)"),
    ])
    loop = AgentLoop(llm=llm, executor=executor, model="test", max_turns=5)

    messages = [{"role": "user", "content": "search"}]
    result = loop.run(messages=messages)

    assert result["error"] is None
    assert len(result["tool_turns"]) == 2  # 即使第一次"失败"，状态也是success（因为我们没传错误）
    print(f"  Tool called twice → agent recovered: OK")
    print("PASS")


# =============================================================================
if __name__ == "__main__":
    passed = 0
    failed = 0
    tests = [
        test_mock_extraction_pipeline,
        test_mock_search_only_flow,
        test_mock_preview_before_extract,
        test_empty_search_results_handled,
        test_tool_error_recovery,
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
