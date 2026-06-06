"""
Agent 集成测试 — 端到端流程（Mock LLM）

运行方法: python platform_init/test/agent_system/test_integration.py
"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from core.agent_tools import AgentTool, UnifiedToolExecutor
from core.agent_loop import AgentLoop


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


def text(s: str) -> str: return sse({"content": s})


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


# =============================================================================
def test_search_then_extract_flow():
    """流程: 用户说提取→Agent先搜索→再提取→汇报"""
    print("\n=== test_search_then_extract_flow ===")
    llm = MockLLM([
        [tc(0, "c1", "search_literature", '"{\\"query\\":\\"perovskite\\"}"')],
        [tc(0, "c2", "extract_from_pdf",
            '"{\\"pdf_path\\":\\"paper.pdf\\",\\"task_description\\":\\"extract params\\"}"')],
        [text("提取完成: 3条记录, 字段: name, value, efficiency")],
    ])
    tools = [
        tool("search_literature", "1. Paper A (score:0.85)\n2. Paper B (score:0.72)"),
        tool("extract_from_pdf", '{"status":"ok","records":3,"fields":["name","value","efficiency"]}'),
    ]
    loop = AgentLoop(llm=llm, executor=UnifiedToolExecutor(tools), model="test", max_turns=10)
    msgs = [{"role": "user", "content": "帮我提取钙钛矿钝化剂论文的参数"}]
    r = loop.run(msgs)
    assert r["error"] is None
    assert len(r["tool_turns"]) == 2
    assert r["tool_turns"][0].tool_name == "search_literature"
    assert r["tool_turns"][1].tool_name == "extract_from_pdf"
    print("PASS")


def test_preview_then_extract_flow():
    """流程: 先预览PDF→确认页码范围→提取"""
    print("\n=== test_preview_then_extract_flow ===")
    llm = MockLLM([
        [tc(0, "c1", "preview_pdf_page", '"{\\"pdf_path\\":\\"paper.pdf\\",\\"page_num\\":1}"')],
        [tc(0, "c2", "extract_from_pdf",
            '"{\\"pdf_path\\":\\"paper.pdf\\",\\"task_description\\":\\"extract\\",\\"pages\\":[1,2,3]}"')],
        [text("done")],
    ])
    tools = [
        tool("preview_pdf_page", "PDF: paper.pdf, 第1/10页"),
        tool("extract_from_pdf", '{"status":"ok","records":5}'),
    ]
    loop = AgentLoop(llm=llm, executor=UnifiedToolExecutor(tools), model="test", max_turns=10)
    r = loop.run([{"role": "user", "content": "分析paper.pdf的实验数据"}])
    assert r["error"] is None
    assert len(r["tool_turns"]) == 2
    assert r["tool_turns"][0].tool_name == "preview_pdf_page"
    assert r["tool_turns"][1].tool_name == "extract_from_pdf"
    print("PASS")


def test_experiment_design_with_clarification():
    """流程: 用户描述实验→Agent design_experiment→返回JSON"""
    print("\n=== test_experiment_design_with_clarification ===")
    design_json = json.dumps({"experiment_name": "旋涂实验", "steps": [
        {"type": "tool", "name": "spin_coating",
         "params": {"spin_speed": 3000, "spin_dur": 30000}}]})
    llm = MockLLM([
        [tc(0, "c1", "design_experiment",
            '"{\\"description\\":\\"design a spin coating experiment at 3000rpm\\"}"')],
        [text("实验设计完成: 旋涂实验, 1步骤")],
    ])
    tools = [tool("design_experiment", design_json)]
    loop = AgentLoop(llm=llm, executor=UnifiedToolExecutor(tools), model="test", max_turns=10)
    r = loop.run([{"role": "user", "content": "帮我设计一个旋涂实验，转速3000rpm"}])
    assert r["error"] is None
    assert len(r["tool_turns"]) == 1
    assert r["tool_turns"][0].tool_name == "design_experiment"
    print("PASS")


def test_data_analysis_flow():
    """流程: 数据分析→选择算法→执行→返回结果"""
    print("\n=== test_data_analysis_flow ===")
    llm = MockLLM([
        [tc(0, "c1", "data_statistics", '"{\\"data\\":\\"sample.csv\\"}"')],
        [text("分析结果: mean=5.2, std=1.3")],
    ])
    tools = [tool("data_statistics", '{"mean":5.2,"std":1.3,"count":100}')]
    loop = AgentLoop(llm=llm, executor=UnifiedToolExecutor(tools), model="test", max_turns=10)
    r = loop.run([{"role": "user", "content": "分析sample.csv的基础统计"}])
    assert r["error"] is None
    assert len(r["tool_turns"]) == 1
    assert r["tool_turns"][0].tool_name == "data_statistics"
    print("PASS")


def test_error_recovery_flow():
    """流程: 工具失败→Agent重试→成功"""
    print("\n=== test_error_recovery_flow ===")
    first_call = True

    def unreliable(args):
        nonlocal first_call
        if first_call:
            first_call = False
            raise RuntimeError("temp failure")
        return "success on retry"

    bad_tool = AgentTool(name="unreliable", description="", parameters={"type": "object", "properties": {}},
                         required=[], func=unreliable, category="builtin")

    llm = MockLLM([
        [tc(0, "c1", "unreliable", '{}')],
        [tc(0, "c2", "unreliable", '{}')],  # Agent retries
        [text("ok now")],
    ])
    loop = AgentLoop(llm=llm, executor=UnifiedToolExecutor([bad_tool]), model="test", max_turns=5)
    r = loop.run([{"role": "user", "content": "go"}])
    assert r["error"] is None
    assert len(r["tool_turns"]) == 2
    assert r["tool_turns"][0].status == "error"
    assert r["tool_turns"][1].status == "success"
    print("PASS")


def test_empty_search_results():
    """搜索无结果→Agent友好汇报"""
    print("\n=== test_empty_search_results ===")
    llm = MockLLM([
        [tc(0, "c1", "search_literature", '"{\\"query\\":\\"nonexistent\\"}"')],
        [text("未找到相关文献，建议换关键词重试")],
    ])
    tools = [tool("search_literature", "未找到相关文献")]
    loop = AgentLoop(llm=llm, executor=UnifiedToolExecutor(tools), model="test", max_turns=5)
    r = loop.run([{"role": "user", "content": "搜索不存在的内容"}])
    assert r["error"] is None
    assert len(r["tool_turns"]) == 1
    assert "未找到" in r["tool_turns"][0].result
    print("PASS")


def test_hardware_control_dangerous():
    """硬件工具标记 dangerous → Agent 应该确认（由 system prompt 控制）"""
    print("\n=== test_hardware_control_dangerous ===")
    hw_tool = AgentTool(name="spin_coating", description="spin coating",
                        parameters={"type": "object", "properties": {}},
                        required=[], func=lambda a: "spin_coating executed",
                        category="hardware", dangerous=True)
    llm = MockLLM([
        [tc(0, "c1", "spin_coating", '"{\\"spin_speed\\":3000}"')],
        [text("硬件操作已完成")],
    ])
    loop = AgentLoop(llm=llm, executor=UnifiedToolExecutor([hw_tool]), model="test", max_turns=5)
    r = loop.run([{"role": "user", "content": "执行旋涂实验3000rpm"}])
    assert r["error"] is None
    assert len(r["tool_turns"]) == 1
    assert r["tool_turns"][0].tool_name == "spin_coating"
    print("PASS")


def test_multiple_tools_same_round():
    """同一轮调用2个工具"""
    print("\n=== test_multiple_tools_same_round ===")
    llm = MockLLM([
        [tc(0, "c1", "search_literature", '"{\\"query\\":\\"A\\"}"'),
         tc(1, "c2", "search_literature", '"{\\"query\\":\\"B\\"}"')],
        [text("both done")],
    ])
    tools = [tool("search_literature", "results")]
    loop = AgentLoop(llm=llm, executor=UnifiedToolExecutor(tools), model="test", max_turns=5)
    r = loop.run([{"role": "user", "content": "search A and B"}])
    assert r["error"] is None
    assert len(r["tool_turns"]) == 2
    assert {t.tool_name for t in r["tool_turns"]} == {"search_literature"}
    print("PASS")


# =============================================================================
if __name__ == "__main__":
    passed = failed = 0
    tests = [test_search_then_extract_flow, test_preview_then_extract_flow,
             test_experiment_design_with_clarification, test_data_analysis_flow,
             test_error_recovery_flow, test_empty_search_results,
             test_hardware_control_dangerous, test_multiple_tools_same_round]
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
