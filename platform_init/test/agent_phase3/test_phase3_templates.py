"""
Phase 3 子 Agent 模板 — 单元测试

验证所有 Domain Agent 模板正确加载、获得正确的工具子集。

运行方法: python platform_init/test/agent_phase3/test_phase3_templates.py
"""
import sys
import io
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from core.agent_tools import create_main_executor, AgentTool, UnifiedToolExecutor
from core.agent_loop import AgentOrchestrator, SubAgent

# =============================================================================
EXPECTED_TOOLS = {
    "experiment_designer":   ["design_experiment", "ask_user"],
    "data_analyst":          ["data_statistics", "data_normalization", "spectrum_analysis", "ask_user"],
    "literature_extractor":  ["preview_pdf_page", "extract_from_pdf", "ask_user"],
    "literature_searcher":   ["search_literature", "ask_user"],
    "extraction_pipeline":   ["search_literature", "preview_pdf_page", "spawn_agent"],
    "summarizer":            [],
}


def _load_agent(template_name: str) -> SubAgent:
    executor = create_main_executor()
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "prompts", "zh", "agents", f"{template_name}.yaml"
    )
    return SubAgent(path, executor=executor)


# =============================================================================
def test_all_templates_exist():
    """所有 6 个模板文件存在且可被 AgentOrchestrator 发现"""
    print("\n=== test_all_templates_exist ===")
    orch = AgentOrchestrator()
    templates = orch.list_templates()
    for name in EXPECTED_TOOLS:
        assert name in templates, f"Missing template: {name}"
    print(f"  All {len(EXPECTED_TOOLS)} templates present: {sorted(templates)}")
    print("PASS")


def test_experiment_designer_tools():
    """experiment_designer 获得 2 个工具"""
    print("\n=== test_experiment_designer_tools ===")
    agent = _load_agent("experiment_designer")
    names = agent.executor.names
    assert "design_experiment" in names, f"Missing design_experiment, got {names}"
    assert "ask_user" in names, f"Missing ask_user, got {names}"
    assert len(names) == 2, f"Expected 2 tools, got {len(names)}: {names}"
    print(f"  Tools: {names}")
    print("PASS")


def test_data_analyst_tools():
    """data_analyst 获得 4 个工具"""
    print("\n=== test_data_analyst_tools ===")
    agent = _load_agent("data_analyst")
    names = agent.executor.names
    assert "data_statistics" in names
    assert "data_normalization" in names
    assert "spectrum_analysis" in names
    assert "ask_user" in names
    assert len(names) == 4, f"Expected 4 tools, got {len(names)}: {names}"
    print(f"  Tools: {names}")
    print("PASS")


def test_literature_extractor_tools():
    """literature_extractor 获得 3 个工具"""
    print("\n=== test_literature_extractor_tools ===")
    agent = _load_agent("literature_extractor")
    names = agent.executor.names
    assert "preview_pdf_page" in names
    assert "extract_from_pdf" in names
    assert "ask_user" in names
    assert len(names) == 3, f"Expected 3 tools, got {len(names)}: {names}"
    print(f"  Tools: {names}")
    print("PASS")


def test_literature_searcher_tools():
    """literature_searcher 获得 2 个工具"""
    print("\n=== test_literature_searcher_tools ===")
    agent = _load_agent("literature_searcher")
    names = agent.executor.names
    assert "search_literature" in names
    assert "ask_user" in names
    assert len(names) == 2, f"Expected 2 tools, got {len(names)}: {names}"
    print(f"  Tools: {names}")
    print("PASS")


def test_summarizer_no_tools():
    """summarizer 无工具（纯文本 Agent）"""
    print("\n=== test_summarizer_no_tools ===")
    agent = _load_agent("summarizer")
    names = agent.executor.names
    assert len(names) == 0, f"Expected 0 tools, got {len(names)}: {names}"
    print(f"  Tools: {names} (text-only agent)")
    print("PASS")


def test_extraction_pipeline_tools():
    """extraction_pipeline 获得 search_literature + preview_pdf_page（spawn_agent 由 session 注入）"""
    print("\n=== test_extraction_pipeline_tools ===")
    agent = _load_agent("extraction_pipeline")
    names = agent.executor.names
    assert "search_literature" in names
    assert "preview_pdf_page" in names
    # spawn_agent 由 app.py 的 _make_session_executor() 按 session 注入，不在主 executor 中
    if "spawn_agent" in names:
        print(f"  spawn_agent: injected (session-level)")
    else:
        print(f"  spawn_agent: not in main executor (per-session tool, expected)")
    print(f"  Tools: {names}")
    print("PASS")


def test_subagent_template_loading():
    """所有模板可正常加载 YAML，不抛异常"""
    print("\n=== test_subagent_template_loading ===")
    for name in EXPECTED_TOOLS:
        agent = _load_agent(name)
        assert agent.system_prompt, f"{name}: system_prompt is empty"
        assert agent.name == name, f"{name}: name mismatch, got {agent.name}"
        assert agent.max_turns >= 3, f"{name}: max_turns={agent.max_turns} too low"
        print(f"  {name}: name={agent.name}, max_turns={agent.max_turns}, prompt_len={len(agent.system_prompt)}")
    print("PASS")


def test_subagent_has_independent_llm():
    """每个 SubAgent 有独立的 LLMClient 实例"""
    print("\n=== test_subagent_has_independent_llm ===")
    a1 = _load_agent("experiment_designer")
    a2 = _load_agent("data_analyst")
    assert a1.llm is not a2.llm, "SubAgents should have independent LLM instances"
    print("PASS")


def test_subagent_uses_talk_config():
    """SubAgent 使用 TALK_API_KEY 和 TALK_API_URL 配置"""
    print("\n=== test_subagent_uses_talk_config ===")
    from core.config import Config
    cfg = Config()
    agent = _load_agent("summarizer")
    assert agent.llm.get_api_key() == cfg.TALK_API_KEY
    assert agent.llm.get_api_url() == cfg.TALK_API_URL
    print("PASS")


def test_orchestrator_spawn_experiment_designer():
    """AgentOrchestrator.spawn 可以创建 experiment_designer"""
    print("\n=== test_orchestrator_spawn_experiment_designer ===")
    executor = create_main_executor()
    orch = AgentOrchestrator(executor=executor)
    result = orch.spawn("experiment_designer", "设计一个旋涂实验，转速3000rpm")
    # 可能成功或LLM调用失败，但不应该抛Python异常
    assert isinstance(result, dict)
    if result.get("error"):
        print(f"  LLM call failed (expected without API key or with wrong model): {result['error'][:100]}")
    else:
        print(f"  Result: OK")
    print("PASS")


def test_design_experiment_tool_dispatch():
    """design_experiment 工具可以直接调用"""
    print("\n=== test_design_experiment_tool_dispatch ===")
    from core.agent_tools import BUILTIN_TOOLS
    tool = next(t for t in BUILTIN_TOOLS if t.name == "design_experiment")
    executor = UnifiedToolExecutor([tool])
    result = executor.dispatch("design_experiment", {
        "description": "设计一个简单的旋涂实验",
    })
    assert isinstance(result, str)
    # 应该包含 JSON（成功）或错误信息
    print(f"  Result: {result[:200]}")
    print("PASS")


# =============================================================================
if __name__ == "__main__":
    passed = 0
    failed = 0
    tests = [
        test_all_templates_exist,
        test_experiment_designer_tools,
        test_data_analyst_tools,
        test_literature_extractor_tools,
        test_literature_searcher_tools,
        test_summarizer_no_tools,
        test_extraction_pipeline_tools,
        test_subagent_template_loading,
        test_subagent_has_independent_llm,
        test_subagent_uses_talk_config,
        test_design_experiment_tool_dispatch,
        test_orchestrator_spawn_experiment_designer,
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
