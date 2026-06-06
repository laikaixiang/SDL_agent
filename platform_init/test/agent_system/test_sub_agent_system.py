"""
SubAgent 系统 — 系统测试

运行方法: python platform_init/test/agent_system/test_sub_agent_system.py
"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from core.agent_tools import create_main_executor, AgentTool, UnifiedToolExecutor
from core.agent_loop import AgentOrchestrator, SubAgent, AgentLoop


executor = create_main_executor()
orch = AgentOrchestrator(executor=executor)

TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "prompts", "zh", "agents"
)


def _sub(tmpl): return SubAgent(os.path.join(TEMPLATE_DIR, f"{tmpl}.yaml"), executor=executor)


# =============================================================================
def test_all_templates_loadable():
    """6 个模板全部可加载不抛异常"""
    print("\n=== test_all_templates_loadable ===")
    for name in orch.list_templates():
        agent = _sub(name)
        assert agent.name == name
        assert agent.system_prompt
        assert agent.max_turns >= 3
        print(f"  {name}: OK ({len(agent.executor.names)} tools)")
    print("PASS")


def test_tool_subset_correct():
    """SubAgent 工具子集与模板 tools 字段一致"""
    print("\n=== test_tool_subset_correct ===")
    import yaml
    for name in orch.list_templates():
        with open(os.path.join(TEMPLATE_DIR, f"{name}.yaml"), encoding="utf-8") as f:
            data = yaml.safe_load(f)
        expected = data.get("tools", [])
        agent = _sub(name)
        actual = agent.executor.names
        # SubAgent filters expected ∩ main_executor
        for tool_name in expected:
            if executor.get(tool_name):  # 在 main executor 中存在
                assert tool_name in actual, f"{name}: expected tool '{tool_name}' not in {actual}"
        print(f"  {name}: expected {expected} → actual {actual}")
    print("PASS")


def test_subagent_independent_llm():
    """每个 SubAgent 有独立的 LLMClient"""
    print("\n=== test_subagent_independent_llm ===")
    a1 = _sub("summarizer")
    a2 = _sub("summarizer")
    assert a1.llm is not a2.llm
    print("PASS")


def test_subagent_uses_talk_config():
    """SubAgent 使用 TALK 配置"""
    print("\n=== test_subagent_uses_talk_config ===")
    from core.config import Config
    cfg = Config()
    a = _sub("summarizer")
    assert a.llm.get_api_key() == cfg.TALK_API_KEY
    assert a.llm.get_api_url() == cfg.TALK_API_URL
    print("PASS")


def test_spawn_valid_template():
    """AgentOrchestrator.spawn() 创建并运行子 Agent"""
    print("\n=== test_spawn_valid_template ===")
    result = orch.spawn("summarizer", "summarize: nothing to do")
    assert isinstance(result, dict)
    if result.get("error"):
        print(f"  LLM call result: error={result['error'][:80]}")
    else:
        print(f"  OK: {str(result.get('final_message',{}).get('content',''))[:80]}")
    print("PASS")


def test_spawn_nonexistent_template():
    """spawn 不存在的模板返回 error"""
    print("\n=== test_spawn_nonexistent_template ===")
    result = orch.spawn("not_a_real_template_xyz", "do something")
    assert "error" in result, f"Expected error, got {result}"
    print(f"  Error: {result['error'][:80]}")
    print("PASS")


def test_spawn_parallel():
    """spawn_parallel 并行执行多个同构子 Agent"""
    print("\n=== test_spawn_parallel ===")
    tasks = ["summarize: data is empty", "summarize: also empty"]
    results = orch.spawn_parallel("summarizer", tasks)
    assert len(results) == 2
    for i, r in enumerate(results):
        assert isinstance(r, dict)
        if r.get("error"):
            print(f"  Task {i}: error={r['error'][:60]}")
        else:
            print(f"  Task {i}: OK")
    print("PASS")


def test_pipeline_valid():
    """pipeline 模式正常执行"""
    print("\n=== test_pipeline_valid ===")
    steps = [
        {"template": "summarizer", "task": "step1: input is 'abc'"},
        {"template": "summarizer", "task": "step2: process previous"},
    ]
    results = orch.spawn_pipeline(steps)
    assert len(results) == 2
    for i, r in enumerate(results):
        assert r.get("error") is None, f"Step {i} failed: {r.get('error')}"
    print("PASS")


def test_pipeline_error_handling():
    """pipeline 中某步失败：该步 error，后续继续"""
    print("\n=== test_pipeline_error_handling ===")
    steps = [
        {"template": "nonexistent_xyz", "task": "this fails"},
        {"template": "summarizer", "task": "recover"},
    ]
    results = orch.spawn_pipeline(steps)
    assert len(results) == 2
    assert results[0].get("error") is not None
    assert results[1].get("error") is None
    print("PASS")


def test_pipeline_empty():
    """空 pipeline 返回空列表"""
    print("\n=== test_pipeline_empty ===")
    assert orch.spawn_pipeline([]) == []
    print("PASS")


def test_subagent_without_executor():
    """无 executor 的 SubAgent 获得空工具集"""
    print("\n=== test_subagent_without_executor ===")
    path = os.path.join(TEMPLATE_DIR, "summarizer.yaml")
    a = SubAgent(path)  # 不传 executor
    assert len(a.executor.names) == 0
    print("PASS")


# =============================================================================
if __name__ == "__main__":
    passed = failed = 0
    tests = [test_all_templates_loadable, test_tool_subset_correct,
             test_subagent_independent_llm, test_subagent_uses_talk_config,
             test_spawn_valid_template, test_spawn_nonexistent_template,
             test_spawn_parallel, test_pipeline_valid,
             test_pipeline_error_handling, test_pipeline_empty,
             test_subagent_without_executor]
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
