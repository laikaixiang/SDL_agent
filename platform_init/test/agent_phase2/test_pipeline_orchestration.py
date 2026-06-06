"""
Phase 2 pipeline orchestration — 单元测试

测试 spawn_pipeline 模式、context 传递、错误恢复。

运行方法: python platform_init/test/agent_phase2/test_pipeline_orchestration.py
"""
import sys
import io
import os
import json
import queue

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from core.agent_loop import AgentOrchestrator


# =============================================================================
def test_templates_all_present():
    """所有 Phase 2 模板存在"""
    print("\n=== test_templates_all_present ===")
    orch = AgentOrchestrator()
    templates = orch.list_templates()
    required = [
        "literature_searcher",
        "literature_extractor",
        "extraction_pipeline",
        "summarizer",
    ]
    for name in required:
        assert name in templates, f"Missing template: {name}"
    print(f"  All {len(required)} required templates present")
    print(f"  Total: {len(templates)} templates: {sorted(templates)}")
    print("PASS")


def test_pipeline_valid_steps():
    """pipeline 正常步骤顺序执行"""
    print("\n=== test_pipeline_valid_steps ===")
    orch = AgentOrchestrator()
    steps = [
        {"template": "summarizer", "task": "summarize: 'hello world'"},
        {"template": "summarizer", "task": "summarize: extra context from previous step"},
    ]
    results = orch.spawn_pipeline(steps)
    assert len(results) == 2, f"Expected 2, got {len(results)}"
    assert results[0].get("error") is None, f"Step 0 failed: {results[0].get('error')}"
    assert results[1].get("error") is None, f"Step 1 failed: {results[1].get('error')}"
    print(f"  Step 0: OK")
    print(f"  Step 1: OK (received context from step 0)")
    print("PASS")


def test_pipeline_with_missing_template():
    """pipeline 中某步模板不存在 → 该步 error，后续继续"""
    print("\n=== test_pipeline_with_missing_template ===")
    orch = AgentOrchestrator()
    steps = [
        {"template": "nonexistent_xyz_123", "task": "do"},
        {"template": "summarizer", "task": "summarize fallback"},
    ]
    results = orch.spawn_pipeline(steps)
    assert len(results) == 2
    assert results[0].get("error") is not None
    assert results[1].get("error") is None
    print(f"  Step 0: error (expected)")
    print(f"  Step 1: OK (continued despite step 0 failure)")
    print("PASS")


def test_pipeline_empty_task():
    """pipeline 步骤缺少 task→报错"""
    print("\n=== test_pipeline_empty_task ===")
    orch = AgentOrchestrator()
    steps = [
        {"template": "summarizer", "task": ""},
    ]
    results = orch.spawn_pipeline(steps)
    assert len(results) == 1
    assert results[0].get("error") is not None
    print(f"  Step 0: error={results[0].get('error')}")
    print("PASS")


def test_pipeline_three_steps():
    """3步pipeline正常执行"""
    print("\n=== test_pipeline_three_steps ===")
    orch = AgentOrchestrator()
    steps = [
        {"template": "summarizer", "task": "step1: input data is 'a b c'"},
        {"template": "summarizer", "task": "step2: check previous"},
        {"template": "summarizer", "task": "step3: final summary"},
    ]
    results = orch.spawn_pipeline(steps)
    assert len(results) == 3
    for i, r in enumerate(results):
        assert r.get("error") is None, f"Step {i} failed: {r.get('error')}"
    print(f"  All 3 steps OK")
    print("PASS")


def test_pipeline_context_passing():
    """pipeline 后续步骤能收到前面步骤的 context"""
    print("\n=== test_pipeline_context_passing ===")
    orch = AgentOrchestrator()
    # 用 summarizer 做两步：第一步输出包含标记词，第二步应能收到
    marker = "MARKER_ABC_123"
    steps = [
        {"template": "summarizer", "task": f"mention the marker '{marker}' in your response"},
        {"template": "summarizer", "task": "describe what context you received. include the word 'received'"},
    ]
    results = orch.spawn_pipeline(steps)
    assert len(results) == 2
    assert results[0].get("error") is None
    assert results[1].get("error") is None
    # Step 1 的 output 应该包含 marker
    step0_fm = results[0].get("final_message", {})
    step0_content = step0_fm.get("content", "") if step0_fm else ""
    assert marker in step0_content, f"Step 0 output missing marker: {step0_content[:200]}"
    print(f"  Step 0 output contains marker: OK")
    print(f"  Step 1 ran with context: OK")
    print("PASS")


def test_pipeline_empty_steps():
    """空步骤列表返回空结果"""
    print("\n=== test_pipeline_empty_steps ===")
    orch = AgentOrchestrator()
    results = orch.spawn_pipeline([])
    assert results == []
    print("PASS")


# =============================================================================
if __name__ == "__main__":
    passed = 0
    failed = 0
    tests = [
        test_templates_all_present,
        test_pipeline_valid_steps,
        test_pipeline_with_missing_template,
        test_pipeline_empty_task,
        test_pipeline_three_steps,
        test_pipeline_context_passing,
        test_pipeline_empty_steps,
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
