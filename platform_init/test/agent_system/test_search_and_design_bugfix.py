"""
针对 search_literature / design_experiment 工具的回归测试

覆盖两个 bug:
- search_literature: 之前用 r.get('title')/r.get('score'),但 SemanticSearch
  返回的是 pdf_name/similarity,导致全部显示"未知 (相关度: 0.00)"
- design_experiment: LLM 输出偶尔带 Python 字面量 / 单引号 / 尾随逗号,
  原 2 策略解析失败 → "_parse_experiment_json 增强为 4 策略"

运行: python platform_init/test/agent_system/test_search_and_design_bugfix.py
"""
import sys, io, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
from unittest.mock import patch, MagicMock

from core.field_inference import ExperimentDesignAgent
from core.agent_tools import _search_literature_func


# =============================================================================
# search_literature 修复验证
# =============================================================================

def test_search_uses_similarity_not_score():
    """_search_literature_func 必须从 SemanticSearch 的实际字段读取"""
    print("\n=== test_search_uses_similarity_not_score ===")
    # SemanticSearch.search() 实际返回的字段
    semantic_results = [
        {"page_id": "x1", "pdf_path": "/abs/p1.pdf",
         "pdf_name": "Perovskite Solar Cell 2024.pdf",
         "page_num": 0, "text_snippet": "x", "similarity": 0.85},
        {"page_id": "x2", "pdf_path": "/abs/p2.pdf",
         "pdf_name": "Annealing Process.pdf",
         "page_num": 0, "text_snippet": "y", "similarity": 0.42},
    ]

    fake_registry_rows = [
        ("Perovskite Solar Cell 2024.pdf", "Perovskite Solar Cells with Enhanced Stability via Annealing"),
        ("Annealing Process.pdf", "Two-Step Annealing Optimization for Perovskite Films"),
    ]

    fake_ss = MagicMock()
    fake_ss.search = MagicMock(return_value=semantic_results)

    with patch("extract.semantic_search.SemanticSearch", return_value=fake_ss), \
         patch("sqlite3.connect") as fake_sqlite:
        fake_conn = MagicMock()
        fake_conn.__enter__.return_value.execute.return_value.fetchall.return_value = fake_registry_rows
        fake_sqlite.return_value = fake_conn

        out = _search_literature_func({"query": "perovskite annealing", "top_k": 5})

    print(out)
    # 不应出现 "未知" 或 "0.00"
    assert "未知" not in out, f"still shows '未知' for known PDFs:\n{out}"
    assert "0.00" not in out, f"still shows '0.00' for valid similarity:\n{out}"
    # 实际标题应出现
    assert "Perovskite Solar Cells with Enhanced Stability via Annealing" in out
    assert "Two-Step Annealing Optimization for Perovskite Films" in out
    # 实际相似度 (0.85, 0.42) 应出现
    assert "0.85" in out
    assert "0.42" in out
    print("PASS")


def test_search_empty_results():
    """无结果时返回'未找到相关文献'"""
    print("\n=== test_search_empty_results ===")
    fake_ss = MagicMock()
    fake_ss.search = MagicMock(return_value=[])

    with patch("extract.semantic_search.SemanticSearch", return_value=fake_ss), \
         patch("sqlite3.connect") as fake_sqlite:
        fake_conn = MagicMock()
        fake_conn.__enter__.return_value.execute.return_value.fetchall.return_value = []
        fake_sqlite.return_value = fake_conn
        out = _search_literature_func({"query": "no match", "top_k": 5})

    assert "未找到" in out, f"Expected '未找到' message, got: {out!r}"
    print("PASS")


def test_search_falls_back_to_pdf_name():
    """注册表查不到时,fallback 到 pdf_name"""
    print("\n=== test_search_falls_back_to_pdf_name ===")
    semantic_results = [
        {"page_id": "x1", "pdf_path": "/abs/unindexed.pdf",
         "pdf_name": "unindexed.pdf",
         "page_num": 0, "text_snippet": "x", "similarity": 0.30},
    ]

    fake_ss = MagicMock()
    fake_ss.search = MagicMock(return_value=semantic_results)

    with patch("extract.semantic_search.SemanticSearch", return_value=fake_ss), \
         patch("os.path.isfile", return_value=False):  # DB 也不存在 → title_map 为空
        out = _search_literature_func({"query": "anything", "top_k": 5})

    # 0.30 = round(1.0 - 0.7, 4) 是 SemanticSearch 内部算的,这里用 0.30
    assert "0.30" in out, f"Expected '0.30' similarity, got: {out!r}"
    assert "unindexed.pdf" in out, f"Expected pdf_name fallback, got: {out!r}"
    assert "未知" not in out, f"Should not show '未知' when pdf_name is available:\n{out}"
    print("PASS")


# =============================================================================
# _parse_experiment_json 健壮性
# =============================================================================

def test_parse_pure_json():
    print("\n=== test_parse_pure_json ===")
    agent = ExperimentDesignAgent.__new__(ExperimentDesignAgent)
    content = '{"experiment_name": "x", "steps": []}'
    result = agent._parse_experiment_json(content)
    assert result == {"experiment_name": "x", "steps": []}
    print("PASS")


def test_parse_markdown_wrapped():
    print("\n=== test_parse_markdown_wrapped ===")
    agent = ExperimentDesignAgent.__new__(ExperimentDesignAgent)
    content = '下面是方案:\n```json\n{"a": 1}\n```\n完毕'
    result = agent._parse_experiment_json(content)
    assert result == {"a": 1}, f"got {result!r}"
    print("PASS")


def test_parse_python_literals():
    print("\n=== test_parse_python_literals ===")
    """策略3: 修复 Python 字面量 None/True/False → JSON null/true/false"""
    agent = ExperimentDesignAgent.__new__(ExperimentDesignAgent)
    # LLM 把 null/true/false 写成 Python 风格
    content = '{"name": "x", "active": True, "deleted": False, "data": None}'
    result = agent._parse_experiment_json(content)
    assert result == {"name": "x", "active": True, "deleted": False, "data": None}, f"got {result!r}"
    print("PASS")


def test_parse_trailing_comma():
    print("\n=== test_parse_trailing_comma ===")
    """策略3: 修复尾随逗号"""
    agent = ExperimentDesignAgent.__new__(ExperimentDesignAgent)
    content = '{"items": [1, 2, 3,], "name": "x",}'
    result = agent._parse_experiment_json(content)
    assert result == {"items": [1, 2, 3], "name": "x"}, f"got {result!r}"
    print("PASS")


def test_parse_outer_block():
    print("\n=== test_parse_outer_block ===")
    """策略2/4: LLM 在 JSON 前后加解释文字"""
    agent = ExperimentDesignAgent.__new__(ExperimentDesignAgent)
    content = '好的,我帮你设计:\n{"steps": [{"type": "tool", "name": "spin_coating", "params": {}}]}\n结束'
    result = agent._parse_experiment_json(content)
    assert result is not None
    assert result["steps"][0]["name"] == "spin_coating"
    print("PASS")


def test_parse_pure_text_returns_none():
    print("\n=== test_parse_pure_text_returns_none ===")
    """无 JSON 时返回 None(让上层返回错误)"""
    agent = ExperimentDesignAgent.__new__(ExperimentDesignAgent)
    result = agent._parse_experiment_json("抱歉,我无法设计这个实验。")
    assert result is None
    print("PASS")


def test_parse_empty_returns_none():
    print("\n=== test_parse_empty_returns_none ===")
    agent = ExperimentDesignAgent.__new__(ExperimentDesignAgent)
    assert agent._parse_experiment_json("") is None
    assert agent._parse_experiment_json(None) is None  # type: ignore[arg-type]
    print("PASS")


# =============================================================================
if __name__ == "__main__":
    passed = failed = 0
    tests = [
        test_search_uses_similarity_not_score,
        test_search_empty_results,
        test_search_falls_back_to_pdf_name,
        test_parse_pure_json,
        test_parse_markdown_wrapped,
        test_parse_python_literals,
        test_parse_trailing_comma,
        test_parse_outer_block,
        test_parse_pure_text_returns_none,
        test_parse_empty_returns_none,
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
