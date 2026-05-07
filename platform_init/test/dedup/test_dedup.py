"""
提取结果去重 — 功能测试

运行方法: python platform_init/test/dedup/test_dedup.py
"""
import sys
import io
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from extract.dedup import (
    deduplicate_extraction_results,
    _normalize_key,
    _merge_field_values,
    _is_non_empty,
)


def test_import():
    """验证模块导入"""
    print("\n=== test_import ===")
    assert callable(deduplicate_extraction_results)
    print("PASS")


def test_no_duplicates():
    """无重复数据直通"""
    print("\n=== test_no_duplicates ===")
    data = [
        {"钝化剂名称": "PEAI", "作用机理": "passivation", "_source_doc": "doc_a.pdf"},
        {"钝化剂名称": "BAI", "作用机理": "crystallization", "_source_doc": "doc_a.pdf"},
    ]
    fields = ["钝化剂名称", "作用机理", "文献来源"]
    result = deduplicate_extraction_results(data, fields)
    assert len(result) == 2
    assert result[0]["钝化剂名称"] == "BAI"  # 排序后 BAI < PEAI
    assert result[1]["钝化剂名称"] == "PEAI"
    print("PASS")


def test_exact_duplicates():
    """完全相同实体名合并为一条"""
    print("\n=== test_exact_duplicates ===")
    data = [
        {"钝化剂名称": "PEAI", "作用机理": "improves stability", "_source_doc": "doc_a.pdf"},
        {"钝化剂名称": "PEAI", "作用机理": "passivates defects", "_source_doc": "doc_a.pdf"},
    ]
    fields = ["钝化剂名称", "作用机理", "文献来源"]
    result = deduplicate_extraction_results(data, fields)
    assert len(result) == 1
    assert result[0]["钝化剂名称"] == "PEAI"
    assert result[0]["_occurrence_count"] == 2
    print("PASS")


def test_whitespace_normalization():
    """尾部空格规范化：'PEAI' 和 'PEAI ' 合并"""
    print("\n=== test_whitespace_normalization ===")
    data = [
        {"钝化剂名称": "PEAI", "作用机理": "A", "_source_doc": "doc_a.pdf"},
        {"钝化剂名称": "PEAI ", "作用机理": "B", "_source_doc": "doc_b.pdf"},
    ]
    fields = ["钝化剂名称", "作用机理", "文献来源"]
    result = deduplicate_extraction_results(data, fields, normalize="strip")
    assert len(result) == 1
    assert result[0]["钝化剂名称"] == "PEAI"
    assert result[0]["_occurrence_count"] == 2
    print("PASS")


def test_case_normalization():
    """大小写规范化：'PEAI' 和 'peai' 合并"""
    print("\n=== test_case_normalization ===")
    data = [
        {"钝化剂名称": "PEAI", "作用机理": "A", "_source_doc": "doc_a.pdf"},
        {"钝化剂名称": "peai", "作用机理": "B", "_source_doc": "doc_b.pdf"},
    ]
    fields = ["钝化剂名称", "作用机理", "文献来源"]
    result = deduplicate_extraction_results(data, fields, normalize="lower")
    assert len(result) == 1
    assert result[0]["_occurrence_count"] == 2
    print("PASS")


def test_strict_no_merge():
    """strict 模式：'PEAI' 和 'PEAI ' 不合并"""
    print("\n=== test_strict_no_merge ===")
    data = [
        {"钝化剂名称": "PEAI", "作用机理": "A", "_source_doc": "doc_a.pdf"},
        {"钝化剂名称": "PEAI ", "作用机理": "B", "_source_doc": "doc_b.pdf"},
    ]
    fields = ["钝化剂名称", "作用机理", "文献来源"]
    result = deduplicate_extraction_results(data, fields, normalize="strict")
    assert len(result) == 2
    print("PASS")


def test_longest_merge_strategy():
    """longest 策略：取最长非空值"""
    print("\n=== test_longest_merge_strategy ===")
    data = [
        {"钝化剂名称": "PEAI", "作用机理": "improves", "_source_doc": "doc_a.pdf"},
        {"钝化剂名称": "PEAI", "作用机理": "improves crystallinity significantly", "_source_doc": "doc_a.pdf"},
    ]
    fields = ["钝化剂名称", "作用机理", "文献来源"]
    result = deduplicate_extraction_results(data, fields, merge_strategy="longest")
    assert len(result) == 1
    assert result[0]["作用机理"] == "improves crystallinity significantly"
    print("PASS")


def test_first_non_empty_merge():
    """first_non_empty 策略：取第一个非空值"""
    print("\n=== test_first_non_empty_merge ===")
    data = [
        {"钝化剂名称": "PEAI", "作用机理": "", "_source_doc": "doc_a.pdf"},
        {"钝化剂名称": "PEAI", "作用机理": "passivates", "_source_doc": "doc_b.pdf"},
    ]
    fields = ["钝化剂名称", "作用机理", "文献来源"]
    result = deduplicate_extraction_results(data, fields, merge_strategy="first_non_empty")
    assert len(result) == 1
    assert result[0]["作用机理"] == "passivates"
    print("PASS")


def test_source_doc_tracking():
    """多文档来源合并 _source_doc"""
    print("\n=== test_source_doc_tracking ===")
    data = [
        {"钝化剂名称": "PEAI", "作用机理": "A", "_source_doc": "nature.pdf"},
        {"钝化剂名称": "PEAI", "作用机理": "B", "_source_doc": "science.pdf"},
    ]
    fields = ["钝化剂名称", "作用机理", "文献来源"]
    result = deduplicate_extraction_results(data, fields)
    assert len(result) == 1
    assert "nature.pdf" in result[0]["_source_doc"]
    assert "science.pdf" in result[0]["_source_doc"]
    assert result[0]["_source_docs"] == "nature.pdf; science.pdf"
    print("PASS")


def test_occurrence_count():
    """出现次数正确统计"""
    print("\n=== test_occurrence_count ===")
    data = [
        {"钝化剂名称": "PEAI", "_source_doc": "doc_a.pdf"},
        {"钝化剂名称": "PEAI", "_source_doc": "doc_a.pdf"},
        {"钝化剂名称": "PEAI", "_source_doc": "doc_b.pdf"},
    ]
    fields = ["钝化剂名称"]
    result = deduplicate_extraction_results(data, fields)
    assert len(result) == 1
    assert result[0]["_occurrence_count"] == 3
    # _source_docs 去重，只有 2 个文档
    assert result[0]["_source_docs"] == "doc_a.pdf; doc_b.pdf"
    print("PASS")


def test_empty_key_skipped():
    """空实体名被丢弃"""
    print("\n=== test_empty_key_skipped ===")
    data = [
        {"钝化剂名称": "PEAI", "作用机理": "A", "_source_doc": "doc_a.pdf"},
        {"钝化剂名称": "", "作用机理": "B", "_source_doc": "doc_b.pdf"},
        {"钝化剂名称": "   ", "作用机理": "C", "_source_doc": "doc_c.pdf"},
    ]
    fields = ["钝化剂名称", "作用机理", "文献来源"]
    result = deduplicate_extraction_results(data, fields)
    assert len(result) == 1
    assert result[0]["钝化剂名称"] == "PEAI"
    print("PASS")


def test_metadata_disabled():
    """add_metadata=False 不添加元数据列"""
    print("\n=== test_metadata_disabled ===")
    data = [
        {"钝化剂名称": "PEAI", "_source_doc": "doc_a.pdf"},
        {"钝化剂名称": "PEAI", "_source_doc": "doc_b.pdf"},
    ]
    fields = ["钝化剂名称"]
    result = deduplicate_extraction_results(data, fields, add_metadata=False)
    assert len(result) == 1
    assert "_occurrence_count" not in result[0]
    assert "_source_docs" not in result[0]
    print("PASS")


def test_missing_field_handling():
    """部分字段缺失时安全处理"""
    print("\n=== test_missing_field_handling ===")
    data = [
        {"钝化剂名称": "PEAI", "作用机理": "A", "_source_doc": "doc_a.pdf"},
        {"钝化剂名称": "PEAI", "文献来源": "ref1", "_source_doc": "doc_b.pdf"},
    ]
    fields = ["钝化剂名称", "作用机理", "文献来源"]
    result = deduplicate_extraction_results(data, fields)
    assert len(result) == 1
    assert result[0]["作用机理"] == "A"
    assert result[0]["文献来源"] == "ref1"
    print("PASS")


def test_multiple_entities():
    """多个不同实体各自保留独立行"""
    print("\n=== test_multiple_entities ===")
    data = [
        {"钝化剂名称": "PEAI", "_source_doc": "doc_a.pdf"},
        {"钝化剂名称": "PEAI", "_source_doc": "doc_a.pdf"},
        {"钝化剂名称": "BAI", "_source_doc": "doc_b.pdf"},
    ]
    fields = ["钝化剂名称"]
    result = deduplicate_extraction_results(data, fields)
    assert len(result) == 2
    assert result[0]["钝化剂名称"] == "BAI"  # 排序后 BAI < PEAI
    assert result[0]["_occurrence_count"] == 1
    assert result[1]["钝化剂名称"] == "PEAI"
    assert result[1]["_occurrence_count"] == 2
    print("PASS")


def test_empty_input():
    """空输入返回空列表"""
    print("\n=== test_empty_input ===")
    result = deduplicate_extraction_results([], [])
    assert result == []
    result2 = deduplicate_extraction_results([], ["钝化剂名称"])
    assert result2 == []
    print("PASS")


def test_dynamic_entity_key():
    """自定义字段列表：fields[0] 作为去重键"""
    print("\n=== test_dynamic_entity_key ===")
    data = [
        {"作者": "张三", "单位": "清华大学", "_source_doc": "doc_a.pdf"},
        {"作者": "张三", "单位": "Tsinghua University", "_source_doc": "doc_b.pdf"},
        {"作者": "李四", "单位": "北京大学", "_source_doc": "doc_c.pdf"},
    ]
    fields = ["作者", "单位"]
    result = deduplicate_extraction_results(data, fields)
    assert len(result) == 2
    assert result[0]["作者"] == "张三"
    assert result[0]["_occurrence_count"] == 2
    assert result[1]["作者"] == "李四"
    assert result[1]["_occurrence_count"] == 1
    print("PASS")


def test_helper_normalize_key():
    """_normalize_key 单元测试"""
    print("\n=== test_helper_normalize_key ===")
    assert _normalize_key("PEAI ", "strip") == "PEAI"
    assert _normalize_key(" PEAI", "strip") == "PEAI"
    assert _normalize_key("PEAI", "lower") == "peai"
    assert _normalize_key(" PeAi ", "lower") == "peai"
    assert _normalize_key("PEAI ", "strict") == "PEAI "
    print("PASS")


def test_helper_merge_field_values():
    """_merge_field_values 单元测试"""
    print("\n=== test_helper_merge_field_values ===")
    assert _merge_field_values(["", "good", "better"], "longest") == "better"
    assert _merge_field_values(["", "good", "better"], "first_non_empty") == "good"
    assert _merge_field_values(["", "", ""], "longest") == ""
    assert _merge_field_values([], "longest") == ""
    print("PASS")


def test_helper_is_non_empty():
    """_is_non_empty 单元测试"""
    print("\n=== test_helper_is_non_empty ===")
    assert _is_non_empty("hello") is True
    assert _is_non_empty("") is False
    assert _is_non_empty("   ") is False
    assert _is_non_empty(None) is False
    assert _is_non_empty(0) is True  # 0 是有效数值
    print("PASS")


def test_real_world_scenario():
    """模拟真实场景：多页提取同一PDF中PEAI出现多次"""
    print("\n=== test_real_world_scenario ===")
    data = [
        {"钝化剂名称": "PEAI", "原文原句": "PEAI effectively passivates the surface defects of FAPbI3 perovskite films.", "作用机理": "表面缺陷钝化", "_source_doc": "nature_articles_s41467-019-10985-5"},
        {"钝化剂名称": "PEAI", "原文原句": "The addition of PEAI significantly improved the open-circuit voltage from 1.02 V to 1.15 V.", "作用机理": "提高开路电压", "_source_doc": "nature_articles_s41467-019-10985-5"},
        {"钝化剂名称": "PEAI ", "原文原句": "", "作用机理": "", "_source_doc": "nature_articles_s41467-019-10985-5"},
        {"钝化剂名称": "BAI", "原文原句": "BAI treatment resulted in better film morphology.", "作用机理": "改善薄膜形貌", "_source_doc": "nature_articles_s41467-019-10985-5"},
        {"钝化剂名称": "PEAI", "原文原句": "PEAI was also studied in another context.", "作用机理": "表面缺陷钝化", "_source_doc": "science_advances_2024"},
    ]
    fields = ["钝化剂名称", "原文原句", "作用机理", "文献来源"]
    result = deduplicate_extraction_results(data, fields, normalize="strip", merge_strategy="longest")

    # 应该有 PEAI 和 BAI 两条
    assert len(result) == 2

    peai = [r for r in result if r["钝化剂名称"] == "PEAI"][0]
    bai = [r for r in result if r["钝化剂名称"] == "BAI"][0]

    # PEAI 出现 4 次（3次在同一文档，1次在另一文档）
    assert peai["_occurrence_count"] == 4
    # 原文原句取最长的（第2条最长，92字符）
    assert "significantly improved" in peai["原文原句"]
    # BAI 只出现 1 次
    assert bai["_occurrence_count"] == 1
    assert "film morphology" in bai["原文原句"]
    # _source_doc 合并了2个文档
    assert "nature_articles" in peai["_source_doc"]
    assert "science_advances" in peai["_source_doc"]
    print("PASS")


if __name__ == "__main__":
    tests = [
        test_import,
        test_no_duplicates,
        test_exact_duplicates,
        test_whitespace_normalization,
        test_case_normalization,
        test_strict_no_merge,
        test_longest_merge_strategy,
        test_first_non_empty_merge,
        test_source_doc_tracking,
        test_occurrence_count,
        test_empty_key_skipped,
        test_metadata_disabled,
        test_missing_field_handling,
        test_multiple_entities,
        test_empty_input,
        test_dynamic_entity_key,
        test_helper_normalize_key,
        test_helper_merge_field_values,
        test_helper_is_non_empty,
        test_real_world_scenario,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*40}")
    print(f"结果: {passed} pass, {failed} fail")
    print(f"{'='*40}")
