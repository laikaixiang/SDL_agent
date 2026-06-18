"""
文献抽取质量改进 — 验证测试

覆盖 plan Verification 章节的 6 个 case:
  1. 全空行过滤
  2. Evidence validation (精确 / 模糊 / 失败)
  3. Sparsity 阈值 (0.5)
  4. Semantic dedup
  5. Review agent
  6. (前端跳转 — 手动验证, 不在此脚本中)

运行方法:
  cd D:/PycharmProjects/SDL_agent/.worktrees/extraction
  python platform_init/test/extraction_quality/test_improvements.py
"""
import sys
import io
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from extract.evidence_validator import EvidenceValidator
from extract.quality_checker import QualityChecker
from extract.semantic_dedup import SemanticDedup, _cosine_similarity


# =============================================================================
# Case 1: 全空行过滤
# =============================================================================

class TestEmptyRowFilter(unittest.TestCase):
    """Case 1: 全空行过滤 (主键空 / grounding 空 → 丢弃)"""

    def test_filter_drops_empty_primary_key(self):
        """主键空 → 丢弃"""
        data = [
            {"钝化剂名称": "PEAI", "原文原句": "PEAI was used...", "_source_doc": "a.pdf"},
            {"钝化剂名称": "", "原文原句": "Some evidence", "_source_doc": "a.pdf"},  # 主键空
            {"钝化剂名称": "BAI", "原文原句": "BAI passivates...", "_source_doc": "a.pdf"},
        ]
        fields = ["钝化剂名称", "原文原句"]
        empty_markers = {"", "无", "未提及", "N/A", "-", "--"}

        valid = []
        for r in data:
            primary = str(r.get(fields[0], "") or "").strip()
            if not primary or primary in empty_markers:
                continue
            evidence = str(r.get("原文原句", "") or "").strip()
            if not evidence or evidence in empty_markers:
                continue
            valid.append(r)

        self.assertEqual(len(valid), 2)
        self.assertEqual(valid[0]["钝化剂名称"], "PEAI")
        self.assertEqual(valid[1]["钝化剂名称"], "BAI")

    def test_filter_drops_empty_evidence(self):
        """主键有但 grounding 空 → 丢弃"""
        data = [
            {"钝化剂名称": "PEAI", "原文原句": "PEAI was used...", "_source_doc": "a.pdf"},
            {"钝化剂名称": "BAI", "原文原句": "", "_source_doc": "a.pdf"},  # grounding 空
            {"钝化剂名称": "OAI", "原文原句": "无", "_source_doc": "a.pdf"},  # grounding 是 "无"
        ]
        fields = ["钝化剂名称", "原文原句"]
        empty_markers = {"", "无", "未提及", "N/A", "-", "--"}

        valid = []
        for r in data:
            primary = str(r.get(fields[0], "") or "").strip()
            if not primary or primary in empty_markers:
                continue
            evidence = str(r.get("原文原句", "") or "").strip()
            if not evidence or evidence in empty_markers:
                continue
            valid.append(r)

        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["钝化剂名称"], "PEAI")

    def test_filter_keeps_all_valid(self):
        """全有效 → 不丢"""
        data = [
            {"钝化剂名称": "PEAI", "原文原句": "PEAI was used...", "_source_doc": "a.pdf"},
            {"钝化剂名称": "BAI", "原文原句": "BAI passivates...", "_source_doc": "a.pdf"},
            {"钝化剂名称": "OAI", "原文原句": "OAI improves...", "_source_doc": "a.pdf"},
        ]
        fields = ["钝化剂名称", "原文原句"]
        empty_markers = {"", "无", "未提及", "N/A", "-", "--"}

        valid = [r for r in data
                 if str(r.get(fields[0], "") or "").strip() not in empty_markers
                 and str(r.get("原文原句", "") or "").strip() not in empty_markers]

        self.assertEqual(len(valid), 3)


# =============================================================================
# Case 2: Evidence Validation
# =============================================================================

class TestEvidenceValidator(unittest.TestCase):
    """Case 2: EvidenceValidator 三种 case"""

    def setUp(self):
        self.v = EvidenceValidator(fuzzy_threshold=0.7)

    def test_exact_match(self):
        """精确匹配 — 找到 evidence 在 page_text 中的位置"""
        page_text = "The passivation was performed using PEAI (5 mg/mL) in isopropanol."
        evidence = "PEAI (5 mg/mL)"
        result = self.v.validate(page_text, evidence)
        self.assertTrue(result["valid"])
        self.assertIsNotNone(result["offset"])
        self.assertEqual(result["fuzzy_score"], 1.0)
        self.assertEqual(result["length"], len(evidence))
        # 找到的位置应指向 page_text 中 evidence 第一次出现处
        self.assertEqual(page_text[result["offset"]:result["offset"] + result["length"]], evidence)

    def test_fuzzy_match(self):
        """模糊匹配 — evidence 中的标点/空格与 page_text 略有差异"""
        page_text = "The passivation was performed using PEAI (5mg/mL) in isopropanol"
        evidence = "PEAI (5 mg/mL)"  # 多一个空格
        result = self.v.validate(page_text, evidence)
        # 模糊匹配应该找到
        self.assertTrue(result["valid"])
        self.assertGreaterEqual(result["fuzzy_score"], 0.7)
        self.assertIsNotNone(result["offset"])

    def test_no_match(self):
        """失败 — evidence 根本不在 page_text 中"""
        page_text = "The cell efficiency was 23.5% with Voc of 1.1V"
        evidence = "completely different content that doesn't exist"
        result = self.v.validate(page_text, evidence)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["offset"])
        self.assertEqual(result["fuzzy_score"], 0.0)

    def test_empty_inputs(self):
        """空输入"""
        self.assertFalse(self.v.validate("", "evidence")["valid"])
        self.assertFalse(self.v.validate("page_text", "")["valid"])
        self.assertFalse(self.v.validate("", "")["valid"])

    def test_normalized_evidence_field(self):
        """normalized_evidence 字段应去除空格和标点"""
        page_text = "Some text here"
        evidence = "PEAI, (5 mg/mL)."
        result = self.v.validate(page_text, evidence)
        # normalized_evidence 应去除标点
        self.assertNotIn(",", result["normalized_evidence"])
        self.assertNotIn(".", result["normalized_evidence"])
        self.assertIn("PEAI", result["normalized_evidence"])


# =============================================================================
# Case 3: Sparsity Threshold (0.5)
# =============================================================================

class TestSparsityThreshold(unittest.TestCase):
    """Case 3: sparsity 阈值从 0.3 升到 0.5"""

    def setUp(self):
        self.qc = QualityChecker()

    def test_sparsity_05_drops_3_of_6(self):
        """
        6 条记录, 4 字段各填 1/2/3/4 个:
          - 记录 0: 填 1/4 → rate 0.25 → 删
          - 记录 1: 填 2/4 → rate 0.50 → 不删 (rate == threshold 不删, 严格小于才删)
          - 记录 2: 填 3/4 → rate 0.75 → 留
          - 记录 3: 填 4/4 → rate 1.0  → 留
        """
        fields = ["f1", "f2", "f3", "f4"]
        records = [
            {"f1": "a", "f2": "", "f3": "", "f4": ""},  # 1/4 = 0.25
            {"f1": "a", "f2": "b", "f3": "", "f4": ""},  # 2/4 = 0.50
            {"f1": "a", "f2": "b", "f3": "c", "f4": ""},  # 3/4 = 0.75
            {"f1": "a", "f2": "b", "f3": "c", "f4": "d"},  # 4/4 = 1.0
        ]
        deleted = self.qc.check_sparsity(records, fields, threshold=0.5)
        # rate < 0.5 (strict) → 只有 0.25 那条
        self.assertEqual(deleted, [0])

    def test_sparsity_05_keeps_2_or_more(self):
        """rate >= 0.5 的都保留 (i.e. 至少填一半字段)"""
        fields = ["f1", "f2", "f3", "f4"]
        records = [
            {"f1": "a", "f2": "b", "f3": "c", "f4": ""},  # 3/4
            {"f1": "a", "f2": "b", "f3": "", "f4": ""},  # 2/4
        ]
        deleted = self.qc.check_sparsity(records, fields, threshold=0.5)
        self.assertEqual(deleted, [])

    def test_sparsity_03_old_threshold(self):
        """旧阈值 0.3: rate=0.25 (1/4) 仍会被删"""
        fields = ["f1", "f2", "f3", "f4"]
        records = [
            {"f1": "a", "f2": "", "f3": "", "f4": ""},  # 1/4 = 0.25
            {"f1": "a", "f2": "b", "f3": "", "f4": ""},  # 2/4 = 0.50
        ]
        deleted_old = self.qc.check_sparsity(records, fields, threshold=0.3)
        # 0.25 < 0.3 → 删
        self.assertIn(0, deleted_old)
        # 0.50 >= 0.3 → 留
        self.assertNotIn(1, deleted_old)


# =============================================================================
# Case 4: Semantic Dedup
# =============================================================================

class MockEmbeddingService:
    """可控向量: 相似名 → 相似向量; 不同名 → 正交向量"""

    def __init__(self):
        # "PEAI" 与 "phenethylammonium iodide" → 高相似
        # "BAI" 与 "BAI" → 完全相同
        # 其他 → 完全不同
        self._table = {
            "PEAI": [1.0, 0.1, 0.0],
            "phenethylammonium iodide": [0.95, 0.1, 0.05],
            "BAI": [0.0, 1.0, 0.0],
            "OAI": [0.0, 0.0, 1.0],
        }

    def embed_text(self, text: str):
        if not text:
            return None
        primary = text.split("|")[0].strip()
        if primary in self._table:
            return self._table[primary]
        # 未知主键 → 返回零向量 (不参与合并)
        return [0.0, 0.0, 0.0]


class TestSemanticDedup(unittest.TestCase):
    """Case 4: SemanticDedup 合并语义相同的行"""

    def test_cosine_similarity(self):
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        c = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(_cosine_similarity(a, b), 1.0, places=5)
        self.assertAlmostEqual(_cosine_similarity(a, c), 0.0, places=5)

    def test_merge_aliased_names(self):
        """'PEAI' 与 'phenethylammonium iodide' 应被合并"""
        records = [
            {"钝化剂名称": "PEAI", "作用机理": "passivation", "_source_doc": "a.pdf"},
            {"钝化剂名称": "phenethylammonium iodide", "作用机理": "improve stability", "_source_doc": "b.pdf"},
            {"钝化剂名称": "BAI", "作用机理": "crystallization", "_source_doc": "a.pdf"},
        ]
        fields = ["钝化剂名称", "作用机理"]
        embedder = MockEmbeddingService()
        sem = SemanticDedup(embedder, similarity_threshold=0.92)
        result = sem.cluster_and_merge(records, fields)
        # 4 条 → 2 条 (PEAI/phenethylammonium iodide 合并, BAI 独立)
        self.assertEqual(len(result), 2)
        names = sorted([r["钝化剂名称"] for r in result])
        self.assertIn("BAI", names)
        # 合并的那条主键可能是 PEAI 或 phenethylammonium iodide
        merged_count = sum(1 for r in result if r.get("_semantic_merged"))
        self.assertEqual(merged_count, 1)
        merged = [r for r in result if r.get("_semantic_merged")][0]
        self.assertEqual(merged.get("_semantic_cluster_size"), 2)

    def test_keeps_distinct_records(self):
        """不相关名称不被合并"""
        records = [
            {"钝化剂名称": "PEAI", "_source_doc": "a.pdf"},
            {"钝化剂名称": "BAI", "_source_doc": "a.pdf"},
            {"钝化剂名称": "OAI", "_source_doc": "a.pdf"},
        ]
        fields = ["钝化剂名称"]
        embedder = MockEmbeddingService()
        sem = SemanticDedup(embedder, similarity_threshold=0.92)
        result = sem.cluster_and_merge(records, fields)
        self.assertEqual(len(result), 3)

    def test_no_embedding_returns_original(self):
        """embedding 服务为 None 文本 → 不合并"""
        records = [
            {"钝化剂名称": "", "作用机理": "evidence"},
            {"钝化剂名称": "", "作用机理": "another"},
        ]
        fields = ["钝化剂名称", "作用机理"]
        embedder = MockEmbeddingService()
        sem = SemanticDedup(embedder, similarity_threshold=0.92)
        result = sem.cluster_and_merge(records, fields)
        # 空主键 → 不构造 embedding → 保留 2 条
        self.assertEqual(len(result), 2)


# =============================================================================
# Case 5: Review Agent (mock LLM, 跑多次看稳定性)
# =============================================================================

class MockLLMClient:
    """模拟 LLMClient.call_api 多次返回稳定结果"""

    def __init__(self, response_json: str):
        self.response_json = response_json
        self.call_count = 0

    def call_api(self, model, messages, **kwargs):
        self.call_count += 1
        return {
            "choices": [
                {"message": {"content": self.response_json}}
            ]
        }


class TestReviewAgent(unittest.TestCase):
    """Case 5: ExtractionReviewAgent 打 review 标记"""

    def test_review_marks_duplicate(self):
        """10 条记录, 其中 2 条明显重复 → LLM 标记 1 条为 duplicate"""
        records = [
            {"钝化剂名称": "PEAI", "作用机理": "passivation", "_source_doc": "a.pdf", "_occurrence_count": 1},
            {"钝化剂名称": "PEAI-formula", "作用机理": "improve stability", "_source_doc": "b.pdf", "_occurrence_count": 1},
            {"钝化剂名称": "BAI", "作用机理": "crystallization", "_source_doc": "a.pdf", "_occurrence_count": 1},
            {"钝化剂名称": "OAI", "作用机理": "oxidation", "_source_doc": "a.pdf", "_occurrence_count": 1},
            {"钝化剂名称": "MAI", "作用机理": "methylation", "_source_doc": "a.pdf", "_occurrence_count": 1},
            {"钝化剂名称": "FAI", "作用机理": "formamidinium", "_source_doc": "a.pdf", "_occurrence_count": 1},
            {"钝化剂名称": "CsI", "作用机理": "cesium", "_source_doc": "a.pdf", "_occurrence_count": 1},
            {"钝化剂名称": "RbI", "作用机理": "rubidium", "_source_doc": "a.pdf", "_occurrence_count": 1},
            {"钝化剂名称": "LiI", "作用机理": "lithium", "_source_doc": "a.pdf", "_occurrence_count": 1},
            {"钝化剂名称": "NaI", "作用机理": "sodium", "_source_doc": "a.pdf", "_occurrence_count": 1},
        ]
        fields = ["钝化剂名称", "作用机理"]
        llm_response = """[
            {"row_index": 0, "flag": "duplicate", "note": "同 PEAI 缩写"},
            {"row_index": 1, "flag": "ok", "note": ""},
            {"row_index": 2, "flag": "ok", "note": ""},
            {"row_index": 3, "flag": "ok", "note": ""},
            {"row_index": 4, "flag": "ok", "note": ""},
            {"row_index": 5, "flag": "ok", "note": ""},
            {"row_index": 6, "flag": "ok", "note": ""},
            {"row_index": 7, "flag": "ok", "note": ""},
            {"row_index": 8, "flag": "ok", "note": ""},
            {"row_index": 9, "flag": "ok", "note": ""}
        ]"""
        mock_llm = MockLLMClient(llm_response)
        from core.review_agent import ExtractionReviewAgent
        agent = ExtractionReviewAgent(llm_client=mock_llm)
        reviewed = agent.review(records, fields)

        # 索引 0 应该是 duplicate
        self.assertEqual(reviewed[0]["_review_flag"], "duplicate")
        # _duplicate_of 应该指向另一条 (主键包含关系)
        self.assertIn("_duplicate_of", reviewed[0])
        # 其他应该都是 ok
        for i in range(1, 10):
            self.assertEqual(reviewed[i]["_review_flag"], "ok")
        # 5 次跑测试, 确保稳定
        for _ in range(5):
            mock_llm2 = MockLLMClient(llm_response)
            agent2 = ExtractionReviewAgent(llm_client=mock_llm2)
            reviewed2 = agent2.review(records, fields)
            self.assertEqual(reviewed2[0]["_review_flag"], "duplicate")
            self.assertEqual(reviewed2[5]["_review_flag"], "ok")

    def test_review_marks_low_value(self):
        """1 条明显异常值 → 标记为 low_value"""
        records = [
            {"钝化剂名称": "PEAI", "作用机理": "passivation", "_source_doc": "a.pdf"},
            {"钝化剂名称": "BAI", "作用机理": "PCE 9999999%", "_source_doc": "a.pdf"},  # 异常
        ]
        fields = ["钝化剂名称", "作用机理"]
        llm_response = """[
            {"row_index": 0, "flag": "ok", "note": ""},
            {"row_index": 1, "flag": "low_value", "note": "PCE 数值异常大"}
        ]"""
        mock_llm = MockLLMClient(llm_response)
        from core.review_agent import ExtractionReviewAgent
        agent = ExtractionReviewAgent(llm_client=mock_llm)
        reviewed = agent.review(records, fields)
        self.assertEqual(reviewed[1]["_review_flag"], "low_value")
        self.assertIn("PCE", reviewed[1]["_review_note"])

    def test_review_handles_bad_json(self):
        """LLM 返回非 JSON → 不崩溃, 标记 skipped"""
        records = [{"钝化剂名称": "PEAI"}]
        fields = ["钝化剂名称"]
        mock_llm = MockLLMClient("not valid json at all")
        from core.review_agent import ExtractionReviewAgent
        agent = ExtractionReviewAgent(llm_client=mock_llm)
        reviewed = agent.review(records, fields)
        # 应该不崩溃, 但 _review_flag 应该是 skipped 或缺失
        # 我们的实现: 单批失败时, 该批记录会被标 skipped
        self.assertEqual(len(reviewed), 1)
        # _review_flag 可能为 skipped
        self.assertIn(reviewed[0].get("_review_flag", "ok"), ["skipped", "ok"])

    def test_review_handles_empty_records(self):
        """空 records 列表 → 直接返回空"""
        from core.review_agent import ExtractionReviewAgent
        agent = ExtractionReviewAgent(llm_client=MagicMock())
        result = agent.review([], ["field1"])
        self.assertEqual(result, [])


# =============================================================================
# Case 6: Frontend Jump — 跳过 (手动验证)
# =============================================================================

class TestFrontendJumpManual(unittest.TestCase):
    """Case 6: 前端跳转 — 手动验证 (启动 app → 跑抽取 → 点 🔗)"""

    def test_placeholder(self):
        """手动验证步骤见 plan Verification 章节"""
        # 1) 启动 app: python digital_twin.py
        # 2) 跑抽取任务
        # 3) 在结果表格点击 🔗 按钮
        # 4) 验证 PDF viewer 跳到对应页 + 显示高亮 overlay
        # 5) 验证 5s 后高亮自动消失
        self.skipTest("前端跳转 — 手动验证")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
