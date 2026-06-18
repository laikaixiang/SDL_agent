"""
Evidence Validator — 验证 LLM 抽取的"原文原句"是否真实出现在该页 PDF 文本中。

两阶段匹配:
  1) 精确匹配   — page_text.find(evidence)
  2) 模糊匹配   — 去除空格/标点后 find，得分 0.85

返回字段:
  - valid: bool
  - offset: int | None      (在 page_text 中的字符偏移)
  - length: int | None      (匹配长度)
  - fuzzy_score: float      (0-1, 1.0=精确)
  - normalized_evidence: str (规范化后文本)
"""

import re
from typing import Dict, Any, Optional


# 用于模糊匹配的字符: 空格、各种中英文标点
_FUZZY_STRIP_RE = re.compile(r"[\s,;。，；、\.!?:：!?,;'\"]")


def _normalize(text: str) -> str:
    """去除所有空白/标点，便于模糊匹配。"""
    if not text:
        return ""
    return _FUZZY_STRIP_RE.sub("", text)


class EvidenceValidator:
    """验证 LLM 抽取的"原文原句"是否真实出现在该页 PDF 文本中。

    返回匹配位置供前端高亮。
    """

    def __init__(self, fuzzy_threshold: float = 0.7):
        """
        Args:
            fuzzy_threshold: 模糊匹配得分阈值（默认 0.7）
        """
        self.fuzzy_threshold = fuzzy_threshold

    def validate(
        self,
        page_text: str,
        evidence: str,
    ) -> Dict[str, Any]:
        """精确 → 模糊两阶段匹配。

        Args:
            page_text: 当前 PDF 页面的全文（fitz 提取的原始文本）
            evidence:  LLM 返回的"原文原句"

        Returns:
            {
                "valid": bool,
                "offset": int | None,
                "length": int | None,
                "fuzzy_score": float,
                "normalized_evidence": str
            }
        """
        result: Dict[str, Any] = {
            "valid": False,
            "offset": None,
            "length": None,
            "fuzzy_score": 0.0,
            "normalized_evidence": _normalize(evidence),
        }

        if not page_text or not evidence:
            return result

        # ----- Stage 1: 精确匹配 -----
        offset = page_text.find(evidence)
        if offset >= 0:
            result["valid"] = True
            result["offset"] = offset
            result["length"] = len(evidence)
            result["fuzzy_score"] = 1.0
            return result

        # ----- Stage 2: 模糊匹配 -----
        norm_evidence = _normalize(evidence)
        norm_page = _normalize(page_text)
        if not norm_evidence or not norm_page:
            return result

        # 在 normalized 页面文本中查找 normalized evidence
        norm_offset = norm_page.find(norm_evidence)
        if norm_offset >= 0:
            # fuzzy_score: evidence 长度 / normalized 文本中最长公共子串长度
            # 这里使用一个简单近似: 找到 → 给 0.85（与 plan 文档一致）
            score = 0.85
            result["valid"] = score >= self.fuzzy_threshold
            result["offset"] = self._map_norm_offset_to_original(
                page_text, norm_offset, len(norm_evidence)
            )
            result["length"] = self._estimate_original_length(
                page_text, norm_offset, len(norm_evidence)
            )
            result["fuzzy_score"] = score
            return result

        # Stage 2 失败: 返回 valid=False, score=0
        return result

    @staticmethod
    def _map_norm_offset_to_original(
        page_text: str, norm_offset: int, norm_length: int
    ) -> Optional[int]:
        """
        将 normalized 文本中的偏移映射回原始 page_text 的偏移。
        用于在 PDF viewer 中高亮（即使 LLM 输出空格/标点不匹配）。

        Returns:
            原始文本中的起始偏移（尽可能接近真实位置），失败返回 None
        """
        try:
            norm_pos = 0
            original_pos = 0
            while norm_pos < norm_offset and original_pos < len(page_text):
                ch = page_text[original_pos]
                if _FUZZY_STRIP_RE.match(ch):
                    # 被 strip 掉的字符
                    original_pos += 1
                else:
                    norm_pos += 1
                    original_pos += 1
            return original_pos
        except Exception:
            return None

    @staticmethod
    def _estimate_original_length(
        page_text: str, norm_offset: int, norm_length: int
    ) -> Optional[int]:
        """
        估算原始文本中模糊匹配段的长度（用于高亮矩形宽度）。
        """
        try:
            start = EvidenceValidator._map_norm_offset_to_original(
                page_text, norm_offset, norm_length
            )
            if start is None:
                return None
            # 从 start 起数 norm_length 个非 strip 字符
            norm_count = 0
            pos = start
            while pos < len(page_text) and norm_count < norm_length:
                ch = page_text[pos]
                if not _FUZZY_STRIP_RE.match(ch):
                    norm_count += 1
                pos += 1
            return pos - start
        except Exception:
            return None
