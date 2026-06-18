"""
Extraction Review Agent — 对 dedup 后的最终结果做一次轻量 LLM 审查.

任务:
  1) 标记仍可能重复的行 (LLM 语义判断)
  2) 标记明显错误 (单位缺失、数值异常大)
  3) 不动数据结构, 只打 review 标记 (_review_flag / _duplicate_of / _review_note)

调用链:
  ExtractionEngine._save_extraction_results
    -> ExtractionReviewAgent.review(records, fields)
    -> 分批 (每批 10 条) 调 LLM, 解析结果, 写回原 records
"""

import json
from typing import Any, Dict, List, Optional

from core.config import Config
from core.llm_client import LLMClient


def _safe_json_loads(text: str) -> Optional[Any]:
    """
    多策略 JSON 解析: 直接加载 → 找首个 [...] 块 → 找首个 {...} 块.
    """
    # 1) 直接加载
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) 找首个 [...] 块
    import re
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # 3) 找首个 {...} 块
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


class ExtractionReviewAgent:
    """对 dedup 后的最终结果做一次轻量 LLM 审查."""

    DEFAULT_BATCH_SIZE: int = 10

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Args:
            llm_client: LLMClient 实例, 为 None 时自动创建 (使用 TALK 模型)
        """
        self.config = Config()
        self.llm = llm_client or LLMClient(
            api_key=self.config.TALK_API_KEY,
            api_url=self.config.TALK_API_URL,
            extra_body=self.config.get_extra_body("TALK"),
        )

    def review(
        self,
        records: List[Dict[str, Any]],
        fields: List[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[Dict[str, Any]]:
        """
        分批审查 records, 给每条记录添加 _review_flag / _duplicate_of / _review_note 字段.

        Args:
            records:   已 dedup 的提取结果
            fields:    字段列表
            batch_size: 每批送审的记录数

        Returns:
            新的 records 列表 (原 records 保持不变, 返回浅拷贝)
        """
        if not records:
            return list(records)

        from prompts import create_prompt_manager

        pm = create_prompt_manager(lang="zh")
        # 在原 records 上做浅拷贝, 避免污染上游
        out: List[Dict[str, Any]] = [dict(r) for r in records]

        n = len(out)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = out[start:end]

            try:
                review_items = self._review_batch(pm, batch, fields, start)
            except Exception as e:
                # 单批失败不阻塞, 标记本批为 skipped
                for r in batch:
                    r.setdefault("_review_flag", "skipped")
                    r.setdefault("_review_note", f"review_agent 调用失败: {e}")
                continue

            for item in review_items:
                idx = item.get("row_index")
                if not isinstance(idx, int):
                    continue
                # row_index 是 batch 内的索引; 转换为全局索引
                global_idx = start + idx
                if global_idx < 0 or global_idx >= n:
                    continue
                flag = item.get("flag", "ok")
                note = item.get("note", "")
                out[global_idx]["_review_flag"] = flag
                out[global_idx]["_review_note"] = note
                if flag == "duplicate":
                    # 找疑似重复的目标行 (主键最相似的另一条)
                    target = self._find_duplicate_of(out[global_idx], out, global_idx, fields)
                    if target is not None:
                        out[global_idx]["_duplicate_of"] = target

        return out

    def _review_batch(
        self,
        pm,
        batch: List[Dict[str, Any]],
        fields: List[str],
        start_offset: int,
    ) -> List[Dict[str, Any]]:
        """对单批 records 调一次 LLM, 返回审查结果列表."""
        # 构造 batch 摘要 (避免 LLM 看到全量长文本)
        simplified = []
        for i, r in enumerate(batch):
            entry: Dict[str, Any] = {"row_index": i}
            for f in fields[:5]:  # 最多 5 个字段
                v = r.get(f)
                if v is None or v == "":
                    continue
                entry[f] = str(v)[:200]
            simplified.append(entry)

        batch_json = json.dumps(simplified, ensure_ascii=False, indent=2)
        sys_prompt = pm.get("review_extraction_system", batch_json=batch_json)

        messages = [{"role": "user", "content": sys_prompt}]

        result = self.llm.call_api(
            model=self.config.MODEL_NAME_TALK,
            messages=messages,
            temperature=0.1,
            max_tokens=None,
        )

        if not result or "choices" not in result:
            return []

        content = result["choices"][0]["message"]["content"].strip()
        # 去掉 markdown code fence
        content = content.replace("```json", "").replace("```", "").strip()
        parsed = _safe_json_loads(content)
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return []
        # 过滤非 dict
        return [x for x in parsed if isinstance(x, dict)]

    @staticmethod
    def _find_duplicate_of(
        record: Dict[str, Any],
        all_records: List[Dict[str, Any]],
        self_idx: int,
        fields: List[str],
    ) -> Optional[int]:
        """
        在 all_records 中找与 record 重复的另一行 (返回全局索引).
        启发式: 主键包含关系 + occurrence_count 较高者优先.
        """
        if not fields:
            return None
        primary_key = fields[0]
        primary = str(record.get(primary_key, "") or "").strip().lower()
        if not primary:
            return None

        best_idx: Optional[int] = None
        best_count = 0
        for i, other in enumerate(all_records):
            if i == self_idx:
                continue
            other_primary = str(other.get(primary_key, "") or "").strip().lower()
            if not other_primary:
                continue
            # 主键互含 或 完全相同
            if primary == other_primary:
                occ = int(other.get("_occurrence_count", 1) or 1)
                if occ > best_count:
                    best_count = occ
                    best_idx = i
            elif primary in other_primary or other_primary in primary:
                occ = int(other.get("_occurrence_count", 1) or 1)
                if occ > best_count:
                    best_count = occ
                    best_idx = i
        return best_idx
