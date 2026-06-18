"""
Semantic Dedup — 对 dedup.py 输出做 embedding 聚类, 合并语义相同的行.

在 fields[0] 规范化后仍不同的行之间做 (例如 "PEAI" vs "phenethylammonium iodide").
复用 extract/embedding_service.py 中的 EmbeddingService.

合并策略同 extract/dedup.py:
  - 主键 (entity key): 取规范化后第一个非空值
  - 其它字段: longest | first_non_empty (默认 longest)
"""

import math
from typing import Any, Dict, List, Optional

from .dedup import _is_non_empty, _merge_field_values


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """两个向量的余弦相似度。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _build_embedding_text(record: Dict[str, Any], fields: List[str], max_extra: int = 2) -> str:
    """
    为单条记录构造 embedding 文本:
      fields[0] (主键) + 1-2 个关键非空字段值
    """
    parts: List[str] = []
    if fields:
        primary = str(record.get(fields[0], "") or "").strip()
        if primary:
            parts.append(primary)
    # 追加 1-2 个非空关键字段 (跳过 _ 前缀的内部字段)
    extra = 0
    for f in fields[1:] if fields else []:
        if extra >= max_extra:
            break
        v = record.get(f)
        if _is_non_empty(v):
            parts.append(str(v).strip())
            extra += 1
    return " | ".join(parts) if parts else ""


class SemanticDedup:
    """对 dedup.py 输出做 embedding 聚类, 合并语义相同的行.

    在 fields[0] 规范化后仍不同的行之间做.
    """

    def __init__(
        self,
        embedding_service,
        similarity_threshold: float = 0.92,
        merge_strategy: str = "longest",
    ):
        """
        Args:
            embedding_service: EmbeddingService 实例 (需有 embed_text)
            similarity_threshold: 余弦相似度阈值, 超过此值视为同义
            merge_strategy: "longest" | "first_non_empty"
        """
        self.embedder = embedding_service
        self.threshold = similarity_threshold
        self.merge_strategy = merge_strategy

    def cluster_and_merge(
        self,
        records: List[Dict[str, Any]],
        fields: List[str],
    ) -> List[Dict[str, Any]]:
        """
        1) 对每条记录用 fields[0] + 1-2 个关键字段拼成 embedding 文本
        2) 简单贪心聚类: cosine >= threshold 合并
        3) 合并策略: longest 字段值优先 (同 dedup.py:_merge_field_values)

        Args:
            records: 已通过规则 dedup 的记录
            fields:  字段列表 (fields[0] 为主键)

        Returns:
            合并后的记录列表
        """
        if not records or not fields:
            return list(records) if records else []

        entity_key = fields[0]

        # 1) 构造 embedding 输入文本
        embed_texts: List[str] = [_build_embedding_text(r, fields) for r in records]

        # 2) 调 embedding (空文本 → zero-vector, 不会被合并)
        try:
            vectors: List[Optional[List[float]]] = [
                self.embedder.embed_text(t) if t else None
                for t in embed_texts
            ]
        except Exception:
            # embedding 失败 → 不做语义合并, 直接返回原列表
            return list(records)

        # 3) 贪心聚类: 按顺序遍历, 找 cluster 代表
        n = len(records)
        parent: List[int] = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            vi = vectors[i]
            if vi is None:
                continue
            for j in range(i + 1, n):
                vj = vectors[j]
                if vj is None:
                    continue
                if find(i) == find(j):
                    continue
                sim = _cosine_similarity(vi, vj)
                if sim >= self.threshold:
                    union(i, j)

        # 4) 按 cluster 聚合
        clusters: Dict[int, List[int]] = {}
        for i in range(n):
            r = find(i)
            clusters.setdefault(r, []).append(i)

        result: List[Dict[str, Any]] = []
        for cluster_indices in clusters.values():
            if len(cluster_indices) == 1:
                result.append(dict(records[cluster_indices[0]]))
                continue

            group = [records[i] for i in cluster_indices]

            # 合并所有字段 (同 dedup.py 风格)
            merged: Dict[str, Any] = {}
            all_field_names: set = set()
            for item in group:
                all_field_names.update(item.keys())

            for fname in all_field_names:
                if fname == entity_key:
                    # 主键: 取第一个非空值
                    primary_val = ""
                    for item in group:
                        v = str(item.get(entity_key, "") or "").strip()
                        if v:
                            primary_val = v
                            break
                    merged[fname] = primary_val
                elif fname == "_source_doc":
                    docs: List[str] = []
                    seen: set = set()
                    for item in group:
                        d = item.get("_source_doc", "")
                        if d and d not in seen:
                            docs.append(d)
                            seen.add(d)
                    merged["_source_doc"] = "; ".join(docs)
                elif fname.startswith("_"):
                    pass
                else:
                    values = [str(item.get(fname, "")) for item in group]
                    merged[fname] = _merge_field_values(values, self.merge_strategy)

            # 元数据
            merged["_occurrence_count"] = sum(
                int(item.get("_occurrence_count", 1) or 1) for item in group
            )
            merged["_semantic_merged"] = True
            merged["_semantic_cluster_size"] = len(cluster_indices)

            # 保留 evidence offset (取主键匹配的记录的 offset)
            if "_evidence_offset" in all_field_names:
                for item in group:
                    offset = item.get("_evidence_offset")
                    if offset is not None and offset != "":
                        merged["_evidence_offset"] = offset
                        merged["_evidence_length"] = item.get("_evidence_length")
                        merged["_evidence_score"] = item.get("_evidence_score", 0.0)
                        merged["_low_confidence"] = item.get("_low_confidence", False)
                        break

            result.append(merged)

        # 按 entity_key 排序, 保持确定性输出
        result.sort(key=lambda x: str(x.get(entity_key, "")))
        return result
