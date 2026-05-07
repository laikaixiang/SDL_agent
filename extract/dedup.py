"""
提取结果去重模块

在所有页面提取完成后、CSV 写入前，按实体名称（第一个字段）去重，合并重复行的信息。

TODO: 后续优化方向
  1. 语义相似度去重：使用 embedding 向量聚类识别同义实体（如 "PEAI" = "phenethylammonium iodide"）
  2. LLM 层面跨页感知：在 prompt 中传入已提取的实体列表，让 LLM 跳过去重
"""

from typing import List, Dict, Any


def _normalize_key(value: str, strategy: str) -> str:
    """规范化实体名称"""
    if strategy == "lower":
        return value.strip().lower()
    elif strategy == "strict":
        return value
    else:  # "strip" (default)
        return value.strip()


def _merge_field_values(values: List[str], strategy: str) -> str:
    """合并重复行的同字段值"""
    non_empty = [v for v in values if v and v.strip()]
    if not non_empty:
        return ""

    if strategy == "first_non_empty":
        return non_empty[0]
    else:  # "longest" (default)
        return max(non_empty, key=len)


def _is_non_empty(value: Any) -> bool:
    """判断字段值是否有意义"""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def deduplicate_extraction_results(
    data: List[Dict[str, Any]],
    fields: List[str],
    *,
    normalize: str = "strip",
    merge_strategy: str = "longest",
    add_metadata: bool = True,
) -> List[Dict[str, Any]]:
    """
    按实体名称（fields[0]）去重，合并重复行的信息。

    Args:
        data: 原始提取数据列表 (all_extracted_data)
        fields: 用户确认的字段列表，fields[0] 为实体键
        normalize: 规范化策略 — "strip" | "lower" | "strict"
        merge_strategy: 合并策略 — "longest" | "first_non_empty"
        add_metadata: 是否添加 _occurrence_count 和 _source_docs 列

    Returns:
        去重后的数据列表，按实体键排序
    """
    if not data or not fields:
        return list(data) if data else []

    entity_key = fields[0]

    # 按规范化实体键分组
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in data:
        raw_key = item.get(entity_key, "")
        if not _is_non_empty(raw_key):
            continue
        norm_key = _normalize_key(str(raw_key), normalize)
        groups.setdefault(norm_key, []).append(item)

    # 合并每组
    result: List[Dict[str, Any]] = []
    for norm_key, group in groups.items():
        merged: Dict[str, Any] = {}

        # 收集所有字段名（union）
        all_field_names = set()
        for item in group:
            all_field_names.update(item.keys())

        for field_name in all_field_names:
            if field_name == entity_key:
                # 实体键：取规范化后第一个非空值
                merged[field_name] = norm_key
            elif field_name == "_source_doc":
                # 收集所有唯一的源文档
                docs = []
                seen = set()
                for item in group:
                    doc = item.get("_source_doc", "")
                    if doc and doc not in seen:
                        docs.append(doc)
                        seen.add(doc)
                merged["_source_doc"] = "; ".join(docs)
            elif field_name.startswith("_"):
                # 跳过其他内部字段（如 _occurrence_count, _source_docs）
                pass
            else:
                values = [str(item.get(field_name, "")) for item in group]
                merged[field_name] = _merge_field_values(values, merge_strategy)

        if add_metadata:
            merged["_occurrence_count"] = len(group)
            # 收集唯一源文档（不修改原 _source_doc 的情况下添加）
            docs = []
            seen = set()
            for item in group:
                doc = item.get("_source_doc", "")
                if doc and doc not in seen:
                    docs.append(doc)
                    seen.add(doc)
            merged["_source_docs"] = "; ".join(docs)

        result.append(merged)

    # 按实体键排序，保证确定性输出
    result.sort(key=lambda x: str(x.get(entity_key, "")))

    return result
