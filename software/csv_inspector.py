"""
CSV 文件类型推断 (software/csv_inspector.py)
============================================

对 CSV 文件进行轻量级"元数据"提取，**不调用 LLM**：

    1. 读取列名（header）
    2. 读取前 N 行数据（默认 20）
    3. 对每列前 K 个非空值做类型推断：
        - 全部能转 int        → "int"
        - 全部能转 float      → "float"
        - 全部是 bool 标志    → "bool"
        - 匹配日期格式         → "date"
        - 否则                 → "str"
    4. 估算文件总行数（用累计字节流，不占内存）

对外接口：
    from software.csv_inspector import inspect_csv, ColumnInfo, PreviewData

    preview = inspect_csv("temporal/extraction.csv", n_rows=20)
    # PreviewData(
    #     path=...,
    #     columns=[ColumnInfo(name="PCE", type="float", sample=["15.0", "15.3", ...]), ...],
    #     row_count=20,
    #     total_rows=1234,
    #     file_size=98765,
    # )
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ==============================================================================
# 数据模型
# ==============================================================================

# 推断过程中使用的"类型"常量
_TYPE_INT = "int"
_TYPE_FLOAT = "float"
_TYPE_BOOL = "bool"
_TYPE_DATE = "date"
_TYPE_STR = "str"

# bool 候选词（不区分大小写）
_BOOL_TRUE = {"true", "yes", "1"}
_BOOL_FALSE = {"false", "no", "0"}
_BOOL_VALUES = _BOOL_TRUE | _BOOL_FALSE

# date 格式: YYYY-MM-DD / YYYY/MM/DD
_DATE_RE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")


@dataclass
class ColumnInfo:
    """单列的元数据"""
    name: str
    type: str = _TYPE_STR
    sample: List[str] = field(default_factory=list)
    non_null_count: int = 0
    null_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreviewData:
    """CSV 预览数据"""
    path: str
    columns: List[ColumnInfo] = field(default_factory=list)
    row_count: int = 0       # 实际读出的行数（n_rows）
    total_rows: int = 0      # 估算的总行数
    file_size: int = 0       # 文件字节数

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "columns": [c.to_dict() for c in self.columns],
            "row_count": self.row_count,
            "total_rows": self.total_rows,
            "file_size": self.file_size,
        }


# ==============================================================================
# 类型推断核心
# ==============================================================================

def _try_int(s: str) -> bool:
    """判断字符串是否可以转为 int（接受 0/+/- 前缀）"""
    if not s:
        return False
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False


def _try_float(s: str) -> bool:
    """判断字符串是否可以转为 float（接受科学计数法）"""
    if not s:
        return False
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _try_bool(s: str) -> bool:
    """判断字符串是否是 bool 标志词"""
    return s.strip().lower() in _BOOL_VALUES


def _try_date(s: str) -> bool:
    """判断字符串是否是 YYYY-MM-DD 或 YYYY/MM/DD"""
    return bool(_DATE_RE.match(s.strip()))


def infer_column_type(values: List[str]) -> str:
    """
    给定一列的非空字符串值列表（最多 N 个），推断列类型。

    优先级：int > float > bool > date > str

    规则：
    - 全空 → str（无法判断）
    - 全部能转 int → int
    - 全部能转 float → float
    - 全部是 bool 标志词 → bool
    - 全部匹配日期格式 → date
    - 否则 → str

    Args:
        values: 字符串值列表

    Returns:
        "int" | "float" | "bool" | "date" | "str"
    """
    non_empty = [v for v in values if v is not None and str(v).strip() != ""]
    if not non_empty:
        return _TYPE_STR

    # 优先级 1: int（强约束）
    if all(_try_int(v) for v in non_empty):
        return _TYPE_INT

    # 优先级 2: float
    if all(_try_float(v) for v in non_empty):
        return _TYPE_FLOAT

    # 优先级 3: bool（候选词完全限定）
    if all(_try_bool(v) for v in non_empty):
        return _TYPE_BOOL

    # 优先级 4: date
    if all(_try_date(v) for v in non_empty):
        return _TYPE_DATE

    # 兜底
    return _TYPE_STR


# ==============================================================================
# 行数估算
# ==============================================================================

def _estimate_total_rows(path: str, sample_bytes: int = 65536) -> int:
    """
    估算 CSV 文件总行数（不把整个文件读进内存）。

    方法：
    1. 读前 64KB，统计行数
    2. 按比例推算总行数（最小返回 1）

    对于小文件会直接读到结尾精确计数。
    """
    file_size = os.path.getsize(path)
    if file_size == 0:
        return 0

    # 小文件: 精确计数
    if file_size <= sample_bytes:
        count = 0
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            for _ in f:
                count += 1
        return max(count - 1, 0)  # 减去 header

    # 大文件: 采样估算
    sample_lines = 0
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        sample = f.read(sample_bytes)
        sample_lines = sample.count("\n")

    if sample_lines <= 0:
        return 0
    # 估算总行数 = sample_lines * (file_size / sample_bytes)
    estimated = int(sample_lines * (file_size / sample_bytes))
    return max(estimated, 0)


# ==============================================================================
# 主入口
# ==============================================================================

def inspect_csv(path: str, n_rows: int = 20) -> PreviewData:
    """
    检查 CSV 文件，返回列元数据 + 前 N 行预览。

    Args:
        path  : CSV 文件绝对路径
        n_rows: 预览行数（默认 20）

    Returns:
        PreviewData

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件无法读取（不是 CSV 格式或编码错误）
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV 文件不存在: {path}")

    file_size = os.path.getsize(path)
    total_rows = _estimate_total_rows(path)

    columns: List[ColumnInfo] = []
    row_count = 0

    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])

        # 初始化列结构
        for name in fieldnames:
            columns.append(ColumnInfo(name=name, type=_TYPE_STR, sample=[], non_null_count=0, null_count=0))

        # 读前 n_rows
        for i, row in enumerate(reader):
            if i >= n_rows:
                break
            row_count += 1
            for col in columns:
                val = row.get(col.name, "")
                if val is None or str(val).strip() == "":
                    col.null_count += 1
                else:
                    col.non_null_count += 1
                    # 收集前 3 个非空值作为 sample
                    if len(col.sample) < 3:
                        col.sample.append(str(val))

    # 类型推断：用 sample 推断
    for col in columns:
        if col.sample:
            col.type = infer_column_type(col.sample)

    return PreviewData(
        path=path,
        columns=columns,
        row_count=row_count,
        total_rows=total_rows,
        file_size=file_size,
    )


# ==============================================================================
# 测试接口
# ==============================================================================

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python csv_inspector.py <path-to-csv>")
        sys.exit(1)

    preview = inspect_csv(sys.argv[1], n_rows=20)
    print(json.dumps(preview.to_dict(), ensure_ascii=False, indent=2))
