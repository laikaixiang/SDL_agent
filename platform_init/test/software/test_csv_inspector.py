"""
software/csv_inspector.py 测试

覆盖：
  - 8 个类型推断 case（int/float/bool/date/str/混合/全空/科学计数法）
  - PreviewData 字段完整性（path/columns/row_count/total_rows/file_size）
  - sample 数量限制（前 3 个非空值）
  - 空文件、缺失文件异常处理
  - 大文件估算（构造 >64KB 的 CSV 验证估算不崩）

运行：pytest platform_init/test/software/test_csv_inspector.py -v
"""
import os
import sys
import csv
import math
import tempfile
from pathlib import Path

# 共享 sys.path 修复
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pytest

from software.csv_inspector import (
    inspect_csv,
    infer_column_type,
    PreviewData,
    ColumnInfo,
    _estimate_total_rows,
)


# =============================================================================
# T1-T8: 类型推断单元测试
# =============================================================================

class TestInferColumnType:
    def test_int_column(self):
        assert infer_column_type(["1", "2", "3", "100", "-5"]) == "int"

    def test_int_with_leading_zeros(self):
        # Python int() 接受 007 → 7，故推断为 int
        assert infer_column_type(["007", "008", "009"]) == "int"

    def test_float_column(self):
        assert infer_column_type(["1.5", "2.7", "3.14", "-0.5"]) == "float"

    def test_scientific_notation(self):
        # 1e-3 / 2.5e2 都是 float
        assert infer_column_type(["1e-3", "2.5e2", "1.0E-5"]) == "float"

    def test_bool_true_values(self):
        # true/yes/1 都是 bool 候选词
        assert infer_column_type(["true", "false", "true"]) == "bool"

    def test_bool_mixed_cases(self):
        # 不区分大小写
        assert infer_column_type(["True", "FALSE", "Yes", "NO", "0", "1"]) == "bool"

    def test_bool_does_not_match_other_words(self):
        # maybe/ok 不是 bool 标志词
        assert infer_column_type(["yes", "no", "maybe"]) == "str"

    def test_date_iso_format(self):
        assert infer_column_type(["2024-01-15", "2024-02-20", "2025-12-31"]) == "date"

    def test_date_slash_format(self):
        assert infer_column_type(["2024/01/15", "2024/02/20"]) == "date"

    def test_str_column(self):
        # 含字母/混合内容 → str
        assert infer_column_type(["A1", "B2", "C3"]) == "str"

    def test_all_empty_returns_str(self):
        assert infer_column_type(["", "", ""]) == "str"

    def test_priority_int_over_float(self):
        # 全部能转 int → int（即使也能转 float）
        assert infer_column_type(["1", "2", "3"]) == "int"

    def test_priority_float_when_has_decimal(self):
        # 含小数点 → float（不能转 int）
        assert infer_column_type(["1.0", "2.5", "3.7"]) == "float"


# =============================================================================
# T9: PreviewData 字段完整性
# =============================================================================

def test_preview_returns_all_fields(tmp_path):
    """inspect_csv 返回的 PreviewData 包含所有字段"""
    csv_file = tmp_path / "sample.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["PCE", "thickness", "label"])
        for i in range(5):
            writer.writerow([15.0 + i, 100 + i * 10, f"sample_{i}"])

    preview = inspect_csv(str(csv_file), n_rows=20)

    assert isinstance(preview, PreviewData)
    assert preview.path == str(csv_file)
    assert preview.row_count == 5
    assert preview.total_rows == 5  # 小文件精确计数
    assert preview.file_size > 0
    assert len(preview.columns) == 3


# =============================================================================
# T10-T11: 列元数据正确性
# =============================================================================

def test_columns_have_correct_types(tmp_path):
    """每列的 type 字段推断正确"""
    csv_file = tmp_path / "types.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["int_col", "float_col", "str_col", "bool_col", "date_col"])
        for i in range(3):
            writer.writerow([i, i + 0.5, f"x{i}", "true", "2024-01-01"])

    preview = inspect_csv(str(csv_file), n_rows=20)
    types_by_name = {c.name: c.type for c in preview.columns}
    assert types_by_name["int_col"] == "int"
    assert types_by_name["float_col"] == "float"
    assert types_by_name["str_col"] == "str"
    assert types_by_name["bool_col"] == "bool"
    assert types_by_name["date_col"] == "date"


def test_sample_has_at_most_3_values(tmp_path):
    """sample 最多 3 个非空值"""
    csv_file = tmp_path / "sample_size.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x"])
        for i in range(10):
            writer.writerow([str(i)])

    preview = inspect_csv(str(csv_file), n_rows=20)
    assert len(preview.columns[0].sample) == 3
    assert preview.columns[0].sample == ["0", "1", "2"]


# =============================================================================
# T12: row_count 与 n_rows 关系
# =============================================================================

def test_row_count_caps_at_n_rows(tmp_path):
    """row_count 不会超过 n_rows"""
    csv_file = tmp_path / "many_rows.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x"])
        for i in range(100):
            writer.writerow([str(i)])

    preview = inspect_csv(str(csv_file), n_rows=10)
    assert preview.row_count == 10
    assert preview.total_rows == 100


# =============================================================================
# T13-T14: 异常处理
# =============================================================================

def test_file_not_found():
    """文件不存在时抛 FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        inspect_csv("/nonexistent/path/to/file.csv")


def test_empty_file(tmp_path):
    """空文件: total_rows=0, row_count=0, columns 为空"""
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("", encoding="utf-8")

    preview = inspect_csv(str(csv_file), n_rows=20)
    assert preview.row_count == 0
    assert preview.total_rows == 0
    assert preview.columns == []


# =============================================================================
# T15: null_count 统计
# =============================================================================

def test_null_count_tracking(tmp_path):
    """null_count 正确统计空值"""
    csv_file = tmp_path / "with_nulls.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        writer.writerow(["1", ""])      # y 是空
        writer.writerow(["2", "5"])
        writer.writerow(["3", ""])      # y 是空
        writer.writerow(["4", "6"])

    preview = inspect_csv(str(csv_file), n_rows=20)
    y_col = next(c for c in preview.columns if c.name == "y")
    assert y_col.null_count == 2
    assert y_col.non_null_count == 2


# =============================================================================
# T16: 大文件估算
# =============================================================================

def test_large_file_estimate(tmp_path):
    """大文件 >64KB 走估算路径，不崩溃"""
    csv_file = tmp_path / "large.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        # 写 10000 行（约 80KB）
        for i in range(10000):
            writer.writerow([str(i), str(i * 0.5)])

    file_size = csv_file.stat().st_size
    assert file_size > 65536, f"test fixture should be > 64KB, got {file_size}"

    preview = inspect_csv(str(csv_file), n_rows=20)
    # 估算的总行数应该在合理范围 (8000-12000)
    assert 8000 <= preview.total_rows <= 12000, f"total_rows={preview.total_rows}"
    assert preview.row_count == 20  # 只读了 20 行


# =============================================================================
# T17: to_dict 序列化
# =============================================================================

def test_to_dict_serializable(tmp_path):
    """PreviewData.to_dict 返回可 JSON 序列化的 dict"""
    import json
    csv_file = tmp_path / "to_dict.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["a"])
        writer.writerow(["1"])

    preview = inspect_csv(str(csv_file), n_rows=5)
    d = preview.to_dict()
    # 不抛异常即视为通过
    json.dumps(d, ensure_ascii=False)
    assert d["path"] == str(csv_file)
    assert d["row_count"] == 1
    assert d["total_rows"] == 1
    assert isinstance(d["columns"], list)


# =============================================================================
# T18: 实际光谱 CSV (复用 conftest 风格)
# =============================================================================

def test_real_world_spectrum_csv(tmp_path):
    """实际光谱数据 (wavelength=int, intensity=float) 推断正确"""
    csv_file = tmp_path / "spectrum.csv"
    wl = list(range(400, 700))
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["wavelength", "intensity"])
        for w in wl:
            intensity = 0.05 + 0.9 * math.exp(-0.5 * ((w - 532) / 15) ** 2)
            writer.writerow([w, round(intensity, 6)])

    preview = inspect_csv(str(csv_file), n_rows=20)
    types_by_name = {c.name: c.type for c in preview.columns}
    assert types_by_name["wavelength"] == "int"
    assert types_by_name["intensity"] == "float"
    assert preview.total_rows == 300
