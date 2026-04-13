"""
CSV 文件读取工具 (software/readfile.py)
=======================================

为自动分析流水线提供统一的 CSV 读取接口。
大语言模型根据列名判断调用哪个函数，Python 通过 READER_REGISTRY 动态分发。

对外接口：
    from software.readfile import (
        read_column_names,
        read_as_columns_dict,
        read_spectrum_format,
        read_numeric_columns,
        read_single_column,
        READER_REGISTRY,
        FUNCTIONS_DESCRIPTION,
    )
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ==============================================================================
# 读取函数
# ==============================================================================

def read_column_names(csv_path: str) -> list:
    """
    读取 CSV 文件的所有列名

    Args:
        csv_path: CSV 文件路径

    Returns:
        列名列表，如 ["wavelength", "intensity", "sample_id"]
    """
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or [])


def read_as_columns_dict(csv_path: str, columns: list = None) -> dict:
    """
    将 CSV 读取为列字典，每列值保留原始字符串

    Args:
        csv_path: CSV 文件路径
        columns : 要读取的列名列表；None 表示读取全部列

    Returns:
        {"列名": ["值1", "值2", ...], ...}
    """
    result = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if columns is None or key in columns:
                    result.setdefault(key, []).append(val)
    return result


def read_spectrum_format(csv_path: str, wavelength_col: str, intensity_col: str) -> dict:
    """
    按光谱格式读取 CSV，返回 wavelength 与 intensity 数组

    Args:
        csv_path      : CSV 文件路径
        wavelength_col: 波长列的列名
        intensity_col : 强度列的列名（单列或第一条光谱）

    Returns:
        {"wavelength": [float, ...], "intensity": [float, ...]}

    Raises:
        ValueError: 列名不存在或数据无法转换为浮点数
    """
    wavelengths = []
    intensities = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        if wavelength_col not in fieldnames:
            raise ValueError(f"列 '{wavelength_col}' 不存在，可用列：{fieldnames}")
        if intensity_col not in fieldnames:
            raise ValueError(f"列 '{intensity_col}' 不存在，可用列：{fieldnames}")

        for row in reader:
            try:
                wavelengths.append(float(row[wavelength_col]))
                intensities.append(float(row[intensity_col]))
            except (ValueError, TypeError):
                pass  # 跳过无法转换的行（如标题行残留）

    if not wavelengths:
        raise ValueError("读取后数据为空，请检查列名和数据格式")

    return {"wavelength": wavelengths, "intensity": intensities}


def read_numeric_columns(csv_path: str) -> dict:
    """
    读取 CSV 中所有数值列（跳过无法全部解析为浮点数的列）

    Args:
        csv_path: CSV 文件路径

    Returns:
        {"列名": [float, ...], ...}  仅包含数值列
    """
    raw = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                raw.setdefault(key, []).append(val)

    numeric = {}
    for col, values in raw.items():
        parsed = []
        ok = True
        for v in values:
            try:
                parsed.append(float(v))
            except (ValueError, TypeError):
                ok = False
                break
        if ok and parsed:
            numeric[col] = parsed

    return numeric


def read_single_column(csv_path: str, column: str) -> list:
    """
    读取 CSV 中的单列数据（保留原始字符串）

    Args:
        csv_path: CSV 文件路径
        column  : 列名

    Returns:
        [值1, 值2, ...]

    Raises:
        ValueError: 列名不存在
    """
    result = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if column not in fieldnames:
            raise ValueError(f"列 '{column}' 不存在，可用列：{fieldnames}")
        for row in reader:
            result.append(row[column])
    return result


# ==============================================================================
# 动态分发注册表（供 auto_analyze.py 使用）
# ==============================================================================

READER_REGISTRY: dict = {
    "read_as_columns_dict" : read_as_columns_dict,
    "read_spectrum_format" : read_spectrum_format,
    "read_numeric_columns" : read_numeric_columns,
    "read_single_column"   : read_single_column,
}

# ==============================================================================
# 函数描述（嵌入 LLM 提示词）
# ==============================================================================

FUNCTIONS_DESCRIPTION: str = """
可用的数据读取函数（read_function 字段必须从以下名称中选择一个）：

- read_as_columns_dict(csv_path, columns=None)
    返回: {"列名": [值列表, ...]}（值为字符串）
    适用: data_statistics、data_normalization 等需要多列数值的算法
    read_params 示例: {} 或 {"columns": ["col1", "col2"]}

- read_spectrum_format(csv_path, wavelength_col, intensity_col)
    返回: {"wavelength": [float, ...], "intensity": [float, ...]}
    适用: spectrum_analysis（光谱峰值/FWHM/峰面积分析）
    read_params 示例: {"wavelength_col": "wavelength", "intensity_col": "intensity"}

- read_numeric_columns(csv_path)
    返回: {"列名": [float, ...]}，仅含所有值均为数字的列
    适用: 不确定列名时，自动筛选数值列
    read_params 示例: {}

- read_single_column(csv_path, column)
    返回: ["值1", "值2", ...]（字符串列表）
    适用: 只需要单列数据的算法
    read_params 示例: {"column": "PCE"}
""".strip()


# ==============================================================================
# 测试接口
# ==============================================================================

if __name__ == "__main__":
    import json

    # 生成临时测试 CSV
    _TEST_CSV = "_test_readfile_tmp.csv"
    _TEST_DATA = [
        ["wavelength", "intensity", "sample", "PCE"],
        ["400", "0.12", "A1", "15.2"],
        ["450", "0.45", "A1", "16.1"],
        ["500", "0.88", "A2", "17.3"],
        ["550", "0.92", "A2", "18.0"],
        ["600", "0.61", "A3", "16.8"],
    ]
    with open(_TEST_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(_TEST_DATA)

    print("=" * 50)
    print("read_column_names:")
    print(read_column_names(_TEST_CSV))

    print("\n" + "=" * 50)
    print("read_as_columns_dict (全列):")
    print(json.dumps(read_as_columns_dict(_TEST_CSV), ensure_ascii=False, indent=2))

    print("\n" + "=" * 50)
    print("read_spectrum_format:")
    print(json.dumps(read_spectrum_format(_TEST_CSV, "wavelength", "intensity"), indent=2))

    print("\n" + "=" * 50)
    print("read_numeric_columns:")
    print(json.dumps(read_numeric_columns(_TEST_CSV), indent=2))

    print("\n" + "=" * 50)
    print("read_single_column('sample'):")
    print(read_single_column(_TEST_CSV, "sample"))

    os.remove(_TEST_CSV)
    print("\n测试完成，临时文件已删除。")
