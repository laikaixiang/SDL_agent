"""
CSV写入模块
负责CSV文件的写入、格式化和数据处理
"""

import csv
import os
import time
from typing import Dict, Any, List, Optional

from core.config import Config


class CSVWriter:
    """
    CSV写入器类 - 负责CSV文件的写入和管理

    职责：
    - CSV文件写入
    - 数据格式化处理
    - 字段自动补全
    - 文件管理和备份
    """

    def __init__(self, session_path: str = None):
        """
        初始化CSV写入器

        Args:
            session_path: 会话基础路径，如果为None则使用默认路径
        """
        self.config = Config()
        self.session_path = session_path

    def write_extraction_results(
        self,
        data: List[Dict[str, Any]],
        fields: List[str],
        prefix: str,
        output_dir: Optional[str] = None
    ) -> str:
        """
        写入提取结果到CSV文件

        Args:
            data: 提取数据
            fields: 字段列表
            prefix: 文件名前缀
            output_dir: 输出目录，如果为None则使用会话路径或配置的目录

        Returns:
            文件路径
        """
        if output_dir is None:
            if self.session_path:
                output_dir = os.path.join(self.session_path, "extract")
            else:
                output_dir = self.config.EXTRACT_DIR

        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        filename = f"{prefix}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)

        # 确定所有字段
        all_keys = self._determine_all_keys(data, fields)

        # 写入CSV文件
        self._write_csv(filepath, data, all_keys)

        return filepath

    def write_temporal_results(
        self,
        data: List[Dict[str, Any]],
        fields: List[str],
        temporal_dir: Optional[str] = None
    ) -> str:
        """
        写入临时结果到CSV文件

        Args:
            data: 提取数据
            fields: 字段列表
            temporal_dir: 临时目录，如果为None则使用会话路径或配置的目录

        Returns:
            文件路径
        """
        if temporal_dir is None:
            if self.session_path:
                temporal_dir = os.path.join(self.session_path, "temporal")
            else:
                temporal_dir = self.config.TEMPORAL_DIR

        os.makedirs(temporal_dir, exist_ok=True)

        # 固定文件名
        filename = "extraction.csv"
        filepath = os.path.join(temporal_dir, filename)

        # 确定所有字段
        all_keys = self._determine_all_keys(data, fields)

        # 写入CSV文件
        self._write_csv(filepath, data, all_keys)

        return filepath

    def _determine_all_keys(self, data: List[Dict[str, Any]], fields: List[str]) -> List[str]:
        """
        确定所有字段名

        Args:
            data: 数据
            fields: 基础字段

        Returns:
            所有字段名列表
        """
        all_keys = set(fields)
        for d in data:
            all_keys.update(d.keys())
        return list(all_keys)

    def _write_csv(self, filepath: str, data: List[Dict[str, Any]], fieldnames: List[str]) -> None:
        """
        写入CSV文件

        Args:
            filepath: 文件路径
            data: 数据
            fieldnames: 字段名
        """
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)

    def write_empty_csv(self, filepath: str, fields: List[str]) -> None:
        """
        写入空CSV文件（只有表头）

        Args:
            filepath: 文件路径
            fields: 字段列表
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            csvfile.write(",".join(fields))

    def append_to_csv(self, filepath: str, data: List[Dict[str, Any]]) -> None:
        """
        追加数据到CSV文件

        Args:
            filepath: 文件路径
            data: 要追加的数据
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")

        # 读取现有表头
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)

        # 追加数据
        with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            for row in data:
                writer.writerow(row)

    def merge_csv_files(self, input_files: List[str], output_file: str) -> None:
        """
        合并多个CSV文件

        Args:
            input_files: 输入文件列表
            output_file: 输出文件
        """
        all_data = []
        all_headers = set()

        # 读取所有文件
        for file in input_files:
            if not os.path.exists(file):
                continue

            with open(file, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                headers = reader.fieldnames or []
                all_headers.update(headers)

                for row in reader:
                    all_data.append(row)

        # 写入合并后的文件
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(all_headers))
            writer.writeheader()
            for row in all_data:
                writer.writerow(row)

    def validate_csv_file(self, filepath: str) -> tuple[bool, str]:
        """
        验证CSV文件

        Args:
            filepath: 文件路径

        Returns:
            (是否有效, 错误信息)
        """
        try:
            if not os.path.exists(filepath):
                return False, "文件不存在"

            if not filepath.lower().endswith('.csv'):
                return False, "不是CSV文件"

            with open(filepath, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader, None)
                if not headers:
                    return False, "CSV文件没有表头"

            return True, ""

        except Exception as e:
            return False, f"验证失败: {str(e)}"

    def get_csv_info(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        获取CSV文件信息

        Args:
            filepath: 文件路径

        Returns:
            CSV文件信息或None
        """
        try:
            if not os.path.exists(filepath):
                return None

            with open(filepath, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader, None)
                row_count = sum(1 for _ in reader)

            return {
                "path": filepath,
                "filename": os.path.basename(filepath),
                "size": os.path.getsize(filepath),
                "headers": headers,
                "row_count": row_count
            }

        except Exception as e:
            print(f"获取CSV文件信息失败: {e}")
            return None