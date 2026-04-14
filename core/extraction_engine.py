"""
提取引擎核心模块
负责PDF文献提取的核心逻辑，包括动态字段处理、LLM交互、数据验证等
"""

import json
import re
import time
import os
import csv
import requests
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field, create_model, ValidationError
from typing import Optional as TypingOptional

from .config import Config
from .llm_client import LLMClient
from .pdf_processor import PDFProcessor
from .field_inference import FieldInference, DynamicFieldsResponse
from .task_manager import TaskManager


class PageExtractionResponse(BaseModel):
    """页面提取响应模型"""
    data: List[BaseModel] = Field(default=[], description="提取到的文献数据实体列表")


class ExtractionEngine:
    """
    提取引擎类 - 负责PDF文献提取的核心逻辑

    职责：
    - 动态字段生成和验证
    - PDF处理和图像转换
    - LLM交互和数据提取
    - 数据验证和清理
    - 任务进度跟踪
    """

    def __init__(self, task_manager: TaskManager):
        """初始化提取引擎"""
        self.config = Config()
        self.llm_client = LLMClient()
        self.pdf_processor = PDFProcessor()
        self.field_inference = FieldInference()
        self.task_manager = task_manager

    def infer_fields(self, task_description: str) -> Tuple[bool, List[str] | str]:
        """
        推断提取字段

        Args:
            task_description: 任务描述

        Returns:
            (成功状态, 字段列表或错误信息)
        """
        return self.field_inference.infer_fields(task_description)

    def get_filename_prefix(self, task_description: str) -> str:
        """
        获取文件名前缀

        Args:
            task_description: 任务描述

        Returns:
            文件名前缀
        """
        return self.field_inference.get_filename_prefix(task_description)

    def process_pdf_library(self, task_id: str, task_description: str, fields: List[str]) -> None:
        """
        处理PDF文献库

        Args:
            task_id: 任务ID
            task_description: 任务描述
            fields: 提取字段
        """
        try:
            self.task_manager.start_task(task_id)
            self.task_manager.put_task_message("info", f"🚀 提取任务启动！目标：【{task_description}】")

            # 准备目录
            save_dir = self.config.EXTRACT_DIR
            os.makedirs(save_dir, exist_ok=True)
            prefix = self.get_filename_prefix(task_description)

            # 验证PDF文件夹
            if not os.path.exists(self.config.PDF_FOLDER):
                self.task_manager.put_task_message("error", f"找不到文件夹: {self.config.PDF_FOLDER}")
                self.task_manager.fail_task(task_id, "PDF文件夹不存在")
                return

            # 创建动态模型
            DynamicRecord = self.field_inference.create_dynamic_model(fields)

            # 包装成最终响应格式
            class LocalExtractionResponse(BaseModel):
                data: List[DynamicRecord] = Field(default=[], description="提取到的文献数据实体列表")

            schema_str = json.dumps(LocalExtractionResponse.model_json_schema(), ensure_ascii=False)

            # 获取PDF文件列表
            pdf_files = self.pdf_processor.list_pdf_files()
            total_files = len(pdf_files)
            all_extracted_data = []

            self.task_manager.put_task_message("progress", f"发现 {total_files} 篇PDF文献")

            # 处理每个PDF文件
            for file_idx, pdf_path in enumerate(pdf_files):
                if self.task_manager.is_task_cancelled():
                    self.task_manager.put_task_message("info", "⚠️ 接收到停止指令！正在终止并保存当前数据...")
                    break

                self._process_single_pdf(
                    pdf_path=pdf_path,
                    file_idx=file_idx,
                    total_files=total_files,
                    fields=fields,
                    schema_str=schema_str,
                    all_extracted_data=all_extracted_data,
                    task_id=task_id
                )

            # 保存结果
            self._save_extraction_results(
                task_id=task_id,
                all_extracted_data=all_extracted_data,
                fields=fields,
                prefix=prefix
            )

            # 完成任务
            self.task_manager.complete_task(task_id, {
                "csv": f"{save_dir}/{prefix}_{time.strftime('%Y%m%d-%H%M%S')}.csv",
                "count": len(all_extracted_data),
                "fields": fields
            })

        except Exception as e:
            self.task_manager.fail_task(task_id, str(e))
            self.task_manager.put_task_message("error", f"提取任务失败: {str(e)}")

    def _process_single_pdf(
        self,
        pdf_path: str,
        file_idx: int,
        total_files: int,
        fields: List[str],
        schema_str: str,
        all_extracted_data: List[Dict[str, Any]],
        task_id: str
    ) -> None:
        """
        处理单个PDF文件

        Args:
            pdf_path: PDF文件路径
            file_idx: 文件索引
            total_files: 总文件数
            fields: 提取字段
            schema_str: Schema字符串
            all_extracted_data: 所有提取数据
            task_id: 任务ID
        """
        filename = os.path.basename(pdf_path)
        doc_id = os.path.splitext(filename)[0]

        try:
            # 获取PDF信息
            pdf_info = self.pdf_processor.get_pdf_info(pdf_path)
            if not pdf_info:
                self.task_manager.put_task_message("error", f"无法读取PDF信息: {filename}")
                return

            num_pages = pdf_info['num_pages']
            self.task_manager.put_task_message("progress", f"正在处理第 {file_idx + 1}/{total_files} 篇: {filename}")

            # 处理每一页
            for page_num in range(num_pages):
                if self.task_manager.is_task_cancelled():
                    break

                self._process_single_page(
                    pdf_path=pdf_path,
                    page_num=page_num,
                    doc_id=doc_id,
                    fields=fields,
                    schema_str=schema_str,
                    all_extracted_data=all_extracted_data,
                    task_id=task_id
                )

        except Exception as e:
            self.task_manager.put_task_message("error", f"处理文件 {filename} 失败: {str(e)}")

    def _process_single_page(
        self,
        pdf_path: str,
        page_num: int,
        doc_id: str,
        fields: List[str],
        schema_str: str,
        all_extracted_data: List[Dict[str, Any]],
        task_id: str
    ) -> None:
        """
        处理单个PDF页面（支持混合提取模式）

        Args:
            pdf_path: PDF文件路径
            page_num: 页码
            doc_id: 文档ID
            fields: 提取字段
            schema_str: Schema字符串
            all_extracted_data: 所有提取数据
            task_id: 任务ID
        """
        try:
            # 获取提取模式
            extraction_mode = self.pdf_processor.get_extraction_mode()

            # 提取页面内容
            markdown_text, img_base64, use_vision = self.pdf_processor.extract_page_content(
                pdf_path, page_num, extraction_mode
            )

            # 根据模式选择处理方式
            if use_vision:
                # 使用Vision API处理
                self._process_with_vision(
                    pdf_path=pdf_path,
                    page_num=page_num,
                    doc_id=doc_id,
                    fields=fields,
                    schema_str=schema_str,
                    all_extracted_data=all_extracted_data,
                    task_id=task_id,
                    img_base64=img_base64,
                    markdown_text=markdown_text
                )
            else:
                # 使用文本API处理
                self._process_with_text(
                    pdf_path=pdf_path,
                    page_num=page_num,
                    doc_id=doc_id,
                    fields=fields,
                    schema_str=schema_str,
                    all_extracted_data=all_extracted_data,
                    task_id=task_id,
                    markdown_text=markdown_text
                )

        except Exception as e:
            self.task_manager.put_task_message("error", f"处理页面 {page_num + 1} 失败: {str(e)}")

    def _process_with_vision(
        self,
        pdf_path: str,
        page_num: int,
        doc_id: str,
        fields: List[str],
        schema_str: str,
        all_extracted_data: List[Dict[str, Any]],
        task_id: str,
        img_base64: str,
        markdown_text: Optional[str] = None
    ) -> None:
        """使用Vision API处理页面"""
        if not img_base64:
            self.task_manager.put_task_message("error", f"无法转换页面 {page_num + 1} 为图片")
            return

        self.task_manager.put_task_message("page_reading", {
            "filename": os.path.basename(pdf_path),
            "page": page_num + 1,
            "image": img_base64
        })

        # 构建示例JSON
        example_item = {f: "提取的内容" for f in fields}
        example_json = json.dumps({"data": [example_item]}, ensure_ascii=False)

        # 构建系统提示词
        sys_prompt = (
            f"你是一个专业的学术文献分析专家。你的任务是从提供的文献页面图像中提取：\n【目标】：{fields}\n\n"
            "提取规则：\n"
            "1. 复合材料（含+、and等）不可拆分，需提取比例，若无比例标注（未说明比例）。若已提取过则不重复。\n"
            "2. 溶剂量/浓度/转速/温度必须包含单位。\n"
            "3. 忽略参考文献条目中的数据。\n\n"
            "🚨 你必须直接输出一个 JSON 对象，绝不要包含 Markdown 标记（如 ```json）或任何其他解释性文字！\n"
            f"🚨 必须严格遵循以下 JSON 格式：\n{example_json}"
        )

        # 构建消息
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}]}
        ]

        # 调用LLM API
        result = self._call_vision_api_with_stream(
            messages=messages,
            schema_str=schema_str,
            page_num=page_num,
            filename=os.path.basename(pdf_path),
            task_id=task_id
        )

        # 处理结果
        if result and isinstance(result, dict) and "data" in result:
            for item in result["data"]:
                item_dict = item if isinstance(item, dict) else item.model_dump()
                item_dict['_source_doc'] = doc_id
                all_extracted_data.append(item_dict)

                self.task_manager.put_task_message("finding", {
                    "page": page_num + 1,
                    "filename": os.path.basename(pdf_path),
                    "details": item_dict
                })

    def _process_with_text(
        self,
        pdf_path: str,
        page_num: int,
        doc_id: str,
        fields: List[str],
        schema_str: str,
        all_extracted_data: List[Dict[str, Any]],
        task_id: str,
        markdown_text: str
    ) -> None:
        """使用文本API处理页面"""
        if not markdown_text or len(markdown_text.strip()) < 50:
            # 文本太少，跳过
            return

        self.task_manager.put_task_message("info", f"📄 使用文本模式处理第 {page_num + 1} 页（节省成本）")

        # 构建示例JSON
        example_item = {f: "提取的内容" for f in fields}
        example_json = json.dumps({"data": [example_item]}, ensure_ascii=False)

        # 构建系统提示词
        sys_prompt = (
            f"你是一个专业的学术文献分析专家。你的任务是从提供的文献页面文本中提取：\n【目标】：{fields}\n\n"
            "提取规则：\n"
            "1. 复合材料（含+、and等）不可拆分，需提取比例，若无比例标注（未说明比例）。若已提取过则不重复。\n"
            "2. 溶剂量/浓度/转速/温度必须包含单位。\n"
            "3. 忽略参考文献条目中的数据。\n\n"
            "🚨 你必须直接输出一个 JSON 对象，绝不要包含 Markdown 标记（如 ```json）或任何其他解释性文字！\n"
            f"🚨 必须严格遵循以下 JSON 格式：\n{example_json}"
        )

        # 构建消息
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"文献页面内容：\n\n{markdown_text}"}
        ]

        # 调用文本API
        result = self._call_text_api_with_stream(
            messages=messages,
            schema_str=schema_str,
            page_num=page_num,
            filename=os.path.basename(pdf_path),
            task_id=task_id
        )

        # 处理结果
        if result and isinstance(result, dict) and "data" in result:
            for item in result["data"]:
                item_dict = item if isinstance(item, dict) else item.model_dump()
                item_dict['_source_doc'] = doc_id
                all_extracted_data.append(item_dict)

                self.task_manager.put_task_message("finding", {
                    "page": page_num + 1,
                    "filename": os.path.basename(pdf_path),
                    "details": item_dict
                })

    def _call_vision_api_with_stream(
        self,
        messages: List[Dict[str, Any]],
        schema_str: str,
        page_num: int,
        filename: str,
        task_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        调用视觉API（带流式响应）

        Args:
            messages: 消息列表
            schema_str: Schema字符串
            page_num: 页码
            filename: 文件名
            task_id: 任务ID

        Returns:
            API响应或None
        """
        headers = self.llm_client.get_default_headers()

        payload = {
            "model": self.config.MODEL_NAME_VL,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1024,
            "stream": True
        }

        max_retries = self.config.MAX_RETRIES

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.config.API_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.config.STREAM_TIMEOUT,
                    stream=True
                )
                response.raise_for_status()

                self.task_manager.put_task_message("reading_start")

                result_text = ""
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                chunk_json = json.loads(data_str)
                                content = chunk_json['choices'][0]['delta'].get('content', '')
                                if content:
                                    result_text += content
                                    self.task_manager.put_task_message("reading_chunk", content)
                            except Exception:
                                pass

                # 清理和解析结果
                return self._parse_llm_response(result_text, schema_str)

            except requests.exceptions.Timeout:
                self.task_manager.put_task_message("error", f"⚠️ 第 {page_num + 1} 页 (第{attempt + 1}次尝试) API 请求超时！")
            except Exception as e:
                self.task_manager.put_task_message("error", f"⚠️ 第 {page_num + 1} 页 (第{attempt + 1}次尝试) 解析失败: {str(e)}")

            if attempt < max_retries - 1:
                time.sleep(2.0)

        return None

    def _call_text_api_with_stream(
        self,
        messages: List[Dict[str, Any]],
        schema_str: str,
        page_num: int,
        filename: str,
        task_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        调用文本API（带流式响应）

        Args:
            messages: 消息列表
            schema_str: Schema字符串
            page_num: 页码
            filename: 文件名
            task_id: 任务ID

        Returns:
            API响应或None
        """
        headers = self.llm_client.get_default_headers()

        payload = {
            "model": self.config.MODEL_NAME_TALK,  # 使用文本模型
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1024,
            "stream": True
        }

        max_retries = self.config.MAX_RETRIES

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.config.API_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.config.STREAM_TIMEOUT,
                    stream=True
                )
                response.raise_for_status()

                self.task_manager.put_task_message("reading_start")

                result_text = ""
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                chunk_json = json.loads(data_str)
                                content = chunk_json['choices'][0]['delta'].get('content', '')
                                if content:
                                    result_text += content
                                    self.task_manager.put_task_message("reading_chunk", content)
                            except Exception:
                                pass

                # 清理和解析结果
                return self._parse_llm_response(result_text, schema_str)

            except requests.exceptions.Timeout:
                self.task_manager.put_task_message("error", f"⚠️ 第 {page_num + 1} 页 (第{attempt + 1}次尝试) API 请求超时！")
            except Exception as e:
                self.task_manager.put_task_message("error", f"⚠️ 第 {page_num + 1} 页 (第{attempt + 1}次尝试) 解析失败: {str(e)}")

            if attempt < max_retries - 1:
                time.sleep(2.0)

        return None

    def _parse_llm_response(self, result_text: str, schema_str: str) -> Optional[Dict[str, Any]]:
        """
        解析LLM响应

        Args:
            result_text: 结果文本
            schema_str: Schema字符串

        Returns:
            解析后的数据或None
        """
        try:
            # 清理文本
            print(f"\n--- 模型原始输出 ---\n{result_text}\n-----------------------")

            # 提取JSON
            json_match = re.search(r'(\{.*\}|\[.*\])', result_text, re.DOTALL)
            if json_match:
                clean_text = json_match.group(1).strip()
            else:
                clean_text = result_text.strip()

            # 处理数组格式
            if clean_text.startswith('['):
                clean_text = f'{{"data": {clean_text}}}'

            # 验证JSON
            parsed_res = json.loads(clean_text)
            return parsed_res

        except Exception as e:
            print(f"解析LLM响应失败: {e}")
            return None

    def _save_extraction_results(
        self,
        task_id: str,
        all_extracted_data: List[Dict[str, Any]],
        fields: List[str],
        prefix: str
    ) -> None:
        """
        保存提取结果

        Args:
            task_id: 任务ID
            all_extracted_data: 所有提取数据
            fields: 提取字段
            prefix: 文件名前缀
        """
        # 确定所有字段
        all_keys = set(fields)
        for d in all_extracted_data:
            all_keys.update(d.keys())
        all_keys = list(all_keys)

        # 保存到extract目录
        csv_filename = os.path.join(self.config.EXTRACT_DIR, f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        self._write_csv(csv_filename, all_extracted_data, all_keys)

        # 保存到temporal目录
        os.makedirs(self.config.TEMPORAL_DIR, exist_ok=True)
        csv_filename_temporal = os.path.join(self.config.TEMPORAL_DIR, "extraction.csv")
        self._write_csv(csv_filename_temporal, all_extracted_data, all_keys)

        self.task_manager.put_task_message("complete", {
            "csv": csv_filename_temporal,
            "count": len(all_extracted_data),
            "fields": fields
        })

    def _write_csv(self, filename: str, data: List[Dict[str, Any]], fieldnames: List[str]) -> None:
        """
        写入CSV文件

        Args:
            filename: 文件名
            data: 数据
            fieldnames: 字段名
        """
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)