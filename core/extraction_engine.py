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
from .embedding_service import create_embedding_service
from .vector_store import ChromaVectorStore
from .page_indexer import PageIndexer, make_page_id
from .page_filter import PageFilter
from .few_shot_retriever import FewShotRetriever


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

    def __init__(self, task_manager: TaskManager, session_path: str = None):
        """
        初始化提取引擎

        Args:
            task_manager: 任务管理器
            session_path: 会话基础路径，如果为None则使用默认路径
        """
        self.config = Config()
        self.llm_client = LLMClient()
        self.pdf_processor = PDFProcessor()
        self.field_inference = FieldInference()
        self.task_manager = task_manager
        self.session_path = session_path
        # Phase 1: 页面预筛选（延迟初始化，在 process_pdf_library 中按需创建）
        self.page_filter: Optional[PageFilter] = None
        self.page_indexer: Optional[PageIndexer] = None
        # Phase 2: Few-Shot 检索（延迟初始化）
        self.few_shot_retriever: Optional[FewShotRetriever] = None
        self.embedding_service = None
        self.vector_store = None

    def _init_page_filter_services(self):
        """
        按需初始化 Phase 1 + Phase 2 所需的所有服务

        初始化链路：
          Config → EmbeddingService → ChromaVectorStore
            → PageIndexer (预索引) + PageFilter (查询时过滤)
            → FewShotRetriever (Phase 2 历史示例检索)

        异常处理：
          如果任何初始化步骤失败（如 API key 未配置、ChromaDB 无法写入），
          则优雅降级：self.page_filter = None，后续页面处理不进行筛选。
          Phase 2 的 FewShotRetriever 也会同步降级。

        这样设计是为了保证：
          - 即使 embedding 服务不可用，提取任务仍可正常运行
          - 新用户无需配置 API key 即可使用其他功能
        """
        if not self.config.PAGE_FILTER_ENABLED:
            return

        try:
            # 1. 创建 Embedding 服务
            self.embedding_service = create_embedding_service()

            # 2. 创建 ChromaDB 向量存储
            chroma_dir = self.config.CHROMADB_PERSIST_DIR
            self.vector_store = ChromaVectorStore(persist_dir=chroma_dir)

            # 3. 创建页面索引器（用于一次性预索引 PDF 文献库）
            sqlite_path = os.path.join(chroma_dir, "page_metadata.db")
            self.page_indexer = PageIndexer(
                self.embedding_service, self.vector_store, sqlite_path, self.pdf_processor
            )

            # 4. 创建页面筛选器（用于查询时逐页判断相关性）
            self.page_filter = PageFilter(
                self.embedding_service, self.vector_store,
                threshold=self.config.PAGE_FILTER_THRESHOLD,
                top_k=self.config.PAGE_FILTER_TOP_K
            )

            # 5. Phase 2: 创建 Few-Shot 检索器
            if self.config.FEW_SHOT_ENABLED:
                fs_sqlite_path = os.path.join(chroma_dir, "extraction_history.db")
                self.few_shot_retriever = FewShotRetriever(
                    self.embedding_service, self.vector_store, fs_sqlite_path
                )
        except Exception as e:
            print(f"[PageFilter] 初始化失败: {e}，已禁用页面筛选和Few-Shot")
            self.page_filter = None
            self.page_indexer = None
            self.few_shot_retriever = None

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

            # 准备目录（使用会话路径）
            if self.session_path:
                save_dir = os.path.join(self.session_path, "extract")
            else:
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

            # ===== Phase 1: 页面预筛选初始化与索引 =====
            self._init_page_filter_services()
            if self.page_indexer:
                self.task_manager.put_task_message("info", "📊 正在索引 PDF 页面（首次运行较慢，后续增量更新）...")
                indexed, skipped = self.page_indexer.index_all_pdfs()
                self.task_manager.put_task_message(
                    "info", f"📊 页面索引完成: {indexed} 页新增, {skipped} 页跳过（内容未变更）"
                )
            if self.page_filter:
                # 将任务描述预嵌入为向量，后续每个页面直接比对
                self.page_filter.set_task(task_description)

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
                    task_id=task_id,
                    task_description=task_description
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
        task_id: str,
        task_description: str = ""
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
            task_description: 任务描述（Phase 1 页面预筛选用）
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

                # Phase 1: 页面预筛选 —— 跳过与任务不相关的页面
                if self.page_filter:
                    if not self.page_filter.should_process(pdf_path, page_num):
                        self.task_manager.put_task_message(
                            "info", f"⏭️ 跳过第{page_num + 1}页 (相似度低于阈值)"
                        )
                        continue

                self._process_single_page(
                    pdf_path=pdf_path,
                    page_num=page_num,
                    doc_id=doc_id,
                    fields=fields,
                    schema_str=schema_str,
                    all_extracted_data=all_extracted_data,
                    task_id=task_id,
                    task_description=task_description
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
        task_id: str,
        task_description: str = ""
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
            task_description: 任务描述（Phase 2 用于检索 Few-Shot 示例）
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
                    markdown_text=markdown_text,
                    task_description=task_description
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
                    markdown_text=markdown_text,
                    task_description=task_description
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
        markdown_text: Optional[str] = None,
        task_description: str = ""
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

        # Phase 2: 检索历史 Few-Shot 示例并注入 prompt
        sys_prompt = self._inject_few_shot_examples(sys_prompt, task_description, fields)

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

            # Phase 2: 保存提取结果到历史数据库
            self._save_to_extraction_history(pdf_path, page_num, result["data"],
                                             task_description, doc_id)

    def _process_with_text(
        self,
        pdf_path: str,
        page_num: int,
        doc_id: str,
        fields: List[str],
        schema_str: str,
        all_extracted_data: List[Dict[str, Any]],
        task_id: str,
        markdown_text: str,
        task_description: str = ""
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

        # Phase 2: 检索历史 Few-Shot 示例并注入 prompt
        sys_prompt = self._inject_few_shot_examples(sys_prompt, task_description, fields)

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

            # Phase 2: 保存提取结果到历史数据库
            self._save_to_extraction_history(pdf_path, page_num, result["data"],
                                             task_description, doc_id)

    def _inject_few_shot_examples(self, sys_prompt: str, task_description: str,
                                    fields: List[str]) -> str:
        """
        从历史提取记录中检索 Few-Shot 示例并注入到 system prompt

        如果 FewShotRetriever 未初始化或没有找到历史记录，则原样返回 sys_prompt。

        Args:
            sys_prompt: 原始系统提示词
            task_description: 当前提取任务描述
            fields: 当前提取字段列表

        Returns:
            可能添加了 Few-Shot 示例的新 sys_prompt
        """
        if not self.few_shot_retriever:
            return sys_prompt

        examples = self.few_shot_retriever.retrieve_examples(
            task_description, fields,
            top_k=self.config.FEW_SHOT_TOP_K
        )

        if not examples:
            return sys_prompt

        # 格式化示例为 JSON，每个一行
        examples_text = "\n".join(
            f"示例 {i + 1}: {json.dumps(ex, ensure_ascii=False)}"
            for i, ex in enumerate(examples)
        )
        few_shot_block = (
            "\n\n📋 参考历史提取示例（从相似页面中提取的数据，供你参考格式和内容）：\n"
            f"{examples_text}\n"
            "请参考以上示例的提取风格和详细程度来处理当前页面。"
        )

        return few_shot_block + "\n\n" + sys_prompt

    def _save_to_extraction_history(self, pdf_path: str, page_num: int,
                                     extracted_items: list,
                                     task_description: str, source_doc: str):
        """
        将提取结果保存到历史数据库供后续 Few-Shot 检索

        仅当 FewShotRetriever 已初始化时才执行保存。
        对 LLM 返回的 data 数组中的每一项分别保存。

        Args:
            pdf_path: PDF 文件路径
            page_num: 页码
            extracted_items: LLM 返回的 data 列表中的每一项（dict 或 Pydantic model）
            task_description: 当前任务描述
            source_doc: 来源文档名
        """
        if not self.few_shot_retriever:
            return

        page_id = make_page_id(pdf_path, page_num)

        for item in extracted_items:
            item_dict = item if isinstance(item, dict) else item.model_dump()
            # 保存的副本中去除内部字段
            clean_dict = {k: v for k, v in item_dict.items() if not k.startswith('_')}
            if not clean_dict:
                continue
            try:
                self.few_shot_retriever.save_extraction(
                    page_id, clean_dict, task_description, source_doc
                )
            except Exception:
                pass  # 保存失败不影响提取流程

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

        # 保存到extract目录（使用会话路径）
        if self.session_path:
            extract_dir = os.path.join(self.session_path, "extract")
            temporal_dir = os.path.join(self.session_path, "temporal")
        else:
            extract_dir = self.config.EXTRACT_DIR
            temporal_dir = self.config.TEMPORAL_DIR

        os.makedirs(extract_dir, exist_ok=True)
        csv_filename = os.path.join(extract_dir, f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        self._write_csv(csv_filename, all_extracted_data, all_keys)

        # 保存到temporal目录
        os.makedirs(temporal_dir, exist_ok=True)
        csv_filename_temporal = os.path.join(temporal_dir, "extraction.csv")
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