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

from core.config import Config
from core.llm_client import LLMClient
from .pdf_processor import PDFProcessor
from core.field_inference import FieldInference, DynamicFieldsResponse
from core.task_manager import TaskManager
from .embedding_service import create_embedding_service
from .vector_store import ChromaVectorStore
from .page_indexer import PageIndexer, make_page_id
from .page_filter import PageFilter
from .few_shot_retriever import FewShotRetriever
from .dedup import deduplicate_extraction_results
from .evidence_validator import EvidenceValidator
from .semantic_dedup import SemanticDedup
from prompts import create_prompt_manager


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

    def __init__(self, task_manager: TaskManager, session_path: str = None, temporal_dir: str = None):
        """
        初始化提取引擎

        Args:
            task_manager: 任务管理器
            session_path: 会话基础路径，如果为None则使用默认路径
            temporal_dir: 全局 temporal 目录，如果为None则使用默认路径
        """
        self.config = Config()
        self.llm_client = LLMClient()
        self.pdf_processor = PDFProcessor()
        self.field_inference = FieldInference()
        self.task_manager = task_manager
        self.session_path = session_path
        self.temporal_dir = temporal_dir
        # Phase 1: 页面预筛选（延迟初始化，在 process_pdf_library 中按需创建）
        self.page_filter: Optional[PageFilter] = None
        self.page_indexer: Optional[PageIndexer] = None
        # Phase 2: Few-Shot 检索（延迟初始化）
        self.few_shot_retriever: Optional[FewShotRetriever] = None
        self.embedding_service = None
        self.vector_store = None
        # Step 2: Evidence Validator (验证 grounding 真实性)
        self.evidence_validator: Optional[EvidenceValidator] = None
        if self.config.EVIDENCE_VALIDATION_ENABLED:
            self.evidence_validator = EvidenceValidator(
                fuzzy_threshold=self.config.EVIDENCE_FUZZY_THRESHOLD
            )

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
            # 1. 创建 Embedding 服务（如果外部已注入则复用）
            if self.embedding_service is None:
                self.embedding_service = create_embedding_service()

            # 2. 创建 ChromaDB 向量存储（如果外部已注入则复用）
            chroma_dir = self.config.CHROMADB_PERSIST_DIR
            if self.vector_store is None:
                self.vector_store = ChromaVectorStore(persist_dir=chroma_dir, expected_dim=self.config.EMBEDDING_DIM)

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

    def infer_fields(self, task_description: str, history: list = None) -> Tuple[bool, List[str] | str]:
        """
        推断提取字段

        Args:
            task_description: 任务描述
            history: 对话历史

        Returns:
            (成功状态, 字段列表或错误信息)
        """
        return self.field_inference.infer_fields(task_description, history)

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

            # ── 内置质量检查：稀疏记录 + 重复记录 ──
            try:
                from extract.quality_checker import QualityChecker
                qc_result = QualityChecker.run_all_checks(
                    all_extracted_data, fields,
                    sparse_threshold=getattr(self.config, 'QUALITY_SPARSE_THRESHOLD', 0.3)
                )

                deleted = set(qc_result.get("sparse_deleted", []) + qc_result.get("duplicate_deleted", []))
                if deleted:
                    # 记录被删除的条目摘要（用于日志）
                    sparse_info = qc_result.get("sparse_rate", {})
                    for idx in sorted(deleted):
                        reason = "稀疏" if idx in qc_result.get("sparse_deleted", []) else "重复"
                        record = all_extracted_data[idx]
                        name_val = str(record.get(fields[0], "?")) if fields else "?"
                        fill_rate = sparse_info.get(idx, 0)
                        print(f"[质量检查] 删除{reason}记录 #{idx}: {name_val} (填充率={fill_rate:.0%})" if reason == "稀疏" else f"[质量检查] 删除{reason}记录 #{idx}: {name_val}")

                    all_extracted_data = [
                        r for i, r in enumerate(all_extracted_data) if i not in deleted
                    ]

                    self.task_manager.put_task_message("info",
                        f"质量检查完成: 删除 {len(qc_result.get('sparse_deleted', []))} 条稀疏记录, "
                        f"{len(qc_result.get('duplicate_deleted', []))} 条重复记录"
                    )
            except ImportError:
                pass  # quality_checker 模块不存在时跳过
            except Exception as e:
                print(f"[质量检查] 执行失败: {e}")
                import traceback
                traceback.print_exc()

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
            "pdf_path": pdf_path.replace('\\', '/'),
            "page": page_num + 1,
            "image": img_base64
        })

        # 构建示例JSON
        example_item = {f: "提取的内容" for f in fields}
        example_json = json.dumps({"data": [example_item]}, ensure_ascii=False)

        # 构建系统提示词 (migrated from inline f-string, source: lines 397-411)
        prompt_manager = create_prompt_manager(lang='zh')
        sys_prompt = prompt_manager.get(
            "extraction_system_vision",
            task_description=task_description,
            fields=str(fields),
            example_json=example_json,
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
                item_dict['_source_page'] = page_num + 1
                # Step 2: Evidence validation
                self._annotate_evidence(item_dict, pdf_path, page_num)
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

        # 构建系统提示词 (migrated from inline f-string, source: lines 471-485)
        prompt_manager = create_prompt_manager(lang='zh')
        sys_prompt = prompt_manager.get(
            "extraction_system_text",
            task_description=task_description,
            fields=str(fields),
            example_json=example_json,
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
                item_dict['_source_page'] = page_num + 1
                # Step 2: Evidence validation
                self._annotate_evidence(item_dict, pdf_path, page_num)
                all_extracted_data.append(item_dict)

                self.task_manager.put_task_message("finding", {
                    "page": page_num + 1,
                    "filename": os.path.basename(pdf_path),
                    "details": item_dict
                })

            # Phase 2: 保存提取结果到历史数据库
            self._save_to_extraction_history(pdf_path, page_num, result["data"],
                                             task_description, doc_id)

    def _annotate_evidence(
        self,
        item_dict: Dict[str, Any],
        pdf_path: str,
        page_num: int,
    ) -> None:
        """
        Step 2: 为单条记录添加 evidence 校验注解。

        从 item_dict 中读取 "原文原句" 字段, 调 EvidenceValidator 验证其是否
        真实出现在该页 PDF 文本中. 验证结果以 `_evidence_offset` /
        `_evidence_length` / `_evidence_score` 写入 item_dict, 失败时设
        `_low_confidence=True`.

        Args:
            item_dict:  LLM 返回的单条记录 (会被原地修改)
            pdf_path:   当前 PDF 路径
            page_num:   0-based 页码
        """
        if not self.evidence_validator:
            return

        evidence = str(item_dict.get("原文原句", "") or "").strip()
        if not evidence:
            # 缺失 grounding 字段 → 标灰
            item_dict["_low_confidence"] = True
            item_dict["_evidence_score"] = 0.0
            return

        try:
            page_text = self.pdf_processor.extract_text_from_page(pdf_path, page_num) or ""
        except Exception:
            page_text = ""

        validation = self.evidence_validator.validate(page_text, evidence)
        item_dict["_evidence_offset"] = validation["offset"]
        item_dict["_evidence_length"] = validation["length"]
        item_dict["_evidence_score"] = validation["fuzzy_score"]

        if not validation["valid"]:
            item_dict["_low_confidence"] = True

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
        # 构建 few-shot 块 (migrated from inline concatenation, source: lines 553-557)
        prompt_manager = create_prompt_manager(lang='zh')
        few_shot_block = prompt_manager.get(
            "extraction_few_shot_block",
            examples_text=examples_text,
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
        headers = {
            "Authorization": f"Bearer {self.config.VL_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.MODEL_NAME_VL,
            "messages": messages,
            "temperature": 0.1,
            "stream": True
        }
        if self.config.MAX_TOKENS is not None:
            payload["max_tokens"] = self.config.MAX_TOKENS
        # merge VL extra_body
        _vl_extra = self.config.get_extra_body("VL")
        payload.update(_vl_extra)

        max_retries = self.config.MAX_RETRIES

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.config.VL_API_URL,
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
                                delta = chunk_json['choices'][0].get('delta', {})
                                reasoning = delta.get('reasoning_content', '')
                                content = delta.get('content', '')
                                if reasoning:
                                    result_text += reasoning
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
        headers = {
            "Authorization": f"Bearer {self.config.TALK_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.MODEL_NAME_TALK,
            "messages": messages,
            "temperature": 0.1,
            "stream": True
        }
        if self.config.MAX_TOKENS is not None:
            payload["max_tokens"] = self.config.MAX_TOKENS
        # merge TALK extra_body
        _talk_extra = self.config.get_extra_body("TALK")
        payload.update(_talk_extra)

        max_retries = self.config.MAX_RETRIES

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.config.TALK_API_URL,
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
                                delta = chunk_json['choices'][0].get('delta', {})
                                reasoning = delta.get('reasoning_content', '')
                                content = delta.get('content', '')
                                if reasoning:
                                    result_text += reasoning
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
        解析LLM响应（带多重回退策略）

        Args:
            result_text: 结果文本
            schema_str: Schema字符串

        Returns:
            解析后的数据或None
        """
        if not result_text or not result_text.strip():
            return None

        print(f"\n--- 模型原始输出 ---\n{result_text[:500]}{'...(truncated)' if len(result_text) > 500 else ''}\n-----------------------")

        # 策略1: 标准提取
        clean_text = self._extract_json_text(result_text)
        if clean_text:
            try:
                return json.loads(clean_text)
            except json.JSONDecodeError:
                pass

        # 策略2: 修复常见JSON错误后重试
        if clean_text:
            fixed = self._fix_common_json_errors(clean_text)
            if fixed:
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass

        # 策略3: 尝试找到最外层 { } 并手动修复
        try:
            return self._extract_json_heuristic(result_text)
        except Exception:
            pass

        print(f"解析LLM响应失败: 所有策略均无法解析")
        return None

    @staticmethod
    def _extract_json_text(text: str) -> Optional[str]:
        """从LLM输出中提取JSON文本"""
        # 移除 markdown 代码块
        cleaned = re.sub(r'```(?:json)?\s*', '', text)
        cleaned = re.sub(r'```\s*$', '', cleaned)

        # 尝试匹配最外层 { }
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            json_str = match.group(0)
            # 处理数组格式
            if json_str.strip().startswith('['):
                json_str = f'{{"data": {json_str.strip()}}}'
            return json_str.strip()
        return None

    @staticmethod
    def _fix_common_json_errors(text: str) -> Optional[str]:
        """修复LLM输出中常见的JSON格式错误"""
        try:
            fixed = text

            # 1. 移除尾部多余逗号
            fixed = re.sub(r',\s*}', '}', fixed)
            fixed = re.sub(r',\s*]', ']', fixed)

            # 2. 修复字符串值中的未转义换行符
            # 在JSON字符串值内，\n 才合法，实际换行不合法
            in_string = False
            escape_next = False
            chars = list(fixed)
            result = []
            for c in chars:
                if escape_next:
                    result.append(c)
                    escape_next = False
                    continue
                if c == '\\':
                    result.append(c)
                    escape_next = True
                    continue
                if c == '"':
                    in_string = not in_string
                    result.append(c)
                    continue
                if in_string and c == '\n':
                    result.append('\\n')
                    continue
                if in_string and c == '\r':
                    continue  # skip \r
                if in_string and c == '\t':
                    result.append('\\t')
                    continue
                result.append(c)

            fixed = ''.join(result)

            # 3. 修复字符串值中的未转义双引号（常见于中文引号混合）
            # 中文双引号 "" 替换为单引号避免JSON冲突
            fixed = fixed.replace('“', "'").replace('”', "'")

            return fixed
        except Exception:
            return None

    @staticmethod
    def _extract_json_heuristic(text: str) -> Optional[Dict[str, Any]]:
        """启发式提取：在大文本中找到 {'data': [...]} 结构"""
        # 查找 "data": 关键词附近的结构
        data_match = re.search(r'"data"\s*:\s*\[', text)
        if not data_match:
            return None

        # 从 data 开始处找完整的 [ ... ] 块
        start = data_match.start()
        bracket_start = text.index('[', data_match.end() - 1)
        depth = 0
        for i in range(bracket_start, len(text)):
            if text[i] == '[':
                depth += 1
            elif text[i] == ']':
                depth -= 1
                if depth == 0:
                    array_str = text[bracket_start:i + 1]
                    # 对整个数组做基础修复
                    array_str = re.sub(r',\s*]', ']', array_str)
                    full_json = f'{{"data": {array_str}}}'
                    return json.loads(full_json)
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
        # Step 3: 全空行过滤 (主键空或 grounding 空)
        if all_extracted_data and fields:
            original_count = len(all_extracted_data)
            primary_key = fields[0]
            evidence_field = "原文原句" if "原文原句" in fields else None
            empty_markers = {"", "无", "未提及", "N/A", "-", "--"}

            valid_data: List[Dict[str, Any]] = []
            for record in all_extracted_data:
                primary = str(record.get(primary_key, "") or "").strip()
                if not primary or primary in empty_markers:
                    continue  # 主键空 → 丢弃
                if evidence_field:
                    evidence = str(record.get(evidence_field, "") or "").strip()
                    if not evidence or evidence in empty_markers:
                        continue  # grounding 空 → 丢弃
                valid_data.append(record)

            if len(valid_data) < original_count:
                self.task_manager.put_task_message(
                    "info",
                    f"全空行过滤: {original_count} → {len(valid_data)} "
                    f"(删 {original_count - len(valid_data)} 条无主键/无原文)"
                )
            all_extracted_data = valid_data

        # 去重：按实体名称（fields[0]）合并重复行
        if self.config.DEDUP_ENABLED and all_extracted_data and fields:
            original_count = len(all_extracted_data)
            all_extracted_data = deduplicate_extraction_results(
                all_extracted_data,
                fields,
                normalize=self.config.DEDUP_NORMALIZE,
                merge_strategy=self.config.DEDUP_MERGE_STRATEGY,
                add_metadata=self.config.DEDUP_ADD_METADATA,
            )
            deduped_count = len(all_extracted_data)
            if original_count != deduped_count:
                self.task_manager.put_task_message(
                    "info",
                    f"去重完成: {original_count} 条 → {deduped_count} 条 (移除 {original_count - deduped_count} 条重复)"
                )

        # Step 4: 语义去重 (embedding 聚类) — 规则 dedup 之后
        if (
            self.config.SEMANTIC_DEDUP_ENABLED
            and self.embedding_service is not None
            and all_extracted_data
            and fields
            and len(all_extracted_data) >= 2
        ):
            try:
                sem_dedup = SemanticDedup(
                    self.embedding_service,
                    similarity_threshold=self.config.SEMANTIC_DEDUP_THRESHOLD,
                    merge_strategy=self.config.DEDUP_MERGE_STRATEGY,
                )
                before_count = len(all_extracted_data)
                all_extracted_data = sem_dedup.cluster_and_merge(all_extracted_data, fields)
                after_count = len(all_extracted_data)
                if before_count != after_count:
                    self.task_manager.put_task_message(
                        "info",
                        f"语义去重完成: {before_count} 条 → {after_count} 条 (合并 {before_count - after_count} 条同义)"
                    )
            except Exception as e:
                # 语义去重失败不应阻塞主流程
                self.task_manager.put_task_message(
                    "warning",
                    f"语义去重失败 (已跳过): {e}"
                )

        # Step 5: Fresh LLM Review Agent (兜底审查)
        if (
            self.config.REVIEW_AGENT_ENABLED
            and all_extracted_data
            and fields
        ):
            try:
                from core.review_agent import ExtractionReviewAgent
                review_agent = ExtractionReviewAgent()
                before_n = len(all_extracted_data)
                all_extracted_data = review_agent.review(all_extracted_data, fields)
                n_dup = sum(1 for r in all_extracted_data if r.get("_review_flag") == "duplicate")
                n_low = sum(1 for r in all_extracted_data if r.get("_review_flag") == "low_value")
                self.task_manager.put_task_message(
                    "info",
                    f"LLM 审查完成: 共 {before_n} 条, 重复 {n_dup} 条, 低置信 {n_low} 条"
                )
            except Exception as e:
                self.task_manager.put_task_message(
                    "warning",
                    f"LLM 审查失败 (已跳过): {e}"
                )

        # 确定所有字段
        all_keys = set(fields)
        for d in all_extracted_data:
            all_keys.update(d.keys())
        all_keys = list(all_keys)

        # 保存到extract目录（使用会话路径）
        if self.session_path:
            extract_dir = os.path.join(self.session_path, "extract")
        else:
            extract_dir = self.config.EXTRACT_DIR
        # temporal 使用全局共享目录
        temporal_dir = self.temporal_dir or self.config.TEMPORAL_DIR

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