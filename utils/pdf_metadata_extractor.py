"""
PDF元数据提取器
负责从单篇PDF的前两页提取标题、摘要、创新点和关键图坐标
混合方案：规则提取关键图 → 回退视觉LLM
"""

import fitz
import json
import os
import re
import hashlib
from typing import Optional
from pydantic import BaseModel

from core.llm_client import LLMClient
from core.config import Config
from extract.pdf_processor import PDFProcessor


class ImageBBox(BaseModel):
    """关键图坐标信息"""
    page: int              # 所在页码 (1 或 2)
    x1: float              # 左上角 x (像素坐标)
    y1: float              # 左上角 y (像素坐标)
    x2: float              # 右下角 x (像素坐标)
    y2: float              # 右下角 y (像素坐标)
    description: str = ""  # 图片描述


class PDFMetadata(BaseModel):
    """单篇PDF提取的完整元数据"""
    title: str                           # 论文标题
    authors: str = ""                    # 作者（逗号分隔）
    abstract_summary: str = ""           # 简短摘要总结（中文，2-3句）
    innovation_points: list[str] = []    # 创新点列表
    key_image: Optional[ImageBBox] = None  # 关键图坐标
    doi: str = ""                        # TODO: 从PDF元数据提取
    arxiv_id: str = ""                   # TODO: 从PDF正文提取
    published_date: str = ""             # TODO: 从PDF元数据提取
    journal: str = ""                    # TODO: 从PDF元数据提取


class PDFMetadataExtractor:
    """
    PDF元数据提取器

    职责：
    - 调用LLM提取论文标题、摘要、创新点
    - 混合方案定位关键图（规则优先，回退视觉LLM）
    - 从PDF内部元数据提取DOI作为唯一ID
    - 生成安全的文件名
    """

    def __init__(self, config: Optional[Config] = None):
        """初始化提取器"""
        self.config = config or Config()
        self.llm_client = LLMClient(
            api_key=self.config.VL_API_KEY,
            api_url=self.config.VL_API_URL,
        )
        self.pdf_processor = PDFProcessor()

    # ---- 唯一ID生成 ----

    def extract_doi_from_pdf(self, pdf_path: str) -> Optional[str]:
        """从PDF内部元数据/XMP中提取DOI"""
        try:
            doc = fitz.open(pdf_path)
            # 尝试从标准元数据读取
            metadata = doc.metadata
            for key in ['doi', 'DOI', 'identifier']:
                if key in metadata and metadata[key]:
                    doi = metadata[key]
                    # 清理常见前缀
                    doi = re.sub(r'^.*doi[:/\s]*', '', doi, flags=re.IGNORECASE).strip()
                    if doi:
                        doc.close()
                        return doi

            # 尝试从 XMP 元数据读取
            xmp = doc.xref_get_key(-1, "DOI")
            if xmp and xmp[0] == 'string':
                doi = xmp[1].strip('/').strip('()')
                if doi:
                    doc.close()
                    return doi

            doc.close()
        except Exception as e:
            print(f"从PDF提取DOI失败: {e}")
        return None

    def generate_unique_id(self, pdf_path: str) -> str:
        """
        生成唯一ID
        优先级: DOI > 文件路径MD5
        """
        doi = self.extract_doi_from_pdf(pdf_path)
        if doi:
            return self._sanitize_id(doi)
        # 回退：文件路径的MD5前12位
        return hashlib.md5(pdf_path.encode('utf-8')).hexdigest()[:12]

    @staticmethod
    def _sanitize_id(raw: str) -> str:
        """清理DOI等标识符中的特殊字符"""
        # 保留字母数字、连字符、点号
        sanitized = re.sub(r'[^\w\-.]', '_', raw)
        return sanitized.strip('_') or "unknown_id"

    # ---- 文件名处理 ----

    @staticmethod
    def sanitize_title_for_filename(title: str, max_len: int = 80) -> str:
        """
        将论文标题转换为安全的文件名
        - 移除不允许的字符
        - 限制长度
        - 保留中英文字符
        """
        # 替换不允许的文件名字符
        illegal_chars = r'[<>:"/\\|?*]'
        sanitized = re.sub(illegal_chars, '', title)
        # 替换多个空格/制表符为单个空格
        sanitized = re.sub(r'\s+', ' ', sanitized)
        # 截断
        if len(sanitized) > max_len:
            sanitized = sanitized[:max_len].rsplit(' ', 1)[0]
        return sanitized.strip()

    # ---- 关键图提取（混合方案） ----

    def _extract_key_image(self, pdf_path: str) -> Optional[ImageBBox]:
        """
        混合方案提取关键图
        1. 规则提取：PyMuPDF提取前两页所有嵌入位图
        2. 0张 → None, 1张 → 直接采用, ≥2张 → 面积判断
        3. 面积接近 → 回退视觉LLM
        """
        try:
            doc = fitz.open(pdf_path)
            images = []  # [(page_num, xref, bbox, width, height, area)]

            for page_num in range(min(2, len(doc))):
                page = doc[page_num]
                # 获取页面上所有图片的引用
                image_list = page.get_images(full=True)
                for img_info in image_list:
                    xref = img_info[0]
                    # 获取图片在页面上的位置
                    img_rects = page.get_image_rects(xref)
                    if img_rects:
                        rect = img_rects[0]
                        w = rect.width
                        h = rect.height
                        # 过滤太小的图片（可能是图标、logo等）
                        if w * h < 5000:
                            continue
                        area = w * h
                        images.append({
                            'page': page_num + 1,  # 1-based
                            'xref': xref,
                            'x1': rect.x0,
                            'y1': rect.y0,
                            'x2': rect.x1,
                            'y2': rect.y1,
                            'width': w,
                            'height': h,
                            'area': area
                        })

            doc.close()

            if len(images) == 0:
                return None
            elif len(images) == 1:
                img = images[0]
                return ImageBBox(
                    page=img['page'], x1=img['x1'], y1=img['y1'],
                    x2=img['x2'], y2=img['y2'],
                    description="关键图（规则提取）"
                )
            else:
                # 按面积降序排列
                images.sort(key=lambda x: x['area'], reverse=True)
                largest = images[0]
                second = images[1]
                # 最大图面积 > 次大图 × 1.5 → 直接采用
                if largest['area'] > second['area'] * 1.5:
                    return ImageBBox(
                        page=largest['page'], x1=largest['x1'], y1=largest['y1'],
                        x2=largest['x2'], y2=largest['y2'],
                        description="关键图（规则提取，面积优势明确）"
                    )
                else:
                    # 面积接近 → 回退视觉LLM
                    print(f"多张候选图面积接近（{largest['area']:.0f} vs {second['area']:.0f}），回退视觉LLM判断")
                    return self._llm_identify_key_image(pdf_path)

        except Exception as e:
            print(f"规则提取关键图失败: {e}")
            return None

    def _llm_identify_key_image(self, pdf_path: str) -> Optional[ImageBBox]:
        """使用视觉LLM识别前两页中最有代表性的关键图"""
        try:
            doc = fitz.open(pdf_path)
            page1_img = self.pdf_processor.pdf_page_to_image(pdf_path, 0)
            page2_img = self.pdf_processor.pdf_page_to_image(pdf_path, 1) if len(doc) > 1 else None
            page1_w = doc[0].rect.width
            page1_h = doc[0].rect.height
            page2_w = doc[1].rect.width if len(doc) > 1 else 0
            page2_h = doc[1].rect.height if len(doc) > 1 else 0
            doc.close()

            if not page1_img:
                return None

            # 构建视觉消息
            image_messages = []
            image_messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{page1_img}", "detail": "high"}
            })
            if page2_img:
                image_messages.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{page2_img}", "detail": "high"}
                })

            image_messages.append({
                "type": "text",
                "text": f"""请从以上论文页面中找出最具有代表性的一张概述图（如系统架构图、方法示意图、Figure 1等，不要选实验结果图）。
页面1尺寸: {page1_w:.0f}x{page1_h:.0f} 像素
页面2尺寸: {page2_w:.0f}x{page2_h:.0f} 像素
返回JSON格式:
{{"page": 页码(1或2), "x1": 左上x(像素), "y1": 左上y(像素), "x2": 右下x(像素), "y2": 右下y(像素), "description": "图片描述"}}
如果找不到合适的概述图，返回 {{"page": 0}}"""
            })

            messages = [{"role": "user", "content": image_messages}]

            result = self.llm_client.call_api(
                model=self.config.METADATA_EXTRACTION_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=None,
                timeout=self.config.METADATA_EXTRACTION_TIMEOUT
            )

            if result:
                content = result['choices'][0]['message']['content']
                content = re.sub(r'```json\n|\n```|```', '', content).strip()
                data = json.loads(content)
                if data.get('page', 0) > 0:
                    return ImageBBox(**data)

        except Exception as e:
            print(f"视觉LLM关键图识别失败: {e}")

        return None

    # ---- 完整元数据提取 ----

    def extract_metadata(self, pdf_path: str) -> PDFMetadata:
        """
        提取单篇PDF的完整元数据
        输入：PDF文件路径
        输出：PDFMetadata对象（标题、摘要、创新点、关键图坐标等）
        调用一次视觉LLM完成全部文本提取
        """
        try:
            doc = fitz.open(pdf_path)
            if len(doc) < 1:
                doc.close()
                return PDFMetadata(title=os.path.basename(pdf_path))

            # 获取前两页的文本内容
            page1_text = ""
            page2_text = ""
            for page_num in range(min(2, len(doc))):
                page = doc[page_num]
                text = page.get_text()
                if page_num == 0:
                    page1_text = text[:3000]  # 截断过长的文本
                else:
                    page2_text = text[:3000]

            doc.close()

            # 获取前两页截图（base64）
            page1_image = self.pdf_processor.pdf_page_to_image(pdf_path, 0)
            page2_image = None
            try:
                doc2 = fitz.open(pdf_path)
                if len(doc2) > 1:
                    page2_image = self.pdf_processor.pdf_page_to_image(pdf_path, 1)
                doc2.close()
            except Exception:
                pass

            # 提取关键图（混合方案，先于LLM调用）
            key_image = self._extract_key_image(pdf_path)

            # 调用LLM提取文本元数据
            llm_metadata = self._llm_extract_text_metadata(
                page1_image, page2_image, page1_text, page2_text,
                has_key_image=(key_image is not None)
            )

            # 合并结果
            return PDFMetadata(
                title=llm_metadata.get('title', os.path.basename(pdf_path)),
                authors=llm_metadata.get('authors', ''),
                abstract_summary=llm_metadata.get('abstract_summary', ''),
                innovation_points=llm_metadata.get('innovation_points', []),
                key_image=key_image,
                doi="",
                arxiv_id="",
                published_date="",
                journal=""
            )

        except Exception as e:
            print(f"提取PDF元数据失败 [{pdf_path}]: {e}")
            return PDFMetadata(title=os.path.basename(pdf_path))

    def _llm_extract_text_metadata(
        self,
        page1_image: Optional[str],
        page2_image: Optional[str],
        page1_text: str,
        page2_text: str,
        has_key_image: bool = False
    ) -> dict:
        """调用视觉LLM提取文本元数据（标题、摘要、创新点）"""
        # 构建消息内容
        content_parts = []

        if page1_image:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{page1_image}", "detail": "high"}
            })
        if page2_image:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{page2_image}", "detail": "high"}
            })

        # 构建提示词
        prompt_text = self._build_extraction_prompt(page1_text, page2_text, has_key_image)
        content_parts.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content_parts}]

        result = self.llm_client.call_api(
            model=self.config.METADATA_EXTRACTION_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=None,
            timeout=self.config.METADATA_EXTRACTION_TIMEOUT
        )

        if not result:
            print("LLM元数据提取调用失败")
            return {}

        try:
            content = result['choices'][0]['message']['content']
            content = re.sub(r'```json\n|\n```|```', '', content).strip()
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"LLM返回内容JSON解析失败: {e}")
            print(f"原始内容: {content[:200]}...")
            return {}

    def _build_extraction_prompt(
        self, page1_text: str, page2_text: str, has_key_image: bool
    ) -> str:
        """构建发送给LLM的提取提示词"""
        key_image_hint = ""
        if has_key_image:
            key_image_hint = "\n注：关键图已通过算法自动识别，你不需要返回key_image字段。"

        prompt = f"""你是学术论文分析专家。请根据以下论文前两页的截图和文本内容，提取结构化元数据。

【论文第1页文本】
{page1_text[:2500]}

【论文第2页文本】
{page2_text[:2500]}

请返回以下JSON格式（务必严格遵守）：
{{{{
  "title": "论文完整标题（原文语言）",
  "authors": "作者姓名（逗号分隔）",
  "abstract_summary": "用2-3句中文总结论文摘要的核心内容",
  "innovation_points": ["创新点1", "创新点2", "创新点3"]
}}}}

要求：
1. title必须是从论文中提取的完整原始标题
2. abstract_summary使用中文总结，简洁精炼
3. innovation_points列出3个以内的核心创新点，每个用一句话描述
4. 仅返回JSON，不要添加任何其他内容{key_image_hint}"""
        return prompt
