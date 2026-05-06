"""
PDF处理模块
负责PDF文件的读取、页面转换、图像处理等功能
"""

import fitz
import os
import base64
import io
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

from core.config import Config
from utils.pdf_to_markdown import pdf_page_to_markdown, detect_complex_content


class PDFProcessor:
    """
    PDF处理器类 - 处理PDF文件的各项操作

    职责：
    - PDF文件读取和验证
    - 页面转换为图片
    - 获取PDF元数据和页面信息
    - 处理PDF相关的错误
    """

    def __init__(self):
        """初始化PDF处理器"""
        self.config = Config()

    def pdf_page_to_image(self, pdf_path: str, page_num: int) -> Optional[str]:
        """
        将PDF页面转换为Base64编码的图片

        Args:
            pdf_path: PDF文件路径
            page_num: 页码（从0开始）

        Returns:
            Base64编码的图片字符串或None
        """
        try:
            doc = fitz.open(pdf_path)
            if page_num >= len(doc) or page_num < 0:
                return None

            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(self.config.DPI / 72, self.config.DPI / 72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            doc.close()
            return img_str

        except Exception as e:
            print(f"PDF页面转换失败: {e}")
            return None

    def get_pdf_info(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """
        获取PDF文件的基本信息

        Args:
            pdf_path: PDF文件路径

        Returns:
            PDF信息字典或None
        """
        try:
            with fitz.open(pdf_path) as doc:
                info = {
                    'filename': os.path.basename(pdf_path),
                    'path': pdf_path,
                    'num_pages': len(doc),
                    'metadata': doc.metadata,
                    'title': doc.metadata.get('title', ''),
                    'author': doc.metadata.get('author', ''),
                    'subject': doc.metadata.get('subject', '')
                }
                return info
        except Exception as e:
            print(f"获取PDF信息失败: {e}")
            return None

    def list_pdf_files(self, folder_path: Optional[str] = None) -> List[str]:
        """
        列出指定文件夹中的所有PDF文件

        Args:
            folder_path: 文件夹路径，如果为None则使用配置的PDF文件夹

        Returns:
            PDF文件路径列表
        """
        if folder_path is None:
            folder_path = self.config.PDF_FOLDER

        if not os.path.exists(folder_path):
            return []

        pdf_files = []
        for f in os.listdir(folder_path):
            if f.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(folder_path, f))

        return pdf_files

    def validate_pdf_file(self, pdf_path: str) -> Tuple[bool, str]:
        """
        验证PDF文件是否有效

        Args:
            pdf_path: PDF文件路径

        Returns:
            (是否有效, 错误信息)
        """
        try:
            if not os.path.exists(pdf_path):
                return False, "文件不存在"

            if not pdf_path.lower().endswith('.pdf'):
                return False, "文件不是PDF格式"

            with fitz.open(pdf_path) as doc:
                if len(doc) == 0:
                    return False, "PDF文件没有页面"

            return True, ""

        except fitz.FileDataError:
            return False, "PDF文件损坏或无法读取"
        except Exception as e:
            return False, f"验证失败: {str(e)}"

    def extract_text_from_page(self, pdf_path: str, page_num: int) -> Optional[str]:
        """
        从PDF页面提取文本

        Args:
            pdf_path: PDF文件路径
            page_num: 页码（从0开始）

        Returns:
            提取的文本或None
        """
        try:
            with fitz.open(pdf_path) as doc:
                if page_num >= len(doc) or page_num < 0:
                    return None

                page = doc.load_page(page_num)
                text = page.get_text()
                return text

        except Exception as e:
            print(f"提取文本失败: {e}")
            return None

    def get_all_pages_text(self, pdf_path: str) -> Optional[List[str]]:
        """
        获取PDF所有页面的文本

        Args:
            pdf_path: PDF文件路径

        Returns:
            页面文本列表或None
        """
        try:
            with fitz.open(pdf_path) as doc:
                pages_text = []
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    pages_text.append(text)
                return pages_text

        except Exception as e:
            print(f"获取所有页面文本失败: {e}")
            return None

    def convert_to_images(self, pdf_path: str, start_page: int = 0, end_page: Optional[int] = None) -> List[str]:
        """
        将PDF页面批量转换为图片

        Args:
            pdf_path: PDF文件路径
            start_page: 起始页码（从0开始）
            end_page: 结束页码（包含），如果为None则转换到最后一页

        Returns:
            Base64编码的图片字符串列表
        """
        images = []

        try:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)

                if end_page is None:
                    end_page = total_pages - 1

                # 确保页码范围有效
                start_page = max(0, start_page)
                end_page = min(total_pages - 1, end_page)

                for page_num in range(start_page, end_page + 1):
                    img_str = self.pdf_page_to_image(pdf_path, page_num)
                    if img_str:
                        images.append(img_str)

        except Exception as e:
            print(f"批量转换图片失败: {e}")

        return images

    def extract_page_content(self, pdf_path: str, page_num: int, mode: str = "hybrid") -> Tuple[Optional[str], Optional[str], bool]:
        """
        提取PDF页面内容（支持多种模式）

        Args:
            pdf_path: PDF文件路径
            page_num: 页码（从0开始）
            mode: 提取模式 - "vision"(纯图片), "text"(纯文本), "hybrid"(混合)

        Returns:
            (markdown_text, image_base64, use_vision) 元组
            - markdown_text: Markdown格式的文本（如果提取了文本）
            - image_base64: Base64编码的图片（如果需要视觉分析）
            - use_vision: 是否建议使用Vision API
        """
        if mode == "vision":
            # 纯视觉模式：只返回图片
            img_base64 = self.pdf_page_to_image(pdf_path, page_num)
            return None, img_base64, True

        elif mode == "text":
            # 纯文本模式：只返回文本
            markdown_text = pdf_page_to_markdown(pdf_path, page_num)
            return markdown_text, None, False

        elif mode == "hybrid":
            # 混合模式：先提取文本，判断是否需要视觉分析
            markdown_text = pdf_page_to_markdown(pdf_path, page_num)

            # 检测是否包含复杂内容
            needs_vision = detect_complex_content(markdown_text)

            if needs_vision:
                # 需要视觉分析，同时返回文本和图片
                img_base64 = self.pdf_page_to_image(pdf_path, page_num)
                return markdown_text, img_base64, True
            else:
                # 纯文本即可
                return markdown_text, None, False

        else:
            raise ValueError(f"不支持的提取模式: {mode}")

    def get_extraction_mode(self) -> str:
        """
        获取当前配置的提取模式

        Returns:
            提取模式字符串
        """
        return self.config.EXTRACTION_MODE