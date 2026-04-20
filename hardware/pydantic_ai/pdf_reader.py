"""
PDF读取工具 - PydanticAI异步工具
"""

import os
import base64
import uuid
from typing import Optional

import PyPDF2
import fitz  # PyMuPDF - 用于将PDF页面渲染为图片
from pydantic_ai import RunContext

from .deps import Deps


async def read_pdf(
    ctx: RunContext[Deps],
    file_path: str,
    page_number: Optional[int] = None,
) -> str:
    """
    从PDF文件中提取文本内容（需用户确认页码范围）

    如果指定了page_number，除了提取文本外还会将该页渲染为图片，
    通过WebSocket事件发送给前端展示在页面右侧。

    Args:
        ctx         : PydanticAI运行上下文，包含deps（依赖注入容器）
        file_path   : PDF文件的完整路径
        page_number : 要读取的页码（从1开始），None表示读取全部页面

    Returns:
        str: 提取到的文本内容。如果指定了不存在的页码，返回错误提示。

    AI使用说明：
        - 上传PDF后系统会提供文件路径
        - 可以先不指定page_number读取全文概览
        - 再指定具体页码深入阅读某一页
    """
    # 生成唯一请求ID
    request_id = str(uuid.uuid4())

    # 推送确认请求到前端
    await ctx.deps.send_event({
        "type": "experiment_confirm",
        "tool": "read_pdf",
        "request_id": request_id,
        "session_id": ctx.deps.session_id,
        "params": {
            "file_path": file_path,
            "page_number": page_number,
        }
    })

    # 等待用户响应
    if ctx.deps.agent:
        response = await ctx.deps.agent.wait_for_response(request_id)

        if response["action"] == "skip":
            return "用户跳过读取PDF"
        elif response["action"] == "cancel":
            return "用户取消读取PDF"
        elif response["action"] == "timeout":
            return "等待用户确认超时"
        elif response["action"] == "confirm":
            # 使用修改后的参数（如果有）
            params = response.get("params", {})
            page_number = params.get("page_number", page_number)

    # 通知前端：read_pdf工具被调用
    await ctx.deps.send_event({
        "type": "tool_call",
        "name": "read_pdf",
        "args": {"file_path": file_path, "page_number": page_number},
    })

    # 检查文件是否存在
    if not os.path.exists(file_path):
        err = f"File not found: {file_path}"
        await ctx.deps.send_event({"type": "tool_result", "name": "read_pdf", "result": err})
        return err

    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)  # PDF总页数

            if page_number is not None:
                # ---------- 读取指定页面 ----------
                if 1 <= page_number <= num_pages:
                    page = reader.pages[page_number - 1]  # PyPDF2使用0索引
                    text = page.extract_text() or ""

                    # 尝试将该页渲染为图片（需要PyMuPDF）
                    try:
                        doc = fitz.open(file_path)
                        page_img = doc[page_number - 1]
                        pix = page_img.get_pixmap()        # 渲染为像素图
                        img_data = pix.tobytes("png")      # 转为PNG二进制
                        img_base64 = base64.b64encode(img_data).decode()
                        # 通过WebSocket将图片发送给前端
                        await ctx.deps.send_event({
                            "type": "pdf_page_image",
                            "page": page_number,
                            "image": img_base64,
                        })
                        doc.close()
                    except Exception as img_err:
                        # 图片渲染失败不影响文本提取
                        await ctx.deps.send_event({
                            "type": "warning",
                            "content": (
                                f"Could not render page {page_number} as image: {img_err}. "
                                "Please install PyMuPDF with 'pip install PyMuPDF'."
                            ),
                        })
                else:
                    text = f"Page {page_number} out of range (1–{num_pages})."
            else:
                # ---------- 读取全部页面 ----------
                text = ""
                for i, page in enumerate(reader.pages):
                    text += f"\n--- Page {i + 1} ---\n"
                    text += page.extract_text() or ""

        # 通知前端：read_pdf工具执行完成
        await ctx.deps.send_event({
            "type": "tool_result",
            "name": "read_pdf",
            "result": f"reading text: {text[:20]}…",
        })
        return text

    except Exception as e:
        err = f"Error reading PDF: {str(e)}"
        return err
