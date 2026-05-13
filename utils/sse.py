"""
Flask SSE 响应封装。

将 StreamEvent 字典生成器包装为 Flask Response，
MIME 类型为 text/event-stream。
"""

import json
from typing import Generator

from flask import Response


def sse_response(event_stream: Generator[dict, None, None]) -> Response:
    """将 StreamEvent 生成器包装为 Flask SSE Response"""

    def generate():
        for event in event_stream:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
            "Connection": "keep-alive",
        },
    )
