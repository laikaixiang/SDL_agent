"""
LLM 原始 SSE 行 → 类型化 StreamEvent 序列。

处理 thinking ↔ text 状态机：
  - reasoning_content 先到达 → THINKING_START → THINKING_DELTA* → THINKING_END
  - content 随后到达 → TEXT_START → TEXT_DELTA* → TEXT_END
  - 无推理能力的模型直接进入 TEXT_START
"""

import json
from typing import Generator

# --- 流事件类型常量 ---
TEXT_START = "text_start"
TEXT_DELTA = "text_delta"
TEXT_END = "text_end"
THINKING_START = "thinking_start"
THINKING_DELTA = "thinking_delta"
THINKING_END = "thinking_end"
ERROR = "error"
DONE = "done"


def make_event(event_type: str, text: str = "", **extra) -> dict:
    """构造一个 StreamEvent 字典，text 为空时省略该键"""
    event: dict = {"type": event_type}
    if text:
        event["text"] = text
    event.update(extra)
    return event


class StreamAdapter:
    """将 LLM API 的原始 SSE 行转换为类型化 StreamEvent 序列"""

    def __init__(self):
        self._thinking_buf = ""
        self._text_buf = ""
        self._thinking_started = False
        self._text_started = False

    def adapt(self, raw_lines: Generator[str, None, None]) -> Generator[dict, None, None]:
        """
        消费原始 SSE 行，产出 StreamEvent 字典。

        Args:
            raw_lines: 解码后的 SSE 行生成器（如 "data: {...}"）

        Yields:
            StreamEvent 字典，包含 type 键，可选 text 键

        TODO: 检测 delta 中的 tool_calls → 产出 tool_call_start / tool_call_delta / tool_call_end 事件
        """
        try:
            for line in raw_lines:
                delta = self._parse_line(line)
                if delta is None:
                    continue

                reasoning = delta.get("reasoning_content", "")
                content = delta.get("content", "")

                # TODO: tool_calls 事件
                # tool_calls = delta.get("tool_calls")
                # if tool_calls:
                #     yield from self._handle_tool_calls(tool_calls)

                if reasoning:
                    yield from self._handle_reasoning(reasoning)

                if content:
                    yield from self._handle_text(content)

            # 流结束 — flush 未完结的 buffer
            yield from self._flush()
            yield make_event(DONE)

        except GeneratorExit:
            yield from self._flush()
            yield make_event(DONE)
        except Exception as e:
            yield from self._flush()
            yield make_event(ERROR, str(e))

    def _parse_line(self, line: str) -> dict | None:
        """解析单行 SSE 数据，返回 delta 字典；非数据行返回 None"""
        if not line.startswith("data: "):
            return None
        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            return None
        try:
            chunk = json.loads(data_str)
            return chunk.get("choices", [{}])[0].get("delta", {})
        except (json.JSONDecodeError, KeyError, IndexError):
            return None

    def _handle_reasoning(self, reasoning: str) -> Generator[dict, None, None]:
        """处理 reasoning_content 块"""
        if not self._thinking_started:
            # 如果正文已在流式输出（少见但可能），先结束正文
            if self._text_started:
                yield make_event(TEXT_END, self._text_buf)
                self._text_started = False
                self._text_buf = ""

            self._thinking_started = True
            self._thinking_buf = ""
            yield make_event(THINKING_START)

        self._thinking_buf += reasoning
        yield make_event(THINKING_DELTA, self._thinking_buf)

    def _handle_text(self, content: str) -> Generator[dict, None, None]:
        """处理正文内容块"""
        # 如果思考正在流式输出，先结束思考
        if self._thinking_started:
            yield make_event(THINKING_END, self._thinking_buf)
            self._thinking_started = False
            self._thinking_buf = ""

        if not self._text_started:
            self._text_started = True
            self._text_buf = ""
            yield make_event(TEXT_START)

        self._text_buf += content
        yield make_event(TEXT_DELTA, self._text_buf)

    def _flush(self) -> Generator[dict, None, None]:
        """flush 所有未完结的 thinking 或 text buffer"""
        if self._thinking_started:
            yield make_event(THINKING_END, self._thinking_buf)
            self._thinking_started = False
            self._thinking_buf = ""
        if self._text_started:
            yield make_event(TEXT_END, self._text_buf)
            self._text_started = False
            self._text_buf = ""
