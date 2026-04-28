"""
自适应流式响应处理器 (core/adaptive_stream.py)
================================================

功能：
1. 自动检测API是否支持流式响应
2. 支持流式时使用流式响应（更好的用户体验）
3. 不支持时自动降级到非流式响应
4. 缓存检测结果，避免重复检测

使用示例::

    from core.adaptive_stream import AdaptiveStreamHandler

    handler = AdaptiveStreamHandler(config, llm_client)

    # 方式1: 直接生成响应
    response = handler.generate_response(user_message)

    # 方式2: 检查是否支持流式
    if handler.supports_streaming():
        # 使用流式
    else:
        # 使用非流式
"""

import requests
import json
import time
from typing import Generator, Optional, Dict, Any
from flask import Response


class AdaptiveStreamHandler:
    """自适应流式响应处理器"""

    def __init__(self, config, llm_client):
        """
        初始化处理器

        Args:
            config: 配置对象
            llm_client: LLM客户端对象
        """
        self.config = config
        self.llm_client = llm_client
        self._streaming_support = None  # None=未检测, True=支持, False=不支持
        self._last_check_time = 0
        self._check_interval = 3600  # 1小时重新检测一次

    def supports_streaming(self) -> bool:
        """
        检测API是否支持流式响应

        Returns:
            是否支持流式响应
        """
        # 如果已经检测过且未过期，直接返回缓存结果
        current_time = time.time()
        if self._streaming_support is not None and \
           (current_time - self._last_check_time) < self._check_interval:
            return self._streaming_support

        # 执行检测
        print("[自适应流式] 正在检测API流式响应支持...")

        try:
            headers = self.llm_client.get_default_headers()
            payload = {
                "model": self.config.MODEL_NAME_TALK,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
                "max_tokens": 5
            }

            response = requests.post(
                self.config.API_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=10
            )

            if response.status_code != 200:
                print(f"[自适应流式] API返回错误状态码: {response.status_code}")
                self._streaming_support = False
                self._last_check_time = current_time
                return False

            # 尝试读取流式响应
            chunks_received = 0
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        chunks_received += 1
                        if chunks_received >= 1:  # 至少收到1个数据块就算支持
                            break

            if chunks_received > 0:
                print(f"[自适应流式] ✓ API支持流式响应 (接收到 {chunks_received} 个数据块)")
                self._streaming_support = True
            else:
                print("[自适应流式] × API不支持流式响应，将使用非流式模式")
                self._streaming_support = False

            self._last_check_time = current_time
            return self._streaming_support

        except Exception as e:
            print(f"[自适应流式] 检测失败: {str(e)}, 默认使用非流式模式")
            self._streaming_support = False
            self._last_check_time = current_time
            return False

    def _build_messages(self, user_message: str, history: list = None) -> list:
        """将前端对话历史（role: user/ai）转换为 LLM API 格式（role: user/assistant），拼接当前消息。"""
        messages = []
        if history:
            for m in history:
                role = m.get("role", "user")
                content = m.get("content", "")
                if not content:
                    continue
                api_role = "assistant" if role == "ai" else "user"
                messages.append({"role": api_role, "content": content})
        messages.append({"role": "user", "content": user_message})
        return messages

    def generate_streaming_response(self, user_message: str, model: Optional[str] = None, history: list = None) -> Generator[str, None, None]:
        """
        生成流式响应

        Args:
            user_message: 用户消息
            model: 模型名称（可选，默认使用配置中的对话模型）
            history: 前端对话历史 [{role, content, ...}]

        Yields:
            响应内容片段
        """
        headers = self.llm_client.get_default_headers()
        payload = {
            "model": model or self.config.MODEL_NAME_TALK,
            "messages": self._build_messages(user_message, history),
            "stream": True
        }

        try:
            response = requests.post(
                self.config.API_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=self.config.STREAM_TIMEOUT
            )

            if response.status_code != 200:
                yield f"\n[API错误: 状态码 {response.status_code}]"
                return

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            content = chunk['choices'][0]['delta'].get('content', '')
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass

        except Exception as e:
            yield f"\n[请求失败: {str(e)}]"

    def generate_non_streaming_response(self, user_message: str, model: Optional[str] = None, history: list = None) -> str:
        """
        生成非流式响应

        Args:
            user_message: 用户消息
            model: 模型名称（可选，默认使用配置中的对话模型）
            history: 前端对话历史 [{role, content, ...}]

        Returns:
            完整的响应内容
        """
        headers = self.llm_client.get_default_headers()
        payload = {
            "model": model or self.config.MODEL_NAME_TALK,
            "messages": self._build_messages(user_message, history),
            "stream": False
        }

        try:
            response = requests.post(
                self.config.API_URL,
                headers=headers,
                json=payload,
                timeout=self.config.TIMEOUT
            )

            if response.status_code != 200:
                return f"[API错误: 状态码 {response.status_code}]"

            data = response.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')

            if not content:
                return "[API返回空响应]"

            return content

        except Exception as e:
            return f"[请求失败: {str(e)}]"

    def generate_response(self, user_message: str, model: Optional[str] = None, force_mode: Optional[str] = None, history: list = None) -> Response:
        """
        自适应生成响应（自动选择流式或非流式）

        Args:
            user_message: 用户消息
            model: 模型名称（可选）
            force_mode: 强制模式 ("streaming" 或 "non-streaming"，可选)
            history: 前端对话历史 [{role, content, ...}]

        Returns:
            Flask Response对象
        """
        # 如果强制指定模式
        if force_mode == "streaming":
            return Response(
                self.generate_streaming_response(user_message, model, history=history),
                content_type='text/plain; charset=utf-8'
            )
        elif force_mode == "non-streaming":
            content = self.generate_non_streaming_response(user_message, model, history=history)
            return Response(content, content_type='text/plain; charset=utf-8')

        # 自适应模式：根据检测结果选择
        if self.supports_streaming():
            # 使用流式响应
            return Response(
                self.generate_streaming_response(user_message, model, history=history),
                content_type='text/plain; charset=utf-8'
            )
        else:
            # 使用非流式响应，但模拟流式输出（逐字输出）
            def simulate_streaming():
                content = self.generate_non_streaming_response(user_message, model, history=history)
                # 逐字输出，模拟流式效果
                for char in content:
                    yield char
                    time.sleep(0.01)  # 每个字符延迟10ms

            return Response(
                simulate_streaming(),
                content_type='text/plain; charset=utf-8'
            )

    def force_recheck(self):
        """强制重新检测流式支持"""
        self._streaming_support = None
        self._last_check_time = 0
        return self.supports_streaming()

    def get_status(self) -> Dict[str, Any]:
        """
        获取当前状态

        Returns:
            状态信息字典
        """
        return {
            "streaming_support": self._streaming_support,
            "last_check_time": self._last_check_time,
            "check_interval": self._check_interval,
            "time_until_recheck": max(0, self._check_interval - (time.time() - self._last_check_time))
        }
