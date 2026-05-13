"""
LLM API调用封装模块
统一处理与LLM API的交互，包括请求、重试、错误处理
"""

import requests
import time
import json
import re
from typing import Dict, Any, Optional, List, Generator
from pydantic import BaseModel, ValidationError

from .config import Config


class LLMClient:
    """
    LLM客户端类 - 统一处理LLM API调用

    职责：
    - 统一处理API请求和响应
    - 处理重试逻辑
    - 错误处理和日志记录
    - 支持流式和非流式请求
    """

    def __init__(self):
        """初始化LLM客户端"""
        self.config = Config()
        self.headers = {
            "Authorization": f"Bearer {self.config.API_KEY}",
            "Content-Type": "application/json"
        }

    def call_api(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: Optional[int] = None,
        stream: bool = False,
        response_format: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        调用LLM API

        Args:
            model: 模型名称
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            timeout: 超时时间
            stream: 是否使用流式响应
            response_format: 响应格式

        Returns:
            API响应结果或None
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }

        if response_format:
            payload["response_format"] = response_format

        if timeout is None:
            timeout = self.config.STREAM_TIMEOUT if stream else self.config.TIMEOUT

        max_retries = self.config.MAX_RETRIES
        last_error = ""

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.config.API_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=timeout,
                    stream=stream
                )
                response.raise_for_status()

                if stream:
                    return self._handle_stream_response(response)
                else:
                    return response.json()

            except requests.exceptions.Timeout:
                last_error = f"API请求超时（{timeout}秒）"
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP错误: {e.response.status_code}"
            except Exception as e:
                last_error = f"请求异常: {str(e)}"

            if attempt < max_retries - 1:
                time.sleep(2.0)

        print(f"LLM API调用失败: {last_error}")
        return None

    def _handle_stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        """
        处理流式响应

        Args:
            response: 响应对象

        Yields:
            内容片段
        """
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
                            yield content
                    except Exception:
                        pass

    def stream_raw(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        从 LLM API 流式获取原始 SSE 行。

        与 _handle_stream_response 不同，此方法 yield 完整的解码 SSE 行
        （如 "data: {...}"），以便 StreamAdapter 可以同时处理
        reasoning_content 和 content。

        Yields:
            解码后的 SSE 行字符串
        """
        if timeout is None:
            timeout = self.config.STREAM_TIMEOUT

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        response = requests.post(
            self.config.API_URL,
            headers=self.headers,
            json=payload,
            stream=True,
            timeout=timeout,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                yield line.decode("utf-8")

    def call_api_with_validation(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        response_model: BaseModel,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ) -> tuple[bool, Any]:
        """
        带验证的API调用

        Args:
            model: 模型名称
            messages: 消息列表
            response_model: 响应验证模型
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            (成功状态, 验证后的数据或错误信息)
        """
        result = self.call_api(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

        if not result:
            return False, "API调用失败"

        try:
            content = result['choices'][0]['message']['content'].strip()
            clean_text = re.sub(r'```json\n|\n```|```', '', content).strip()
            validated_data = response_model.model_validate_json(clean_text)
            return True, validated_data
        except ValidationError as ve:
            return False, f"验证失败: {ve}"
        except Exception as e:
            return False, f"解析失败: {str(e)}"

    def get_default_headers(self) -> Dict[str, str]:
        """
        获取默认请求头

        Returns:
            默认请求头字典
        """
        return self.headers.copy()