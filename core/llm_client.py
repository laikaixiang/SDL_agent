"""
LLM API调用封装模块
统一处理与LLM API的交互，包括请求、重试、错误处理
"""

import requests
import time
import json
import re
from typing import Dict, Any, Optional, List, Generator, Callable
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
    - 支持 tool calling (OpenAI-compatible)
    - 支持 extra_body 注入 (DeepSeek thinking 等供应商特有参数)
    """

    def __init__(self, api_key: str = None, api_url: str = None, extra_body: dict = None):
        """
        初始化LLM客户端

        Args:
            api_key: API密钥，未提供则使用配置中的全局默认值
            api_url: API端点，未提供则使用配置中的全局默认值
            extra_body: 额外请求体字段，merge 到每个请求的 payload 中
        """
        self.config = Config()
        self._api_key = api_key if api_key else self.config.API_KEY
        self._api_url = api_url if api_url else self.config.API_URL
        self._extra_body = extra_body or {}
        self.headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }

    def get_api_url(self) -> str:
        """获取当前实例使用的 API URL"""
        return self._api_url

    def get_api_key(self) -> str:
        """获取当前实例使用的 API Key"""
        return self._api_key

    # ---- 通用 API 调用 ----

    def call_api(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = None,
        timeout: Optional[int] = None,
        stream: bool = False,
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
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
            tools: OpenAI-format tool definitions
            extra_body: 额外请求体字段（与实例级 extra_body 合并）

        Returns:
            API响应结果或None；流式时返回 generator
        """
        actual_max_tokens = max_tokens if max_tokens is not None else self.config.MAX_TOKENS

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        if actual_max_tokens is not None:
            payload["max_tokens"] = actual_max_tokens

        if response_format:
            payload["response_format"] = response_format

        if tools:
            payload["tools"] = tools

        # merge instance-level + call-level extra_body
        merged_extra = {}
        merged_extra.update(self._extra_body)
        if extra_body:
            merged_extra.update(extra_body)
        payload.update(merged_extra)

        if timeout is None:
            timeout = self.config.STREAM_TIMEOUT if stream else self.config.TIMEOUT

        max_retries = self.config.MAX_RETRIES
        last_error = ""

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self._api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=timeout,
                    stream=stream
                )
                response.raise_for_status()

                if stream:
                    return self._handle_stream_response(response)
                else:
                    return self._extract_response(response.json())

            except requests.exceptions.Timeout:
                last_error = f"API请求超时（{timeout}秒）"
            except requests.exceptions.HTTPError as e:
                body = ""
                try:
                    body = e.response.text[:500]
                except Exception:
                    pass
                last_error = f"HTTP {e.response.status_code}: {body}"

                # 检测纯文本模型被用于 VL 任务的常见错误（仅第一次打印提示）
                if e.response.status_code == 400 and attempt == 0:
                    _hints = [
                        ("image_url provided is not a valid image", "VL",
                         "当前 MODEL_NAME_VL 可能是一个纯文本模型，不支持图片输入，"
                         "请将 config.json 中 MODEL_NAME_VL 换为 VL 模型（如 Qwen/Qwen3-VL-30B-A3B-Instruct）"),
                        ("does not support image", "VL",
                         "当前模型不支持图片输入，请换为 VL 模型"),
                    ]
                    for keyword, model_type, hint in _hints:
                        if keyword in body:
                            print(f"[LLMClient] [警告] 检测到{model_type}模型配置错误: {hint}")
                            break
            except Exception as e:
                last_error = f"请求异常: {str(e)}"

            if attempt < max_retries - 1:
                time.sleep(2.0)

        print(f"LLM API调用失败: {last_error}")
        return None

    @staticmethod
    def _extract_response(data: dict) -> dict:
        """从非流式响应中提取 content，content 为空时 fallback 到 reasoning_content"""
        message = data.get('choices', [{}])[0].get('message', {})
        content = message.get('content', '')
        if not content:
            reasoning = message.get('reasoning_content', '')
            if reasoning:
                # 将 reasoning_content 作为 content 返回，兼容下游
                message['content'] = reasoning
        return data

    def _handle_stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        """
        处理流式响应，同时 yield content 和 reasoning_content

        Yields:
            内容片段（reasoning_content 以 [reasoning] 前缀标记）
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
                        delta = chunk_json['choices'][0].get('delta', {})
                        reasoning = delta.get('reasoning_content', '')
                        content = delta.get('content', '')
                        if reasoning:
                            yield reasoning
                        if content:
                            yield content
                    except Exception:
                        pass

    def stream_typed(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = None,
        timeout: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Generator[tuple, None, None]:
        """
        流式调用 LLM API，yield (type, text) 元组。

        Yields:
            ('reasoning', text) — 推理/思考内容
            ('content', text) — 正文内容

        相比 _handle_stream_response，此方法包含重试逻辑，
        并在 chunk 级别区分 reasoning 和 content。
        """
        actual_max_tokens = max_tokens if max_tokens is not None else self.config.MAX_TOKENS
        if timeout is None:
            timeout = self.config.STREAM_TIMEOUT

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if actual_max_tokens is not None:
            payload["max_tokens"] = actual_max_tokens

        if tools:
            payload["tools"] = tools

        # merge instance-level + call-level extra_body
        merged_extra = {}
        merged_extra.update(self._extra_body)
        if extra_body:
            merged_extra.update(extra_body)
        payload.update(merged_extra)

        max_retries = self.config.MAX_RETRIES

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self._api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=timeout,
                    stream=True,
                )
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[6:]
                            if data_str.strip() == "[DONE]":
                                return
                            try:
                                chunk_json = json.loads(data_str)
                                delta = chunk_json['choices'][0].get('delta', {})
                                reasoning = delta.get('reasoning_content', '')
                                content = delta.get('content', '')
                                if reasoning:
                                    yield ('reasoning', reasoning)
                                if content:
                                    yield ('content', content)
                            except Exception:
                                pass
                return
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise

    def stream_raw(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = None,
        timeout: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
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
        if max_tokens is None:
            max_tokens = self.config.MAX_TOKENS

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if tools:
            payload["tools"] = tools

        merged_extra = {}
        merged_extra.update(self._extra_body)
        if extra_body:
            merged_extra.update(extra_body)
        payload.update(merged_extra)

        response = requests.post(
            self._api_url,
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
        max_tokens: int = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, Any]:
        """
        带验证的API调用

        Returns:
            (成功状态, 验证后的数据或错误信息)
        """
        result = self.call_api(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            tools=tools,
            extra_body=extra_body,
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

    # ---- Tool Calling 循环 ----

    def run_with_tools(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_executor: Callable[[str, dict], str],
        extra_body: Optional[Dict[str, Any]] = None,
        max_turns: int = 10,
    ) -> dict:
        """
        执行 tool calling 循环：model → tool_calls → execute → tool_result → model → ...

        Args:
            model: 模型名称
            messages: 初始消息列表（会被原地修改，附加 assistant/tool 消息）
            tools: OpenAI-format 工具定义列表
            tool_executor: (tool_name: str, arguments: dict) -> result_str
            extra_body: 额外请求体
            max_turns: 最大循环轮次，防止无限循环

        Returns:
            {
                "final_message": assistant_message_dict,
                "tool_calls_history": [{tool_name, arguments, result}, ...]
            }

        TODO: 流式 tool calling 支持（当前仅非流式）
        TODO: 前端实时展示 tool_call 进度（SSE 事件推送）
        """
        tool_calls_history = []

        for _turn in range(max_turns):
            result = self.call_api(
                model=model,
                messages=messages,
                tools=tools,
                extra_body=extra_body,
                stream=False,
            )

            if not result:
                return {
                    "final_message": None,
                    "error": "API调用失败",
                    "tool_calls_history": tool_calls_history,
                }

            choice = result['choices'][0]
            msg = choice['message']
            finish_reason = choice.get('finish_reason', '')

            # tool_calls may be None (final answer) or a list
            tool_calls = msg.get('tool_calls')

            if tool_calls is None:
                # Final answer
                return {
                    "final_message": msg,
                    "tool_calls_history": tool_calls_history,
                }

            # Append assistant message with tool_calls
            messages.append(msg)

            # Execute each tool and collect results
            for tc in tool_calls:
                func_name = tc['function']['name']
                try:
                    arguments = json.loads(tc['function']['arguments'])
                except (json.JSONDecodeError, KeyError):
                    arguments = {}

                try:
                    exec_result = tool_executor(func_name, arguments)
                except Exception as e:
                    exec_result = f"工具执行错误: {str(e)}"

                tool_calls_history.append({
                    "tool_name": func_name,
                    "arguments": arguments,
                    "result": exec_result,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc['id'],
                    "content": str(exec_result),
                })

        # Max turns exceeded
        return {
            "final_message": None,
            "error": f"达到最大循环轮次 ({max_turns})，工具调用未收敛",
            "tool_calls_history": tool_calls_history,
        }
