"""
实验设计智能体 - 基于 PydanticAI Agent 原生 tool-use，AI 自主选择工具并规划实验流程
"""

import os
import asyncio
import queue as thread_queue
from typing import Dict

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import Config

# 从 field_inference 导入实验设计提示词
from .field_inference import ExperimentDesignAgent

# 从 hardware/tools.py 导入 PydanticAI 异步工具函数和依赖容器
from hardware.tools import (
    Deps,                    # 依赖注入容器（包含 send_event 回调）
    read_pdf,                # 读取 PDF 文件内容
    save_experiment_step,    # 注册一步旋涂实验参数
    start_experiment,        # 启动已注册的实验序列
    get_all_reagents,        # 列出所有可用试剂
)


class ExperimentDesignAgent:
    """
    实验设计智能体

    AI 通过工具函数的 docstring 理解每个工具的功能和参数，
    在对话过程中自主决定何时调用哪个工具、使用什么参数。
    支持多步实验设计：先读论文 -> 注册多步实验 -> 启动执行。

    Attributes:
        config    : 配置管理器
        _agent    : PydanticAI Agent 实例（所有会话共享）
        _sessions : 会话存储 {session_id: {"history": [...], "pdf_path": "..."}}
    """

    def __init__(self):
        self.config = Config()
        self._agent = self._create_agent()       # PydanticAI Agent（全局单例）
        self._sessions: Dict[str, dict] = {}     # 每个会话独立的对话历史
        self._response_queues: Dict[str, thread_queue.Queue] = {}  # {request_id: queue} 线程安全队列，用于等待用户确认

    def _create_agent(self) -> Agent:
        """创建 PydanticAI Agent，绑定 API 和实验工具"""
        model = OpenAIChatModel(
            self.config.EXPERIMENT_MODEL_NAME,          # 复用大语言模型
            provider=OpenAIProvider(
                base_url=self.config.API_URL.rsplit('/chat/completions', 1)[0],  # 提取 base_url
                api_key=self.config.API_KEY,
            ),
        )
        return Agent(
            model,
            system_prompt=ExperimentDesignParser.EXPERIMENT_AGENT_SYSTEM_PROMPT,
            deps_type=Deps,
            # AI 会自动分析每个函数的 docstring，理解功能后自主决定调用
            tools=[read_pdf, save_experiment_step, start_experiment, get_all_reagents],
        )

    def _get_or_create_session(self, session_id: str) -> dict:
        """获取或创建会话（含对话历史和 PDF 路径）"""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "history": [],      # PydanticAI 对话历史消息列表
                "pdf_path": None,   # 当前关联的 PDF 文件路径
            }
        return self._sessions[session_id]

    async def run(self, session_id: str, user_message: str, send_event) -> str:
        """
        运行实验设计智能体，处理用户消息并返回 AI 回复

        Args:
            session_id   : 会话 ID，隔离不同用户的对话
            user_message : 用户消息文本
            send_event   : 异步回调，用于向前端推送工具调用状态
                           签名: async def send_event(event: dict) -> None

        Returns:
            str: AI 生成的回复文本
        """
        print(f"[ExperimentAgent] 开始处理会话 {session_id}")
        print(f"[ExperimentAgent] 用户消息: {user_message[:100]}...")

        session = self._get_or_create_session(session_id)

        # 如果有关联的 PDF，在消息前附加路径提示，让 AI 知道文件位置
        if session["pdf_path"]:
            full_input = f"[Current PDF is at: {session['pdf_path']}]\n\n{user_message}"
            print(f"[ExperimentAgent] PDF路径: {session['pdf_path']}")
        else:
            full_input = user_message
            print(f"[ExperimentAgent] 无关联PDF")

        # 创建依赖容器，将 send_event 回调、agent 引用和 session_id 注入到所有工具函数中
        deps = Deps(send_event=send_event, agent=self, session_id=session_id)
        print(f"[ExperimentAgent] 依赖容器已创建")

        # 调用 PydanticAI Agent：AI 自主选择工具、决定参数、规划调用顺序
        print(f"[ExperimentAgent] 开始调用PydanticAI Agent...")
        result = await self._agent.run(
            full_input, deps=deps, message_history=session["history"],
        )
        print(f"[ExperimentAgent] Agent调用完成")

        # 更新会话的对话历史（包含本轮的工具调用记录）
        session["history"] = list(result.all_messages())
        print(f"[ExperimentAgent] 会话历史已更新，共 {len(session['history'])} 条消息")

        return result.output

    def set_pdf_path(self, session_id: str, pdf_path: str):
        """为会话关联 PDF 文件路径（上传 PDF 后调用）"""
        self._get_or_create_session(session_id)["pdf_path"] = pdf_path

    def clear_session(self, session_id: str):
        """清除指定会话的所有数据"""
        self._sessions.pop(session_id, None)

    def get_active_sessions(self) -> list:
        """获取所有活跃会话 ID"""
        return list(self._sessions.keys())

    def create_response_queue(self, request_id: str) -> thread_queue.Queue:
        """创建一个线程安全队列用于等待用户响应"""
        q = thread_queue.Queue()
        self._response_queues[request_id] = q
        return q

    async def wait_for_response(self, request_id: str, timeout: int = 300) -> dict:
        """
        等待用户响应，带超时保护（默认5分钟）

        使用 run_in_executor 将阻塞式 queue.get() 放到线程池执行，
        既不阻塞事件循环，又能安全地跨线程通信。

        Args:
            request_id: 请求ID
            timeout: 超时时间（秒）

        Returns:
            dict: 用户响应 {"action": "confirm"|"skip"|"cancel"|"timeout", "params": {...}}
        """
        # 如果队列不存在则自动创建（工具函数在 send_event 之后立即调用此方法）
        if request_id not in self._response_queues:
            self._response_queues[request_id] = thread_queue.Queue()
        q = self._response_queues[request_id]

        loop = asyncio.get_event_loop()
        try:
            # 在线程池中阻塞等待，不阻塞事件循环
            response = await loop.run_in_executor(
                None, lambda: q.get(timeout=timeout)
            )
            return response
        except thread_queue.Empty:
            return {"action": "timeout"}
        finally:
            self._response_queues.pop(request_id, None)

    def submit_response(self, request_id: str, response: dict):
        """
        提交用户响应到等待队列（由 Flask 请求线程同步调用）

        使用线程安全的 queue.Queue.put()，可安全地从任意线程调用。

        Args:
            request_id: 请求ID
            response: 用户响应数据
        """
        # 防御性创建：极端情况下 submit 可能先于 wait_for_response 执行
        if request_id not in self._response_queues:
            self._response_queues[request_id] = thread_queue.Queue()
        self._response_queues[request_id].put(response)
