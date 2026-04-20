"""
实验设计智能体 - 基于 Approach 2（JSON + 提示词）的交互式实验设计

完全使用 core/field_inference.py 的 ExperimentDesignAgent（Approach 2）
不依赖 PydanticAI，支持任何 LLM
"""

import asyncio
import queue as thread_queue
from typing import Dict

from .config import Config
from .field_inference import ExperimentDesignAgent as FieldInferenceAgent


class ExperimentDesignAgent:
    """
    实验设计智能体 - 基于 Approach 2 的交互式版本

    使用 core/field_inference.py 的 ExperimentDesignAgent 生成实验设计
    支持多轮对话和会话管理

    Attributes:
        config    : 配置管理器
        _agent    : FieldInference ExperimentDesignAgent 实例
        _sessions : 会话存储 {session_id: {"history": [...], "pdf_path": "..."}}
    """

    def __init__(self):
        self.config = Config()
        self._agent = FieldInferenceAgent()  # 使用 Approach 2 的实验设计代理
        self._sessions: Dict[str, dict] = {}  # 每个会话独立的对话历史
        self._response_queues: Dict[str, thread_queue.Queue] = {}  # {request_id: queue} 线程安全队列，用于等待用户确认

    def _get_or_create_session(self, session_id: str) -> dict:
        """获取或创建会话（含对话历史和 PDF 路径）"""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "history": [],      # 对话历史消息列表
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
            str: AI 生成的回复文本（实验设计 JSON 或错误信息）
        """
        print(f"[ExperimentAgent] 开始处理会话 {session_id}")
        print(f"[ExperimentAgent] 用户消息: {user_message[:100]}...")

        session = self._get_or_create_session(session_id)

        # 如果有关联的 PDF，在消息前附加路径提示
        if session["pdf_path"]:
            full_input = f"[Current PDF is at: {session['pdf_path']}]\n\n{user_message}"
            print(f"[ExperimentAgent] PDF路径: {session['pdf_path']}")
        else:
            full_input = user_message
            print(f"[ExperimentAgent] 无关联PDF")

        # 使用 Approach 2 生成实验设计
        print(f"[ExperimentAgent] 开始调用 Approach 2 Agent...")
        success, result = self._agent.parse_experiment_design(full_input)
        print(f"[ExperimentAgent] Agent调用完成")

        if success:
            # 成功生成实验设计
            import json
            experiment_json = json.dumps(result, ensure_ascii=False, indent=2)

            # 更新会话历史
            session["history"].append({
                "role": "user",
                "content": user_message
            })
            session["history"].append({
                "role": "assistant",
                "content": f"已生成实验设计：{result.get('experiment_name', '未命名实验')}"
            })

            print(f"[ExperimentAgent] 会话历史已更新，共 {len(session['history'])} 条消息")

            # 推送成功事件到前端
            await send_event({
                "type": "experiment_design_generated",
                "experiment_json": result,
                "experiment_name": result.get('experiment_name', '未命名实验'),
                "steps_count": len(result.get('steps', []))
            })

            return f"✅ 已生成实验设计方案：{result.get('experiment_name', '未命名实验')}\n\n共 {len(result.get('steps', []))} 个步骤。"
        else:
            # 生成失败
            error_message = f"❌ 实验设计生成失败：{result}"

            # 更新会话历史
            session["history"].append({
                "role": "user",
                "content": user_message
            })
            session["history"].append({
                "role": "assistant",
                "content": error_message
            })

            print(f"[ExperimentAgent] 生成失败: {result}")

            # 推送失败事件到前端
            await send_event({
                "type": "error",
                "message": error_message
            })

            return error_message

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
        # 如果队列不存在则自动创建
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
