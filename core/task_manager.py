"""
任务管理模块
负责任务队列管理、任务状态跟踪、任务执行控制
"""

import queue
import threading
import json
import os
import time
from typing import Dict, Any, Optional, List
from enum import Enum

from .config import Config


class TaskType(Enum):
    """任务类型枚举"""
    INFO = "info"
    PROGRESS = "progress"
    PAGE_READING = "page_reading"
    READING_START = "reading_start"
    READING_CHUNK = "reading_chunk"
    FINDING = "finding"
    ERROR = "error"
    COMPLETE = "complete"
    TASK_TRIGGER = "task_trigger"
    FIELD_CONFIRM = "field_confirm"
    SYSTEM = "system"


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TaskInfo:
    """任务信息类"""

    def __init__(self, task_id: str, task_type: str, data: Any = None):
        self.task_id = task_id
        self.task_type = task_type
        self.data = data
        self.timestamp = time.time()


class TaskManager:
    """
    任务管理器类 - 负责任务队列管理和状态跟踪

    职责：
    - 管理任务队列
    - 跟踪任务状态
    - 支持任务取消
    - 提供任务进度信息
    """

    def __init__(self):
        """初始化任务管理器"""
        self.config = Config()
        self.task_queue = queue.Queue()
        self.task_running = False
        self.cancel_requested = False
        self.current_task_id: Optional[str] = None
        self.task_history: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def start_task(self, task_id: str) -> None:
        """
        开始新任务

        Args:
            task_id: 任务ID
        """
        with self.lock:
            self.current_task_id = task_id
            self.task_running = True
            self.cancel_requested = False
            self.task_history.append({
                "task_id": task_id,
                "status": TaskStatus.RUNNING.value,
                "start_time": time.time()
            })

    def cancel_task(self) -> None:
        """取消当前任务"""
        with self.lock:
            self.cancel_requested = True

    def is_task_cancelled(self) -> bool:
        """
        检查任务是否已取消

        Returns:
            是否已取消
        """
        with self.lock:
            return self.cancel_requested

    def complete_task(self, task_id: str, result: Dict[str, Any]) -> None:
        """
        完成任务

        Args:
            task_id: 任务ID
            result: 任务结果
        """
        with self.lock:
            self.task_running = False
            self.current_task_id = None

            # 更新任务历史
            for task in self.task_history:
                if task["task_id"] == task_id:
                    task["status"] = TaskStatus.COMPLETED.value
                    task["end_time"] = time.time()
                    task["result"] = result
                    break

    def fail_task(self, task_id: str, error: str) -> None:
        """
        标记任务失败

        Args:
            task_id: 任务ID
            error: 错误信息
        """
        with self.lock:
            self.task_running = False
            self.current_task_id = None

            # 更新任务历史
            for task in self.task_history:
                if task["task_id"] == task_id:
                    task["status"] = TaskStatus.FAILED.value
                    task["end_time"] = time.time()
                    task["error"] = error
                    break

    def put_task_message(self, msg_type: str, data: Any = None) -> None:
        """
        向任务队列添加消息

        Args:
            msg_type: 消息类型
            data: 消息数据
        """
        message = {
            "type": msg_type,
            "data": data,
            "timestamp": time.time()
        }
        self.task_queue.put(message)

    def get_task_message(self, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
        """
        从任务队列获取消息

        Args:
            timeout: 超时时间

        Returns:
            消息或None
        """
        try:
            return self.task_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_task_status(self) -> Dict[str, Any]:
        """
        获取任务状态

        Returns:
            任务状态信息
        """
        with self.lock:
            status = {
                "task_running": self.task_running,
                "current_task_id": self.current_task_id,
                "queue_size": self.task_queue.qsize(),
                "cancel_requested": self.cancel_requested
            }

            if self.current_task_id:
                status["current_task"] = self.get_task_by_id(self.current_task_id)

            return status

    def get_task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取任务信息

        Args:
            task_id: 任务ID

        Returns:
            任务信息或None
        """
        for task in self.task_history:
            if task["task_id"] == task_id:
                return task
        return None

    def get_recent_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的任务

        Args:
            limit: 限制数量

        Returns:
            任务列表
        """
        return self.task_history[-limit:]

    def clear_completed_tasks(self) -> None:
        """清除已完成的任务"""
        with self.lock:
            self.task_history = [
                task for task in self.task_history
                if task.get("status") != TaskStatus.COMPLETED.value
            ]

    def generate_task_id(self) -> str:
        """
        生成任务ID

        Returns:
            任务ID
        """
        return f"task_{int(time.time() * 1000)}_{os.urandom(4).hex()}"

    def is_queue_empty(self) -> bool:
        """
        检查队列是否为空

        Returns:
            是否为空
        """
        return self.task_queue.empty()

    def clear_queue(self) -> None:
        """清空队列"""
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break