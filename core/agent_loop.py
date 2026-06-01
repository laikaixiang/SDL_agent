"""
Agent 循环引擎 (core/agent_loop.py)
====================================

驱动 LLM tool-use 循环的核心模块。

组件:
    AgentTurn dataclass       — 单次工具调用的记录
    AgentLoop                 — 流式 tool-use 循环引擎
    SubAgent                  — 从 YAML 模板加载的子 agent
    AgentOrchestrator         — 子 agent 生命周期管理器
"""

import json
import os
import queue
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass

import yaml

from core.config import Config
from core.llm_client import LLMClient
from core.agent_tools import UnifiedToolExecutor, create_main_executor
from utils.stream_adapter import StreamAdapter


# =============================================================================
# AgentTurn 数据类
# =============================================================================


@dataclass
class AgentTurn:
    """单次工具调用的记录"""
    tool_name: str
    arguments: dict
    result: str
    status: str  # "success" | "error"


# =============================================================================
# AgentLoop — 流式 tool-use 循环引擎
# =============================================================================


class AgentLoop:
    """
    流式 tool-use 循环引擎。

    使用 StreamAdapter 处理 LLM 流式输出，检测 tool_calls，
    执行工具并将结果反馈给 LLM，直到获得纯文本响应或达到最大轮次。

    使用示例::

        from core.agent_loop import AgentLoop
        from core.llm_client import LLMClient
        from core.agent_tools import create_main_executor

        llm = LLMClient()
        executor = create_main_executor()
        loop = AgentLoop(llm, executor, model="deepseek-chat")

        messages = [{"role": "user", "content": "帮我查一下今天的天气"}]
        result = loop.run(messages)
        print(result["final_message"])
        for turn in result["tool_turns"]:
            print(f"  -> {turn.tool_name}: {turn.result}")
    """

    def __init__(
        self,
        llm: LLMClient,
        executor: UnifiedToolExecutor,
        model: str,
        max_turns: int = 15,
        extra_body: dict = None,
    ):
        """
        Args:
            llm: LLMClient 实例
            executor: UnifiedToolExecutor 实例
            model: 模型名称字符串
            max_turns: 最大循环迭代次数
            extra_body: 可选的额外请求体 dict
        """
        self.llm = llm
        self.executor = executor
        self.model = model
        self.max_turns = max_turns
        self.extra_body = extra_body

    def run(
        self,
        messages: list[dict],
        event_callback=None,
        ask_user_queue=None,
        timeout: float | None = None,
    ) -> dict:
        """
        执行 agent tool-use 循环。

        Args:
            messages: OpenAI-format 消息列表（会被原地修改）
            event_callback: 可选回调 callable(event_dict)，用于 SSE 事件转发
            ask_user_queue: 可选 queue.Queue，用于 ask_user 阻塞等待用户输入
            timeout: 可选整体超时秒数，None 表示无限制；超时后返回 error dict

        Returns:
            {
                "final_message": dict | None,   # None 表示未收敛（达到轮次上限）
                "tool_turns": [AgentTurn, ...],
                "error": str | None,
            }
        """
        tool_turns: list[AgentTurn] = []

        def _run_loop() -> dict:
            """线程安全的主体循环封装"""
            for turn in range(self.max_turns):
                # ---- 1. 流式调用 LLM ----
                try:
                    raw_lines = self.llm.stream_raw(
                        model=self.model,
                        messages=messages,
                        tools=self.executor.build_openai_tools(),
                        extra_body=self.extra_body,
                    )
                except Exception as e:
                    return {
                        "final_message": None,
                        "tool_turns": tool_turns,
                        "error": f"LLM 流式调用失败: {str(e)}",
                    }

                # ---- 2. StreamAdapter 处理流式输出 ----
                adapter = StreamAdapter()
                accumulated_text = ""

                for event in adapter.adapt(raw_lines):
                    # 捕获正文内容，用于构建 final_message
                    if event.get("type") in ("text_delta", "text_end"):
                        accumulated_text = event.get("text", accumulated_text)

                    # 转发事件到回调
                    if event_callback:
                        try:
                            event_callback(event)
                        except Exception:
                            pass  # 回调异常不中断主循环

                # ---- 3. 检查是否有 tool_calls ----
                pending = adapter.get_pending_tool_calls()
                active_slots = [s for s in pending if s.get("started")]

                if not active_slots:
                    # 纯文本响应 — 循环结束
                    content = accumulated_text if accumulated_text else None
                    final_message = {"role": "assistant", "content": content}
                    return {
                        "final_message": final_message,
                        "tool_turns": tool_turns,
                        "error": None,
                    }

                # ---- 4. 构建 parsed_tool_calls（TOOL_CALL_END 已由 StreamAdapter._flush 在迭代中发送）----
                parsed_tool_calls: list[dict] = []
                for slot in active_slots:
                    try:
                        arguments = json.loads(slot["args_buf"])
                    except (json.JSONDecodeError, TypeError):
                        arguments = {"_raw": slot["args_buf"]}

                    parsed_tool_calls.append({
                        "id": slot["id"],
                        "name": slot["name"],
                        "index": slot["index"],
                        "arguments": arguments,
                    })

                # ---- 5. 构建 assistant 消息（含 tool_calls 数组） ----
                tool_calls_payload = []
                for tc in parsed_tool_calls:
                    tool_calls_payload.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                        },
                    })

                assistant_msg = {
                    "role": "assistant",
                    "content": accumulated_text if accumulated_text else None,
                    "tool_calls": tool_calls_payload,
                }
                messages.append(assistant_msg)

                # ---- 6. 逐个执行工具 ----
                for i, tc in enumerate(parsed_tool_calls):
                    tool_name = tc["name"]
                    arguments = tc["arguments"]

                    # ---- ask_user 特殊处理 ----
                    if tool_name == "ask_user" and ask_user_queue is not None:
                        question = arguments.get("question", "")
                        options = arguments.get("options", "")

                        # 发送 agent_question 事件
                        if event_callback:
                            try:
                                event_callback({
                                    "type": "agent_question",
                                    "question": question,
                                    "options": options,
                                })
                            except Exception:
                                pass

                        # 阻塞等待用户回答（5 分钟超时）
                        try:
                            result = ask_user_queue.get(timeout=300)
                        except queue.Empty:
                            result = "用户未在超时时间内响应（5分钟）"
                        status = "success"
                    else:
                        # ---- 常规工具分发 ----
                        result = self.executor.dispatch(tool_name, arguments)
                        status = "error" if "错误" in result or "未找到" in result else "success"

                    # ---- 发送 tool_result 事件 ----
                    if event_callback:
                        try:
                            event_callback({
                                "type": "tool_result",
                                "index": i,
                                "name": tool_name,
                                "result": result,
                                "status": status,
                            })
                        except Exception:
                            pass

                    # ---- 追加 tool 结果消息 ----
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": str(result),
                    })

                    # ---- 记录 AgentTurn ----
                    tool_turns.append(AgentTurn(
                        tool_name=tool_name,
                        arguments=arguments,
                        result=str(result),
                        status=status,
                    ))

            # ---- 7. 达到最大轮次 ----
            return {
                "final_message": None,
                "tool_turns": tool_turns,
                "error": f"达到最大循环轮次 ({self.max_turns})，工具调用未收敛",
            }

        # ---- 超时控制 ----
        if timeout is not None:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run_loop)
                try:
                    return future.result(timeout=timeout)
                except TimeoutError:
                    return {
                        "final_message": None,
                        "tool_turns": tool_turns,
                        "error": f"Agent loop timed out after {timeout} seconds",
                    }
        return _run_loop()


# =============================================================================
# SubAgent — 从 YAML 模板创建子 agent
# =============================================================================


class SubAgent:
    """
    从 YAML 模板文件创建的子 agent。

    YAML 模板格式::

        name: string                    # agent 名称
        system_prompt: string           # 系统提示词
        tools: [string, ...]           # 工具名称列表
        max_turns: int                  # 最大循环轮次（可选，默认 10）

    使用示例::

        agent = SubAgent("prompts/zh/agents/data_analyst.yaml")
        result = agent.run("分析文件 data.csv 中的趋势")
    """

    def __init__(self, template_path: str, executor: UnifiedToolExecutor = None):
        """
        Args:
            template_path: YAML 模板文件绝对路径
            executor: 可选 UnifiedToolExecutor；如未提供则创建空 executor
        """
        self.name = ""
        self.system_prompt = ""
        self.tool_names: list[str] = []
        self.max_turns = 10

        # 加载 YAML 模板
        self._load_template(template_path)

        # 创建 LLMClient 实例（使用 TALK 专用配置）
        config = Config()
        self.llm = LLMClient(
            api_key=config.TALK_API_KEY,
            api_url=config.TALK_API_URL,
            extra_body=config.get_extra_body("talk"),
        )

        # 构建工具执行器
        if executor is not None:
            self.executor = self._build_subset_executor(executor)
        else:
            self.executor = UnifiedToolExecutor([])

    def _load_template(self, template_path: str):
        """加载并解析 YAML 模板"""
        if not os.path.isfile(template_path):
            raise FileNotFoundError(f"Agent 模板文件不存在: {template_path}")

        with open(template_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Agent 模板必须是 YAML dict，实际: {type(data).__name__}")

        self.name = data.get("name", "")
        self.system_prompt = data.get("system_prompt", "")
        self.tool_names = data.get("tools", [])
        self.max_turns = data.get("max_turns", 10)

        if not self.name:
            raise ValueError("Agent 模板缺少必需的 'name' 字段")
        if not self.system_prompt:
            raise ValueError("Agent 模板缺少必需的 'system_prompt' 字段")

    def _build_subset_executor(self, executor: UnifiedToolExecutor) -> UnifiedToolExecutor:
        """
        从已有 executor 中筛选模板指定的工具，构建子集 executor。

        Args:
            executor: 完整的 UnifiedToolExecutor

        Returns:
            仅包含模板指定工具的 UnifiedToolExecutor
        """
        from core.agent_tools import AgentTool

        subset: list[AgentTool] = []
        for name in self.tool_names:
            tool = executor.get(name)
            if tool is not None:
                subset.append(tool)
            else:
                # 工具名不在主 executor 中 — 静默跳过，由 AgentLoop 的 dispatch 返回错误
                pass

        return UnifiedToolExecutor(subset)

    def run(self, task: str, context: dict = None) -> dict:
        """
        执行子 agent 任务。

        Args:
            task: 用户任务描述
            context: 可选的上下文 dict（作为额外 system 消息注入）

        Returns:
            AgentLoop.run() 的返回结果 dict
        """
        messages: list[dict] = []

        # system prompt
        messages.append({"role": "system", "content": self.system_prompt})

        # context data
        if context:
            context_str = json.dumps(context, ensure_ascii=False, indent=2)
            messages.append({
                "role": "system",
                "content": f"上下文数据:\n{context_str}",
            })

        # user task
        messages.append({"role": "user", "content": task})

        # 创建 AgentLoop 并运行
        loop = AgentLoop(
            llm=self.llm,
            executor=self.executor,
            model=Config().MODEL_NAME_TALK,
            max_turns=self.max_turns,
        )

        return loop.run(messages)


# =============================================================================
# AgentOrchestrator — 子 agent 生命周期管理器
# =============================================================================


class AgentOrchestrator:
    """
    管理子 agent 的创建与执行。

    使用示例::

        orch = AgentOrchestrator()
        print("可用模板:", orch.list_templates())

        # 顺序执行
        result = orch.spawn("data_analyst", "分析 data.csv")

        # 并行执行
        results = orch.spawn_parallel(
            "data_analyst",
            ["分析 file1.csv", "分析 file2.csv", "分析 file3.csv"],
        )
    """

    def __init__(self, templates_dir: str = "prompts/zh/agents", executor: UnifiedToolExecutor = None):
        """
        Args:
            templates_dir: 模板目录，相对于项目根目录的路径
            executor: 可选 UnifiedToolExecutor，传递给子 agent 用于构建子集工具
        """
        # 计算项目根目录（core/ 的上两级）
        _core_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(_core_dir)
        self.templates_dir = os.path.join(_project_root, templates_dir)
        self._executor = executor

    def list_templates(self) -> list[str]:
        """
        列出所有可用的 agent 模板名称。

        Returns:
            模板文件名列表（不含 .yaml 扩展名）
        """
        if not os.path.isdir(self.templates_dir):
            return []

        templates = []
        for fname in os.listdir(self.templates_dir):
            if fname.endswith(".yaml") or fname.endswith(".yml"):
                name = fname.rsplit(".", 1)[0]  # 去掉扩展名
                templates.append(name)
        return sorted(templates)

    def spawn(self, template: str, task: str, context: dict = None) -> dict:
        """
        创建并运行一个子 agent。

        Args:
            template: 模板名称（不含 .yaml 扩展名）
            task: 用户任务描述
            context: 可选的上下文 dict

        Returns:
            SubAgent.run() 的返回结果 dict，错误时返回 {"error": str}
        """
        template_path = os.path.join(self.templates_dir, f"{template}.yaml")

        # 尝试 .yaml -> .yml 回退
        if not os.path.isfile(template_path):
            yml_path = os.path.join(self.templates_dir, f"{template}.yml")
            if os.path.isfile(yml_path):
                template_path = yml_path
            else:
                return {"error": f"Agent 模板不存在: {template}.yaml (已检查 .yaml 和 .yml)"}

        try:
            sub = SubAgent(template_path, executor=self._executor)
            return sub.run(task, context)
        except Exception as e:
            return {"error": f"子 agent 启动失败: {str(e)}"}

    def spawn_parallel(self, template: str, tasks: list[str]) -> list[dict]:
        """
        并行运行多个子 agent（每个子 agent 执行同一个模板的不同任务）。

        Args:
            template: 模板名称（不含 .yaml 扩展名）
            tasks: 任务描述列表

        Returns:
            结果 dict 列表，与 tasks 顺序一一对应
        """
        template_path = os.path.join(self.templates_dir, f"{template}.yaml")
        if not os.path.isfile(template_path):
            yml_path = os.path.join(self.templates_dir, f"{template}.yml")
            if os.path.isfile(yml_path):
                template_path = yml_path
            else:
                return [{"error": f"Agent 模板不存在: {template}.yaml"}] * len(tasks)

        max_workers = min(len(tasks), 4)

        def _run_one(task: str) -> dict:
            try:
                sub = SubAgent(template_path, executor=self._executor)
                return sub.run(task)
            except Exception as e:
                return {"error": f"子 agent 启动失败: {str(e)}"}

        # 维护结果顺序
        results: list[dict] = [{}] * len(tasks)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(_run_one, task): idx
                for idx, task in enumerate(tasks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = {"error": str(e)}

        return results

    def spawn_pipeline(self, steps: list[dict]) -> list[dict]:
        """
        Execute agent steps sequentially, each output feeding the next as context.

        Args:
            steps: [{"template": "literature_searcher", "task": "search for X"},
                    {"template": "summarizer", "task": "summarize"}]

        Returns:
            List of result dicts, one per step
        """
        results = []
        accumulated_context = {}

        for i, step in enumerate(steps):
            template = step.get("template", "")
            task = step.get("task", "")

            if not template or not task:
                results.append({"error": f"Step {i}: missing template or task"})
                continue

            # Feed previous results as context
            context = accumulated_context.copy() if accumulated_context else None

            result = self.spawn(template, task, context)
            results.append(result)

            # Extract summary from result for next step
            if result and not result.get("error"):
                fm = result.get("final_message", {})
                content = fm.get("content", "") if fm else str(result)
                accumulated_context[f"step_{i}_{template}"] = str(content)[:2000]

        return results
