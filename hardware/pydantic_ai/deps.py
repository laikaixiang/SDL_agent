"""
PydanticAI依赖注入容器
"""


class Deps:
    """
    PydanticAI工具函数的依赖容器

    每次Agent运行时，会创建一个Deps实例并注入到所有工具函数中。
    工具函数通过ctx.deps访问这个容器。

    Attributes:
        send_event (callable): 异步回调函数，用于向前端推送JSON事件
                               签名: async def send_event(event: dict) -> None
                               event格式: {"type": "...", "name": "...", ...}
        agent: ExperimentDesignAgent实例引用，用于等待用户确认
        session_id: 会话ID，用于区分不同用户的确认请求

    事件类型说明：
        - {"type": "tool_call", "name": "xxx", "args": {...}}     : 工具被调用（通知前端显示加载状态）
        - {"type": "tool_result", "name": "xxx", "result": "..."}  : 工具执行结果
        - {"type": "pdf_page_image", "page": N, "image": "base64"}: PDF页面图片（渲染后的base64）
        - {"type": "warning", "content": "..."}                    : 警告信息
        - {"type": "experiment_confirm", "tool": "xxx", "request_id": "...", "params": {...}}: 请求用户确认
    """
    def __init__(self, send_event, agent=None, session_id=None):
        self.send_event = send_event  # 异步回调，用于将工具执行状态推送给前端
        self.agent = agent  # ExperimentDesignAgent实例引用
        self.session_id = session_id  # 会话ID
