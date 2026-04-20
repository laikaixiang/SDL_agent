"""
(后续要删)
硬件工具模块 (hardware/tools.py)
================================

本模块包含两类函数：

1. **PydanticAI 异步工具函数**（供 ExperimentDesignAgent 的 AI 自主调用）：
   - read_pdf()              : 读取 PDF 文件内容，支持按页渲染图片
   - save_experiment_step()  : 注册一步旋涂实验参数到平台
   - start_experiment()      : 启动已注册的多步实验序列
   - get_all_reagents()      : 扫描试剂配置文件，列出所有可用试剂
   - do_experiment()         : 单次执行旋涂实验（旧接口，保留兼容）

2. **同步底层函数**（供 core/hardware_controller.py 的 LLM 命令解析路径调用）：
   - execute_spin_coating()     : 向平台发送旋涂实验 MQTT 指令
   - execute_set_temperature()  : 设置加热台温度
   - execute_move_robot_arm()   : 移动机械臂到指定坐标
   - execute_start_experiment() : 发送实验序列启动指令
   - execute_collect_spectrum() : 启动光谱仪数据采集

辅助函数：
   - find_reagent()    : 根据试剂名称查找其在平台上的物理位置（BPxx）
   - get_mqtt_client() : 获取全局 MQTT 客户端实例（懒加载模式）

MQTT 消息协议说明：
   - 主题 "do_experiment" 用于向 C# 平台发送实验指令
   - 注册步骤消息格式: "p{转速},{加速度},{时长},{试剂位置},{体积}"
     例如: "p3000,1000,30000,BP01,10"
   - 启动实验消息: "pstart"

使用示例（PydanticAI Agent 模式）::

    from pydantic_ai import Agent
    from hardware.tools import read_pdf, save_experiment_step, start_experiment, get_all_reagents, Deps

    agent = Agent(model, tools=[read_pdf, save_experiment_step, start_experiment, get_all_reagents], deps_type=Deps)
    # AI 会根据用户指令自行决定调用哪些工具

使用示例（同步底层调用）::

    from hardware.tools import execute_spin_coating, execute_start_experiment

    # 注册一步实验
    result = execute_spin_coating(3000, 1000, 30000, "Perovskite", 10)
    # 启动实验
    result = execute_start_experiment()
"""

import json
from typing import Optional
import os
import base64
import logging
import uuid

import PyPDF2
import fitz  # PyMuPDF - 用于将 PDF 页面渲染为图片
from pydantic_ai import RunContext

from .agent_client import MQTTConnector

# ---------- 日志配置 ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- 试剂配置文件路径 ----------
# 默认路径，指向项目根目录下的 reagent_layout.json
# 也可以通过 core/config.py 的 REAGENT_LAYOUT_PATH 配置覆盖
json_path = "reagent_layout.json"

# ---------- MQTT 主题 ----------
# 用于向 C# 自动化平台发送实验指令的 MQTT 主题名称
topic = "do_experiment"


# =============================================================================
# MQTT 客户端管理（懒加载模式）
# =============================================================================

# _local_client: 全局 MQTT 客户端实例（模块级单例）
# 使用 None 初始化，第一次调用 get_mqtt_client() 时才会创建并连接
# 这样避免了模块导入时就尝试连接 MQTT（如果服务器不可达会导致启动卡顿）
_local_client = None


def get_mqtt_client() -> MQTTConnector:
    """
    获取全局 MQTT 客户端实例（懒加载 + 自动重连）

    首次调用时会创建 MQTTConnector 实例并尝试连接。
    后续调用时如果连接已断开，会自动尝试重连。

    Returns:
        MQTTConnector: 已连接（或已尝试连接）的 MQTT 客户端实例

    使用示例::

        client = get_mqtt_client()
        if client.is_connected:
            client.publish("do_experiment", "pstart")
    """
    global _local_client
    if _local_client is None:
        # 首次调用：创建新的 MQTT 连接器实例
        _local_client = MQTTConnector()
    if not _local_client.is_connected:
        # 如果当前未连接，尝试重新连接（超时 2 秒）
        _local_client.connect(timeout=2)
    return _local_client


# 兼容旧代码：保留 local_client 变量名供 core/hardware_controller.py 引用
# 注意：这里不再在模块导入时就连接，而是通过属性访问时懒加载
class _LazyClient:
    """
    懒加载代理类，使 `local_client.is_connected` 等属性访问时
    自动触发 get_mqtt_client()，避免模块导入时就连接 MQTT
    """
    @property
    def is_connected(self) -> bool:
        """获取当前 MQTT 连接状态"""
        return get_mqtt_client().is_connected

    def connect(self, timeout=5) -> bool:
        """连接 MQTT 服务器"""
        return get_mqtt_client().connect(timeout)

    def check_connect(self) -> bool:
        """检查 MQTT 连接是否正常"""
        return get_mqtt_client().check_connect()

    def publish(self, topic: str, msg: str):
        """发布 MQTT 消息"""
        get_mqtt_client().publish(topic, msg)


# local_client: 供外部模块（如 core/hardware_controller.py）直接引用的 MQTT 客户端
# 通过 _LazyClient 代理实现懒加载
local_client = _LazyClient()


# =============================================================================
# 依赖注入容器
# =============================================================================

class Deps:
    """
    PydanticAI 工具函数的依赖容器

    每次 Agent 运行时，会创建一个 Deps 实例并注入到所有工具函数中。
    工具函数通过 ctx.deps 访问这个容器。

    Attributes:
        send_event (callable): 异步回调函数，用于向前端推送 JSON 事件
                               签名: async def send_event(event: dict) -> None
                               event 格式: {"type": "...", "name": "...", ...}
        agent: ExperimentDesignAgent 实例引用，用于等待用户确认
        session_id: 会话ID，用于区分不同用户的确认请求

    事件类型说明：
        - {"type": "tool_call", "name": "xxx", "args": {...}}     : 工具被调用（通知前端显示加载状态）
        - {"type": "tool_result", "name": "xxx", "result": "..."}  : 工具执行结果
        - {"type": "pdf_page_image", "page": N, "image": "base64"}: PDF 页面图片（渲染后的base64）
        - {"type": "warning", "content": "..."}                    : 警告信息
        - {"type": "experiment_confirm", "tool": "xxx", "request_id": "...", "params": {...}}: 请求用户确认
    """
    def __init__(self, send_event, agent=None, session_id=None):
        self.send_event = send_event  # 异步回调，用于将工具执行状态推送给前端
        self.agent = agent  # ExperimentDesignAgent 实例引用
        self.session_id = session_id  # 会话ID


# =============================================================================
# 辅助函数
# =============================================================================

def find_reagent(name: str, path: str = json_path) -> str:
    """
    根据试剂名称在 reagent_layout.json 中查找其物理位置

    reagent_layout.json 文件结构示例::

        {
            "Points": {
                "BP01": {"name": "Perovskite", "x": 100, "y": 200},
                "BP02": {"name": "DMF", "x": 150, "y": 250},
                ...
            }
        }

    Args:
        name : 要查找的试剂名称（必须完全匹配，区分大小写）
        path : reagent_layout.json 文件路径，默认使用模块级 json_path 变量

    Returns:
        str: 如果找到，返回位置编号字符串（如 "BP01"）
             如果未找到，返回 "Reagent is missing"
             如果读取文件出错，返回错误描述字符串
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # "Points" 字典中每个 key 是位置编号（如 "BP01"），value 包含试剂信息
            points = data.get("Points", {})
            for point_id, info in points.items():
                reagent_name = info.get("name", "")
                if reagent_name == name:
                    return point_id  # 找到匹配的试剂，返回位置编号
            return "Reagent is missing"  # 遍历完毕未找到
    except Exception as e:
        err = str(e)
        return err


# 兼容旧代码：get_reagent 是 find_reagent 的别名
get_reagent = find_reagent


# =============================================================================
# PydanticAI 异步工具函数
# =============================================================================
# 以下函数通过 PydanticAI 的 tools=[] 参数注册给 Agent。
# AI 模型会根据函数的 docstring 和参数签名来理解工具的功能，
# 并在对话过程中自主决定何时、以什么参数调用这些工具。

async def read_pdf(
    ctx: RunContext[Deps],
    file_path: str,
    page_number: Optional[int] = None,
) -> str:
    """
    从 PDF 文件中提取文本内容（需用户确认页码范围）

    如果指定了 page_number，除了提取文本外还会将该页渲染为图片，
    通过 WebSocket 事件发送给前端展示在页面右侧。

    Args:
        ctx         : PydanticAI 运行上下文，包含 deps（依赖注入容器）
        file_path   : PDF 文件的完整路径
        page_number : 要读取的页码（从 1 开始），None 表示读取全部页面

    Returns:
        str: 提取到的文本内容。如果指定了不存在的页码，返回错误提示。

    AI 使用说明：
        - 上传 PDF 后系统会提供文件路径
        - 可以先不指定 page_number 读取全文概览
        - 再指定具体页码深入阅读某一页
    """
    # 生成唯一请求ID
    request_id = str(uuid.uuid4())

    # 推送确认请求到前端
    await ctx.deps.send_event({
        "type": "experiment_confirm",
        "tool": "read_pdf",
        "request_id": request_id,
        "session_id": ctx.deps.session_id,
        "params": {
            "file_path": file_path,
            "page_number": page_number,
        }
    })

    # 等待用户响应
    if ctx.deps.agent:
        response = await ctx.deps.agent.wait_for_response(request_id)

        if response["action"] == "skip":
            return "用户跳过读取PDF"
        elif response["action"] == "cancel":
            return "用户取消读取PDF"
        elif response["action"] == "timeout":
            return "等待用户确认超时"
        elif response["action"] == "confirm":
            # 使用修改后的参数（如果有）
            params = response.get("params", {})
            page_number = params.get("page_number", page_number)

    # 通知前端：read_pdf 工具被调用
    await ctx.deps.send_event({
        "type": "tool_call",
        "name": "read_pdf",
        "args": {"file_path": file_path, "page_number": page_number},
    })

    # 检查文件是否存在
    if not os.path.exists(file_path):
        err = f"File not found: {file_path}"
        await ctx.deps.send_event({"type": "tool_result", "name": "read_pdf", "result": err})
        return err

    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)  # PDF 总页数

            if page_number is not None:
                # ---------- 读取指定页面 ----------
                if 1 <= page_number <= num_pages:
                    page = reader.pages[page_number - 1]  # PyPDF2 使用 0 索引
                    text = page.extract_text() or ""

                    # 尝试将该页渲染为图片（需要 PyMuPDF）
                    try:
                        doc = fitz.open(file_path)
                        page_img = doc[page_number - 1]
                        pix = page_img.get_pixmap()        # 渲染为像素图
                        img_data = pix.tobytes("png")      # 转为 PNG 二进制
                        img_base64 = base64.b64encode(img_data).decode()
                        # 通过 WebSocket 将图片发送给前端
                        await ctx.deps.send_event({
                            "type": "pdf_page_image",
                            "page": page_number,
                            "image": img_base64,
                        })
                        doc.close()
                    except Exception as img_err:
                        # 图片渲染失败不影响文本提取
                        await ctx.deps.send_event({
                            "type": "warning",
                            "content": (
                                f"Could not render page {page_number} as image: {img_err}. "
                                "Please install PyMuPDF with 'pip install PyMuPDF'."
                            ),
                        })
                else:
                    text = f"Page {page_number} out of range (1–{num_pages})."
            else:
                # ---------- 读取全部页面 ----------
                text = ""
                for i, page in enumerate(reader.pages):
                    text += f"\n--- Page {i + 1} ---\n"
                    text += page.extract_text() or ""

        # 通知前端：read_pdf 工具执行完成
        await ctx.deps.send_event({
            "type": "tool_result",
            "name": "read_pdf",
            "result": f"reading text: {text[:20]}…",
        })
        return text

    except Exception as e:
        err = f"Error reading PDF: {str(e)}"
        return err


async def get_all_reagents(
    ctx: RunContext[Deps],
    path: str = json_path,
) -> str:
    """
    扫描 reagent_layout.json，列出平台上所有已配置的试剂名称

    当 AI 调用 save_experiment_step 时收到 "Reagent is missing" 错误，
    可以调用此工具检查是否是试剂名拼写错误，或确认试剂是否已装载到平台上。

    Args:
        ctx  : PydanticAI 运行上下文
        path : reagent_layout.json 文件路径

    Returns:
        str: 逗号分隔的试剂名称列表，如 "Perovskite, DMF, DMSO, "
             如果发生错误，返回错误描述字符串

    AI 使用说明：
        - 当收到 "Reagent is missing" 时，调用此工具查看有哪些试剂可用
        - 检查试剂名是否拼写正确
    """
    try:
        # 通知前端：get_all_reagents 工具被调用
        await ctx.deps.send_event({
            "type": "tool_call",
            "name": "get_all_reagents",
            "args": {},
        })

        with open(path, "r", encoding="utf-8") as f:
            available_reagents = ""   # 汇总字符串
            idx = 0                   # 已找到的试剂计数
            data = json.load(f)
            points = data.get("Points", {})
            for point_id, info in points.items():
                if info.get("name") != "":
                    # 将非空试剂名追加到结果字符串
                    available_reagents += f"{info.get('name')}, "
                    idx += 1

            msg = f"扫描完成，找到 {idx} 种可用试剂"
            await ctx.deps.send_event({
                "type": "tool_result",
                "name": "get_all_reagents",
                "result": msg,
            })
            return available_reagents

    except Exception as e:
        err = str(e)
        return err


async def save_experiment_step(
    ctx: RunContext[Deps],
    spin_speed: int = 3000,
    spin_acc: int = 1000,
    spin_dur: int = 30000,
    reagent: str = "",
    volume: int = 10,
) -> str:
    """
    注册一步旋涂实验参数到自动化平台（需用户确认）

    此函数将实验参数格式化为 MQTT 消息发送给 C# 平台保存。
    一轮完整实验可能包含多个步骤（如先涂底层、再涂活性层），
    每个步骤需调用一次 save_experiment_step()，
    所有步骤注册完毕后再调用 start_experiment() 启动执行。

    MQTT 消息格式: "p{转速},{加速度},{时长},{试剂位置},{体积}"
    示例: "p3000,1000,30000,BP01,10"

    Args:
        ctx        : PydanticAI 运行上下文
        spin_speed : 旋涂转速，单位 rpm，最大 6000rpm，默认 3000rpm
        spin_acc   : 旋涂加速度，单位 rpm/s，默认 1000rpm/s
        spin_dur   : 旋涂持续时间，单位毫秒(ms)，默认 30000ms（即 30 秒）
        reagent    : 使用的试剂名称（必须与 reagent_layout.json 中的名称一致）
        volume     : 试剂滴加体积，单位微升(µl)，默认 10µl

    Returns:
        str: 成功时返回包含所有参数的确认消息（带 ✅ 前缀）
             试剂未找到时返回 "Reagent is missing"
             MQTT 连接失败时返回 "Connect server failed"

    AI 使用说明：
        - 先读取论文获取实验参数（转速、时间、试剂等）
        - 对每一步实验调用一次此函数
        - 所有步骤注册完毕后，调用 start_experiment() 启动
        - 如果收到 "Reagent is missing"，可调用 get_all_reagents() 检查拼写
    """
    try:
        # 生成唯一请求ID
        request_id = str(uuid.uuid4())

        # 推送确认请求到前端
        await ctx.deps.send_event({
            "type": "experiment_confirm",
            "tool": "save_experiment_step",
            "request_id": request_id,
            "session_id": ctx.deps.session_id,
            "params": {
                "spin_speed": spin_speed,
                "spin_acc": spin_acc,
                "spin_dur": spin_dur,
                "reagent": reagent,
                "volume": volume,
            }
        })

        # 等待用户响应
        if ctx.deps.agent:
            response = await ctx.deps.agent.wait_for_response(request_id)

            if response["action"] == "skip":
                return "用户跳过此步骤"
            elif response["action"] == "cancel":
                return "用户取消操作"
            elif response["action"] == "timeout":
                return "等待用户确认超时"
            elif response["action"] == "confirm":
                # 使用修改后的参数（如果有）
                params = response.get("params", {})
                spin_speed = params.get("spin_speed", spin_speed)
                spin_acc = params.get("spin_acc", spin_acc)
                spin_dur = params.get("spin_dur", spin_dur)
                reagent = params.get("reagent", reagent)
                volume = params.get("volume", volume)

        # 通知前端：save_experiment_step 工具被调用，附带参数详情
        await ctx.deps.send_event({
            "type": "tool_call",
            "name": "save_experiment_step",
            "args": {
                "spin_speed": spin_speed,
                "spin_acc": spin_acc,
                "spin_dur": spin_dur,
                "reagent": reagent,
                "volume": volume,
            },
        })

        # 在 reagent_layout.json 中查找试剂的物理位置（如 "BP01"）
        reagent_pos = find_reagent(reagent)
        if reagent_pos[:2] != "BP":
            # 未找到试剂（返回值不以 "BP" 开头，说明是错误消息）
            return reagent_pos

        # 尝试通过 MQTT 发送实验参数
        client = get_mqtt_client()
        if client.is_connected:
            # 格式化并发送 MQTT 消息
            client.publish(topic, f"p{spin_speed},{spin_acc},{spin_dur},{reagent_pos},{volume}")
            msg = (
                f"✅ 实验步骤已注册: 试剂 {reagent} (位置 {reagent_pos}), "
                f"转速 {spin_speed} rpm, 加速度 {spin_acc} rpm/s, "
                f"持续 {spin_dur} ms, 体积 {volume} µl"
            )
            await ctx.deps.send_event({
                "type": "tool_result",
                "name": "save_experiment_step",
                "result": msg,
            })
            return msg
        else:
            # 当前未连接，尝试重新连接
            connect_state = client.connect()
            if connect_state:
                client.publish(topic, f"p{spin_speed},{spin_acc},{spin_dur},{reagent_pos},{volume}")
                msg = (
                    f"✅ 实验步骤已注册: 试剂 {reagent} (位置 {reagent_pos}), "
                    f"转速 {spin_speed} rpm, 加速度 {spin_acc} rpm/s, "
                    f"持续 {spin_dur} ms, 体积 {volume} µl"
                )
                await ctx.deps.send_event({
                    "type": "tool_result",
                    "name": "save_experiment_step",
                    "result": msg,
                })
                return msg
            else:
                return "Connect server failed"
    except Exception as e:
        err = f"Error occurred: {str(e)}"
        return err


async def start_experiment(
    ctx: RunContext[Deps],
) -> bool:
    """
    启动已注册的多步实验序列（需用户确认）

    向 C# 自动化平台发送 "pstart" 命令，平台会按照之前通过
    save_experiment_step() 注册的步骤顺序，依次执行所有实验操作。

    调用前提：
        必须先通过 save_experiment_step() 注册至少一个实验步骤。
        如果没有注册任何步骤就调用此函数，平台不会执行任何操作。

    Args:
        ctx : PydanticAI 运行上下文

    Returns:
        bool: True 表示启动指令发送成功，False 表示 MQTT 连接失败

    AI 使用说明：
        - 确保所有实验步骤都已通过 save_experiment_step() 注册
        - 检查步骤数量是否与论文描述一致
        - 确认后再调用此函数启动实验
    """
    try:
        # 生成唯一请求ID
        request_id = str(uuid.uuid4())

        # 推送确认请求到前端
        await ctx.deps.send_event({
            "type": "experiment_confirm",
            "tool": "start_experiment",
            "request_id": request_id,
            "session_id": ctx.deps.session_id,
            "params": {}
        })

        # 等待用户响应
        if ctx.deps.agent:
            response = await ctx.deps.agent.wait_for_response(request_id)

            if response["action"] == "skip":
                return False
            elif response["action"] == "cancel":
                return False
            elif response["action"] == "timeout":
                return False

        # 通知前端：start_experiment 工具被调用
        await ctx.deps.send_event({
            "type": "tool_call",
            "name": "start_experiment",
            "args": {},
        })

        client = get_mqtt_client()
        if client.is_connected:
            # 发送启动命令 "pstart"
            client.publish(topic, "pstart")
            msg = "✅ 实验序列已启动"
            await ctx.deps.send_event({
                "type": "tool_result",
                "name": "start_experiment",
                "result": msg,
            })
            return True
        else:
            # 尝试重连
            connect_state = client.connect()
            if connect_state:
                client.publish(topic, "pstart")
                msg = "✅ 实验序列已启动"
                await ctx.deps.send_event({
                    "type": "tool_result",
                    "name": "start_experiment",
                    "result": msg,
                })
                return True
            else:
                return False

    except Exception:
        return False


async def do_experiment(
    ctx: RunContext[Deps],
    spin_speed: int = 3000,
    spin_acc: int = 1000,
    spin_dur: int = 30000,
    reagent: str = "",
    volume: int = 10,
) -> str:
    """
    执行单次旋涂实验（旧接口，保留向后兼容）

    与 save_experiment_step + start_experiment 的区别：
    - do_experiment 是单次执行，适合只有一步的简单实验
    - save_experiment_step + start_experiment 支持多步实验序列

    新代码建议使用 save_experiment_step + start_experiment 组合。

    Args:
        ctx        : PydanticAI 运行上下文
        spin_speed : 旋涂转速 (rpm)，最大 6000，默认 3000
        spin_acc   : 加速度 (rpm/s)，默认 1000
        spin_dur   : 持续时间 (ms)，默认 30000
        reagent    : 试剂名称
        volume     : 体积 (µl)，默认 10

    Returns:
        str: 成功时返回确认消息，失败时返回错误描述
    """
    try:
        await ctx.deps.send_event({
            "type": "tool_call",
            "name": "do_experiment",
            "args": {
                "spin_speed": spin_speed,
                "spin_acc": spin_acc,
                "spin_dur": spin_dur,
                "reagent": reagent,
                "volume": volume,
            },
        })

        # 查找试剂位置
        reagent_pos = find_reagent(reagent)
        if reagent_pos[:2] != "BP":
            return reagent_pos

        client = get_mqtt_client()
        if client.is_connected:
            client.publish(topic, f"p{spin_speed},{spin_acc},{spin_dur},{reagent_pos},{volume}")
            msg = (
                f"✅ Experiment started: seeking {reagent} at {reagent_pos}, "
                f"{spin_speed} rpm, acc {spin_acc} rpm/s, "
                f"duration {spin_dur} ms, volume {volume} µl."
            )
            await ctx.deps.send_event({"type": "tool_result", "name": "do_experiment", "result": msg})
            return msg
        else:
            connect_state = client.connect()
            if connect_state:
                client.publish(topic, f"p{spin_speed},{spin_acc},{spin_dur},{reagent_pos},{volume}")
                msg = (
                    f"✅ Experiment started: seeking {reagent} at {reagent_pos}, "
                    f"{spin_speed} rpm, acc {spin_acc} rpm/s, "
                    f"duration {spin_dur} ms, volume {volume} µl."
                )
                await ctx.deps.send_event({"type": "tool_result", "name": "do_experiment", "result": msg})
                return msg
            else:
                return "Connect server failed"
    except Exception as e:
        err = f"Error occurred: {str(e)}"
        return err


# =============================================================================
# 同步底层函数
# =============================================================================
# 以下函数由 core/hardware_controller.py 的 HardwareAgent.execute_tool_call() 调用。
# 它们是同步的（非 async），不依赖 PydanticAI 的 RunContext。
# 每个函数对应一种硬件操作。

def execute_spin_coating(
    spin_speed: int,
    spin_acc: int,
    spin_dur: int,
    reagent: str,
    volume: int,
) -> str:
    """
    底层同步函数：向自动化平台发送旋涂实验 MQTT 指令

    此函数将实验参数打包为 JSON 格式发送到 "do_experiment" MQTT 主题。
    与异步版本 save_experiment_step 不同，此函数直接发送 JSON payload
    而非逗号分隔的字符串格式。

    Args:
        spin_speed : 旋涂转速 (rpm)
        spin_acc   : 加速度 (rpm/s)
        spin_dur   : 持续时间 (ms)
        reagent    : 试剂名称
        volume     : 体积 (µl)

    Returns:
        str: 成功时返回 "实验指令下发成功..." 消息，失败时返回 "指令下发失败: ..." 消息
    """
    payload = {
        "action": "do_experiment",
        "params": {
            "spin_speed": spin_speed,
            "spin_acc": spin_acc,
            "spin_dur": spin_dur,
            "reagent": reagent,
            "volume": volume,
        },
    }
    try:
        client = get_mqtt_client()
        if not client.check_connect():
            client.connect(timeout=2)
        client.publish("do_experiment", json.dumps(payload))
        return f"实验指令下发成功。试剂:{reagent}, 转速:{spin_speed}rpm, 时长:{spin_dur}ms"
    except Exception as e:
        return f"指令下发失败: {str(e)}"


def execute_set_temperature(target: float) -> str:
    """
    底层同步函数：设置加热台温度

    当前为模拟实现（返回确认消息），实际部署时需取消注释
    subprocess 调用以执行真实的 C/C++ 温控程序。

    Args:
        target : 目标温度值（单位：℃）

    Returns:
        str: 温度设置结果消息
    """
    try:
        # TODO: 取消以下注释以连接真实硬件
        # cmd_list = ["./temp_ctrl", "--set", str(target)]
        # res = subprocess.run(cmd_list, capture_output=True, text=True)
        # return res.stdout.strip()
        return f"硬件加热台温度已成功设置为 {target} ℃"
    except Exception as e:
        return f"温度设置失败: {str(e)}"


def execute_move_robot_arm(x: float, y: float, z: float) -> str:
    """
    底层同步函数：移动机械臂到指定坐标位置

    当前为模拟实现，实际部署时需取消注释 subprocess 调用。

    Args:
        x : X 轴坐标
        y : Y 轴坐标
        z : Z 轴坐标

    Returns:
        str: 机械臂移动结果消息
    """
    try:
        # TODO: 取消以下注释以连接真实硬件
        # res = subprocess.run(
        #     ["python", "arm_ctrl.py", str(x), str(y), str(z)],
        #     capture_output=True, text=True,
        # )
        # return res.stdout.strip()
        return f"机械臂已精准移动至坐标 ({x}, {y}, {z})"
    except Exception as e:
        return f"机械臂移动失败: {str(e)}"


def execute_start_experiment() -> str:
    """
    底层同步函数：向自动化平台发送实验序列启动指令 "pstart"

    此函数是 start_experiment() 异步工具函数的同步版本，
    供 core/hardware_controller.py 的前缀命令路径调用。

    Returns:
        str: 成功时返回 "实验序列启动指令已发送"，失败时返回错误消息
    """
    try:
        client = get_mqtt_client()
        if client.is_connected:
            client.publish("do_experiment", "pstart")
            return "实验序列启动指令已发送"
        else:
            if client.connect(timeout=2):
                client.publish("do_experiment", "pstart")
                return "实验序列启动指令已发送"
            else:
                return "MQTT 连接失败，无法启动实验"
    except Exception as e:
        return f"启动实验失败: {str(e)}"


def execute_collect_spectrum(duration: int = 60) -> str:
    """
    底层同步函数：启动光谱仪数据采集

    创建 SpectrometerClient 实例并启动后台采集线程。
    采集的数据会存储在 SpectrometerClient 内部的 DataFrame 中，
    可通过 spec_client.get_latest_data() 获取。

    Args:
        duration : 预计采集时长（秒），默认 60 秒（仅用于提示，不控制实际停止）

    Returns:
        str: 启动结果消息
    """
    try:
        from .spec_client import SpectrometerClient
        spec = SpectrometerClient()
        spec.start_collection()
        return f"光谱仪数据采集已启动，预计持续 {duration} 秒"
    except Exception as e:
        return f"光谱仪启动失败: {str(e)}"
