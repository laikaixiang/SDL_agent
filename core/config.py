"""
配置参数管理模块 (core/config.py)
================================

集中管理所有配置参数，便于维护和扩展。

配置分类：
    - API 配置  : 文献提取和普通对话使用的 LLM 服务
    - 实验设计智能体模型配置 : 复用 API 的模型
    - 模型配置              : 各功能模块使用的模型名称
    - 文件路径配置          : PDF 存储、提取结果、模板等目录
    - 处理参数配置          : 超时、重试、延迟等运行时参数
    - 硬件控制配置          : 硬件操作超时时间
    - 光谱仪 MQTT 配置      : 光谱仪数据采集的 MQTT 连接参数
    - 试剂配置              : 试剂布局文件路径
    - 实验设计智能体配置    : PydanticAI Agent 的系统提示词
"""

import os
from typing import Optional


class Config:
    """
    配置类 - 集中管理所有配置参数

    职责：
    - 管理 API 密钥和端点
    - 管理模型名称
    - 管理文件路径
    - 管理其他运行时参数

    使用示例::

        config = Config()
        print(config.API_URL)                    # API 地址
        print(config.EXPERIMENT_MODEL_NAME)      # 实验设计Agent模型名称
        print(config.EXPERIMENT_AGENT_SYSTEM_PROMPT)  # 实验设计Agent提示词
    """

    # ======================== API 配置 ========================
    # 用于文献提取（视觉语言模型）和普通对话（文本模型）
    API_KEY: str = "sk-zskagakurneevlklkhhzbaxunehikfyeinnjvyizyfstvtci"
    API_URL: str = "https://api.siliconflow.cn/v1/chat/completions"

    # ======================== 实验设计智能体模型配置 ========================
    # 实验设计智能体使用的API，使用对话模型
    EXPERIMENT_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"

    # ======================== 模型配置 ========================
    MODEL_NAME_VL: str = "Qwen/Qwen2.5-VL-72B-Instruct"  # 视觉语言模型（文献图表识别）
    MODEL_NAME_TALK: str = "Qwen/Qwen2.5-7B-Instruct"  # 对话模型（普通聊天、命令解析）

    # ======================== 文件路径配置 ========================
    PDF_FOLDER: str = r"test"          # PDF 文件存储目录（用户上传的文献）
    EXTRACT_DIR: str = "extract"       # 提取结果存储目录（CSV 文件）
    TEMPORAL_DIR: str = "temporal"     # 临时文件存储目录（实时任务数据）
    TEMPLATES_DIR: str = "templates"   # HTML 模板文件目录

    # ======================== 处理参数配置 ========================
    DPI: int = 200                     # PDF 转图片的 DPI（分辨率）
    REQUEST_DELAY: float = 3.0         # API 请求间隔延迟（秒），避免触发限流
    MAX_RETRIES: int = 3               # API 请求最大重试次数
    TIMEOUT: int = 60                  # 普通 API 请求超时时间（秒）
    STREAM_TIMEOUT: int = 90           # 流式 API 请求超时时间（秒）

    # ======================== 硬件控制配置 ========================
    HARDWARE_TIMEOUT: int = 20         # 硬件控制操作超时时间（秒）

    # ======================== 光谱仪 MQTT 配置 ========================
    # 光谱仪数据采集使用独立的 MQTT 客户端连接到 EMQX 服务器
    SPECTROMETER_BROKER_IP: str = "192.168.120.129"    # EMQX 服务器 IP 地址
    SPECTROMETER_BROKER_PORT: int = 1883               # MQTT 端口号
    SPECTROMETER_CLIENT_ID: str = "987zyx"             # 光谱仪客户端 ID
    SPECTROMETER_USERNAME: str = "s208"                # MQTT 认证用户名
    SPECTROMETER_PASSWORD: str = "s208ht"              # MQTT 认证密码

    # ======================== 试剂配置 ========================
    # 试剂布局配置文件路径（JSON 格式），记录每个位置上装载的试剂名称
    # 默认位于项目根目录的 reagent_layout.json
    REAGENT_LAYOUT_PATH: str = "reagent_layout.json"

    # ======================== 实验设计智能体配置 ========================
    # PydanticAI Agent 的系统提示词，定义 AI 的角色和可用工具
    # AI 会根据此提示词理解自己的能力边界，并自主决定调用哪些工具
    EXPERIMENT_AGENT_SYSTEM_PROMPT: str = (
        "You are an experienced materials scientist. "
        "When the user uploads a PDF, you can read it with `read_pdf(file_path, page_number)`. "
        "The file path will be provided by the system. "
        "You may also register spin-coating steps with `save_experiment_step`. "
        "You can start the whole experiment round with `start_experiment`. "
        "When the human scientist asks you to do spin-coating experiments, remember to check how many steps the "
        "experiment round takes."
    )

    @classmethod
    def get_config(cls, key: str) -> Optional[str]:
        """
        获取配置值

        Args:
            key: 配置键名

        Returns:
            配置值或None
        """
        return getattr(cls, key, None)

    @classmethod
    def set_config(cls, key: str, value: str) -> None:
        """
        设置配置值

        Args:
            key: 配置键名
            value: 配置值
        """
        setattr(cls, key, value)

    @classmethod
    def validate_config(cls) -> bool:
        """
        验证必要配置是否存在

        Returns:
            配置是否有效
        """
        required_configs = [
            'API_KEY',
            'API_URL',
            'MODEL_NAME_VL',
            'MODEL_NAME_TALK'
        ]

        for config in required_configs:
            if not getattr(cls, config, None):
                return False

        return True