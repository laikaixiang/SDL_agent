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

    """

    # ======================== API 配置 ========================
    # 用于文献提取（视觉语言模型）和普通对话（文本模型）
    API_KEY: str = "sk-zskagakurneevlklkhhzbaxunehikfyeinnjvyizyfstvtci"
    # 注意：API_URL 是完整的 endpoint，已包含 /chat/completions 路径
    # 直接使用即可，不需要再拼接 /chat/completions
    # 如果需要 base_url（如 PydanticAI），使用 API_URL.rsplit('/chat/completions', 1)[0]
    API_URL: str = "https://api.siliconflow.cn/v1/chat/completions"

    # ======================== 实验设计智能体模型配置 ========================
    # 实验设计智能体使用的API，使用对话模型
    EXPERIMENT_MODEL_NAME: str = "Pro/MiniMaxAI/MiniMax-M2.5"

    # ======================== 模型配置 ========================
    MODEL_NAME_VL: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"  # 视觉语言模型（文献图表识别）
    MODEL_NAME_TALK: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"  # 对话模型（普通聊天、命令解析）

    # ======================== 文件路径配置 ========================
    DIALOGUE_DATA_DIR: str = "dialogue data/history"  # 历史对话数据根目录
    PDF_FOLDER: str = r"dialogue data/PDF_TARGET"          # PDF 文件存储目录（用户上传的文献）
    EXTRACT_DIR: str = "dialogue data/extract"  # 提取结果存储目录（CSV 文件）
    TEMPORAL_DIR: str = "dialogue data/temporal"     # 临时文件存储目录（实时任务数据）
    TEMPLATES_DIR: str = "templates"   # HTML 模板文件目录

    # ======================== 处理参数配置 ========================
    DPI: int = 200                     # PDF 转图片的 DPI（分辨率）
    REQUEST_DELAY: float = 3.0         # API 请求间隔延迟（秒），避免触发限流
    MAX_RETRIES: int = 3               # API 请求最大重试次数
    TIMEOUT: int = 60                  # 普通 API 请求超时时间（秒）
    STREAM_TIMEOUT: int = 90           # 流式 API 请求超时时间（秒）

    # ======================== PDF提取模式配置 ========================
    # 提取模式：
    #   - "vision": 纯视觉模式，将PDF转图片后用Vision API分析（准确但贵）
    #   - "text": 纯文本模式，提取PDF文本后用文本API分析（快速便宜但可能丢失图表）
    #   - "hybrid": 混合模式，先提取文本判断复杂度，复杂内容用Vision，简单内容用文本（推荐）
    EXTRACTION_MODE: str = "vision"

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
    # TODO: 修改试剂布局
    REAGENT_LAYOUT_PATH: str = "reagent_layout.json"

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