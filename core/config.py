"""
配置参数管理模块
集中管理所有配置参数，便于维护和扩展
"""

import os
from typing import Optional


class Config:
    """
    配置类 - 集中管理所有配置参数

    职责：
    - 管理API密钥和端点
    - 管理模型名称
    - 管理文件路径
    - 管理其他运行时参数
    """

    # API配置
    SILICONFLOW_API_KEY: str = "sk-zskagakurneevlklkhhzbaxunehikfyeinnjvyizyfstvtci"
    API_URL: str = "https://api.siliconflow.cn/v1/chat/completions"

    # 模型配置
    MODEL_NAME_VL: str = "Qwen/Qwen2.5-VL-72B-Instruct"  # 视觉语言模型
    MODEL_NAME_TALK: str = "Qwen/Qwen2.5-7B-Instruct"    # 对话模型

    # 文件路径配置
    PDF_FOLDER: str = r"test"  # PDF文件存储目录
    EXTRACT_DIR: str = "extract"  # 提取结果存储目录
    TEMPORAL_DIR: str = "temporal"  # 临时文件存储目录
    TEMPLATES_DIR: str = "templates"  # 模板文件目录

    # 处理参数配置
    DPI: int = 200  # PDF转图片的DPI
    REQUEST_DELAY: float = 3.0  # 请求延迟（秒）
    MAX_RETRIES: int = 3  # 最大重试次数
    TIMEOUT: int = 60  # API请求超时时间（秒）
    STREAM_TIMEOUT: int = 90  # 流式请求超时时间（秒）

    # 硬件控制配置
    HARDWARE_TIMEOUT: int = 20  # 硬件控制超时时间（秒）

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
            'SILICONFLOW_API_KEY',
            'API_URL',
            'MODEL_NAME_VL',
            'MODEL_NAME_TALK'
        ]

        for config in required_configs:
            if not getattr(cls, config, None):
                return False

        return True