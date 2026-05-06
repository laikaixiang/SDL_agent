"""
配置加载器 (utils/config_loader.py)
==================================

从项目根目录的 config.json 加载配置，解耦敏感信息与代码。

加载优先级（从高到低）：
  1. 环境变量（同名的环境变量会覆盖 config.json 中的值）
  2. config.json 文件（项目根目录）
  3. 内置硬编码默认值（core/config.py 中的原值）

用法：
    from utils.config_loader import load_config
    cfg = load_config()
    api_key = cfg.get("API_KEY", "fallback_default")

config.json 中以下划线 _ 开头的 key 被视为注释，自动跳过。
"""

import json
import os
from typing import Any, Optional


def _find_project_root() -> str:
    """
    查找项目根目录

    从当前文件位置向上查找，直到找到包含 config.example.json 的目录。
    这样可以保证无论从哪里运行脚本都能正确定位。

    Returns:
        项目根目录的绝对路径
    """
    current = os.path.dirname(os.path.abspath(__file__))
    # utils/ → 项目根目录
    root = os.path.dirname(current)
    return root


def _load_json_config(root: str) -> dict:
    """
    读取 config.json 文件

    Args:
        root: 项目根目录路径

    Returns:
        配置字典（已过滤 _ 开头的注释 key），文件不存在则返回空 dict
    """
    config_path = os.path.join(root, "config.json")
    if not os.path.isfile(config_path):
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 过滤掉 _ 开头的注释 key
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _merge_env_vars(config: dict) -> dict:
    """
    用同名环境变量覆盖配置值

    如果环境中设置了与 config key 同名的变量，优先使用环境变量的值。
    对于 bool 类型，环境变量 "true"/"1"/"yes" → True，"false"/"0"/"no"/"" → False。
    对于 int/float 类型，自动类型转换。

    Args:
        config: 当前配置字典

    Returns:
        合并后的配置字典（可能被环境变量覆盖）
    """
    for key in list(config.keys()):
        env_val = os.environ.get(key)
        if env_val is not None:
            orig = config[key]
            # 根据原值类型做转换
            if isinstance(orig, bool):
                config[key] = env_val.lower() in ("true", "1", "yes")
            elif isinstance(orig, int):
                config[key] = int(env_val)
            elif isinstance(orig, float):
                config[key] = float(env_val)
            else:
                config[key] = env_val
    return config


def load_config() -> dict:
    """
    加载完整配置

    1. 从 config.json 读取
    2. 用环境变量覆盖
    3. 返回配置字典（不包含默认值——默认值由 core/config.py 的类属性提供）

    Returns:
        配置字典，key 为大写的配置项名称
    """
    root = _find_project_root()
    config = _load_json_config(root)
    config = _merge_env_vars(config)
    return config


def get_config_value(key: str, default: Any = None) -> Any:
    """
    获取单个配置值

    依次从 config.json → 环境变量 查找，找不到返回 default。

    Args:
        key: 配置项名称（大写）
        default: 默认值

    Returns:
        配置值或默认值
    """
    cfg = load_config()
    return cfg.get(key, default)
