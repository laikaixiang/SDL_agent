"""
配置参数管理模块 (core/config.py)
================================

集中管理所有配置参数，便于维护和扩展。

配置来源（优先级从高到低）：
  1. 环境变量（同名环境变量直接覆盖）
  2. 项目根目录 config.json（敏感信息存放于此，gitignore 忽略）
  3. 本文件中的硬编码默认值

使用方式：
  - 复制 config.example.json → config.json，填入实际值
  - 或设置同名环境变量
  - 都不设置则使用本文件的默认值（API key 类敏感字段默认为空）

配置分类：
    - API 配置  : 文献提取和普通对话使用的 LLM 服务
    - 实验设计智能体模型配置 : 复用 API 的模型
    - 模型配置              : 各功能模块使用的模型名称
    - Embedding 配置        : RAG 页面预筛选使用的向量化服务
    - 文件路径配置          : PDF 存储、提取结果、模板等目录
    - 处理参数配置          : 超时、重试、延迟等运行时参数
    - 硬件控制配置          : 硬件操作超时时间
    - 光谱仪 MQTT 配置      : 光谱仪数据采集的 MQTT 连接参数
    - 试剂配置              : 试剂布局文件路径
"""

import os
from typing import Optional


def _load_external_config() -> dict:
    """
    尝试从 config.json 加载外部配置

    如果 config.json 存在，用其值覆盖类属性；
    如果不存在，静默跳过，使用硬编码默认值。

    Returns:
        从 config.json 读取的配置字典（仅包含非 _ 前缀的 key）
    """
    import json as _json
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
    if not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = _json.load(f)
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    except Exception:
        return {}


def _apply_env_overrides(config: dict) -> dict:
    """用同名环境变量覆盖配置值（自动类型转换）"""
    for key in list(config.keys()):
        env_val = os.environ.get(key)
        if env_val is not None:
            orig = config[key]
            if isinstance(orig, bool):
                config[key] = env_val.lower() in ("true", "1", "yes")
            elif isinstance(orig, int):
                config[key] = int(env_val)
            elif isinstance(orig, float):
                config[key] = float(env_val)
            else:
                config[key] = env_val
    return config


# 模块加载时读取外部配置
_external = _apply_env_overrides(_load_external_config())


class Config:
    """
    配置类 - 集中管理所有配置参数

    职责：
    - 管理 API 密钥和端点
    - 管理模型名称
    - 管理文件路径
    - 管理其他运行时参数

    配置读取优先级：环境变量 > config.json > 类属性默认值
    """

    # ======================== API 配置（全局默认）========================
    # 当各模块未单独配置 API_KEY/URL 时，使用此全局默认值
    API_KEY: str = _external.get("API_KEY", "")
    # API_URL 是完整的 endpoint，已包含 /chat/completions 路径
    API_URL: str = _external.get("API_URL", "https://api.siliconflow.cn/v1/chat/completions")

    # ======================== 各模块独立 API 配置 ========================
    # 每个模型可独立配置 API 密钥和端点，未设置时回退到全局 API_KEY/API_URL
    # Talk 模型（对话、数据分析、字段推断、算法生成）
    TALK_API_KEY: str = _external.get("TALK_API_KEY") or _external.get("API_KEY", "")
    TALK_API_URL: str = _external.get("TALK_API_URL") or _external.get("API_URL", "https://api.siliconflow.cn/v1/chat/completions")
    # VL 模型（PDF 视觉提取）
    VL_API_KEY: str = _external.get("VL_API_KEY") or _external.get("API_KEY", "")
    VL_API_URL: str = _external.get("VL_API_URL") or _external.get("API_URL", "https://api.siliconflow.cn/v1/chat/completions")
    # 实验设计模型
    EXPERIMENT_API_KEY: str = _external.get("EXPERIMENT_API_KEY") or _external.get("API_KEY", "")
    EXPERIMENT_API_URL: str = _external.get("EXPERIMENT_API_URL") or _external.get("API_URL", "https://api.siliconflow.cn/v1/chat/completions")

    # ======================== Extra Body 配置（供应商特有参数）========================
    # JSON 字符串，会被 merge 到每个 API 请求的 body 中
    # 例: DeepSeek thinking 模式 → {"thinking": {"type": "enabled"}}
    EXTRA_BODY: str = _external.get("EXTRA_BODY", "")
    TALK_EXTRA_BODY: str = _external.get("TALK_EXTRA_BODY") or _external.get("EXTRA_BODY", "")
    VL_EXTRA_BODY: str = _external.get("VL_EXTRA_BODY") or _external.get("EXTRA_BODY", "")
    EXPERIMENT_EXTRA_BODY: str = _external.get("EXPERIMENT_EXTRA_BODY") or _external.get("EXTRA_BODY", "")

    @classmethod
    def get_extra_body(cls, model_type: str) -> dict:
        """解析指定模型类型的 EXTRA_BODY JSON 字符串为 dict，解析失败返回 {} """
        attr = f"{model_type.upper()}_EXTRA_BODY"
        raw = getattr(cls, attr, "")
        if not raw:
            return {}
        try:
            import json as _extra_json
            return _extra_json.loads(raw)
        except Exception:
            return {}

    # ======================== 实验设计智能体模型配置 ========================
    EXPERIMENT_MODEL_NAME: str = _external.get("EXPERIMENT_MODEL_NAME", "Pro/MiniMaxAI/MiniMax-M2.5")

    # ======================== 模型配置 ========================
    MODEL_NAME_VL: str = _external.get("MODEL_NAME_VL", "Qwen/Qwen3-VL-30B-A3B-Instruct")
    MODEL_NAME_TALK: str = _external.get("MODEL_NAME_TALK", "Qwen/Qwen3-VL-30B-A3B-Instruct")

    # ======================== Embedding 配置 ========================
    EMBEDDING_BACKEND: str = _external.get("EMBEDDING_BACKEND", "api")
    EMBEDDING_API_KEY: str = _external.get("EMBEDDING_API_KEY", "")
    EMBEDDING_API_URL: str = _external.get("EMBEDDING_API_URL", "https://api.siliconflow.cn/v1/embeddings")
    EMBEDDING_MODEL: str = _external.get("EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-8B")
    EMBEDDING_DIM: int = _external.get("EMBEDDING_DIM", 4096)
    EMBEDDING_MAX_CHARS: int = _external.get("EMBEDDING_MAX_CHARS", 1000)
    LOCAL_EMBEDDING_MODEL: str = _external.get("LOCAL_EMBEDDING_MODEL", "")

    # ======================== Vector Store 配置 ========================
    VECTOR_STORE_BACKEND: str = _external.get("VECTOR_STORE_BACKEND", "chromadb")
    CHROMADB_PERSIST_DIR: str = _external.get("CHROMADB_PERSIST_DIR", "dialogue data/vector_store")

    # ======================== Page Pre-filter 配置 (Phase 1) ========================
    PAGE_FILTER_ENABLED: bool = _external.get("PAGE_FILTER_ENABLED", True)
    PAGE_FILTER_THRESHOLD: float = _external.get("PAGE_FILTER_THRESHOLD", 0.25)
    PAGE_FILTER_TOP_K: int = _external.get("PAGE_FILTER_TOP_K", 4000)

    # ======================== Few-Shot 配置 (Phase 2) ========================
    FEW_SHOT_ENABLED: bool = _external.get("FEW_SHOT_ENABLED", True)
    FEW_SHOT_TOP_K: int = _external.get("FEW_SHOT_TOP_K", 3)

    # ======================== Semantic Search 配置 (Phase 3) ========================
    SEMANTIC_SEARCH_ENABLED: bool = _external.get("SEMANTIC_SEARCH_ENABLED", True)

    # ======================== 去重配置 ========================
    # DEDUP_NORMALIZE: "strip" | "lower" | "strict"
    # DEDUP_MERGE_STRATEGY: "longest" | "first_non_empty"
    # TODO: 后续可扩展为语义相似度去重（embedding 聚类）
    DEDUP_ENABLED: bool = _external.get("DEDUP_ENABLED", True)
    DEDUP_NORMALIZE: str = _external.get("DEDUP_NORMALIZE", "strip")
    DEDUP_MERGE_STRATEGY: str = _external.get("DEDUP_MERGE_STRATEGY", "longest")
    DEDUP_ADD_METADATA: bool = _external.get("DEDUP_ADD_METADATA", True)

    # ======================== 提取质量检查 ========================
    QUALITY_CHECK_ENABLED: bool = _external.get("QUALITY_CHECK_ENABLED", True)
    QUALITY_SPARSE_THRESHOLD: float = _external.get("QUALITY_SPARSE_THRESHOLD", 0.3)

    # ======================== PDF预览 ========================
    PDF_PREVIEW_ENABLED: bool = _external.get("PDF_PREVIEW_ENABLED", True)

    # ======================== 文件路径配置 ========================
    DIALOGUE_DATA_DIR: str = _external.get("DIALOGUE_DATA_DIR", "dialogue data/history")
    PDF_FOLDER: str = _external.get("PDF_FOLDER", r"dialogue data/PDF_TARGET")
    EXTRACT_DIR: str = _external.get("EXTRACT_DIR", "dialogue data/extract")
    TEMPORAL_DIR: str = _external.get("TEMPORAL_DIR", "dialogue data/temporal")

    # ======================== 处理参数配置 ========================
    DPI: int = _external.get("DPI", 200)
    REQUEST_DELAY: float = _external.get("REQUEST_DELAY", 3.0)
    MAX_RETRIES: int = _external.get("MAX_RETRIES", 3)
    TIMEOUT: int = _external.get("TIMEOUT", 60)
    STREAM_TIMEOUT: int = _external.get("STREAM_TIMEOUT", 90)
    # 全局 max_tokens，None 表示不限制（不传该参数，由 API 自行决定）
    # 设具体数值则所有模型调用统一生效
    MAX_TOKENS: Optional[int] = _external.get("MAX_TOKENS", None)

    # ======================== PDF提取模式配置 ========================
    EXTRACTION_MODE: str = _external.get("EXTRACTION_MODE", "vision")

    # ======================== 硬件控制配置 ========================
    HARDWARE_TIMEOUT: int = _external.get("HARDWARE_TIMEOUT", 20)

    # ======================== 光谱仪 MQTT 配置 ========================
    SPECTROMETER_BROKER_IP: str = _external.get("SPECTROMETER_BROKER_IP", "192.168.120.129")
    SPECTROMETER_BROKER_PORT: int = _external.get("SPECTROMETER_BROKER_PORT", 1883)
    SPECTROMETER_CLIENT_ID: str = _external.get("SPECTROMETER_CLIENT_ID", "987zyx")
    SPECTROMETER_USERNAME: str = _external.get("SPECTROMETER_USERNAME", "")
    SPECTROMETER_PASSWORD: str = _external.get("SPECTROMETER_PASSWORD", "")

    # ======================== 试剂配置 ========================
    REAGENT_LAYOUT_PATH: str = _external.get("REAGENT_LAYOUT_PATH", "reagent_layout.json")

    # ======================== 文献元数据提取配置 ========================
    LITERATURE_REGISTRY_DB_PATH: str = _external.get("LITERATURE_REGISTRY_DB_PATH", "dialogue data/PDF_TARGET/literature_registry.db")
    METADATA_EXTRACTION_MODEL: str = _external.get("METADATA_EXTRACTION_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct")
    BATCH_MAX_WORKERS: int = _external.get("BATCH_MAX_WORKERS", 3)
    METADATA_EXTRACTION_TIMEOUT: int = _external.get("METADATA_EXTRACTION_TIMEOUT", 120)
    METADATA_RETRY_ATTEMPTS: int = _external.get("METADATA_RETRY_ATTEMPTS", 2)

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
        # 至少需要一种可用的 API 配置（全局或独立均可）
        has_credentials = bool(
            cls.API_KEY or cls.TALK_API_KEY or cls.VL_API_KEY or cls.EXPERIMENT_API_KEY
        )
        has_url = bool(
            cls.API_URL or cls.TALK_API_URL or cls.VL_API_URL or cls.EXPERIMENT_API_URL
        )
        has_models = bool(cls.MODEL_NAME_VL and cls.MODEL_NAME_TALK)
        return has_credentials and has_url and has_models

if __name__ == "__main__":
    # 测试读取模型名
    print(Config.get_config("MODEL_NAME_VL"))