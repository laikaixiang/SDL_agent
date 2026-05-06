"""
Embedding 服务抽象层
====================

提供多模态 embedding 的统一接口，支持文本嵌入（部分后端支持图片混合输入）。

后端选择（config.EMBEDDING_BACKEND）:
  "api"   — 通用 OpenAI 兼容接口，通过 EMBEDDING_API_URL / EMBEDDING_MODEL 自由配置
            已测试: SiliconFlow (BAAI/bge-large-zh-v1.5 / BAAI/bge-m3)
            兼容: DeepSeek、Jina、OpenAI 等任意 OpenAI 格式接口
  "jina"  — Jina AI 原生接口，支持 text+image 多模态输入（jina-clip-v2）
  "local" — 本地模型（TODO，预留接口）

切换后端只需修改 config 对应字段，上层代码无需改动。
"""

import time
from abc import ABC, abstractmethod
from typing import Optional

import requests

from core.config import Config


class EmbeddingService(ABC):
    """
    Embedding 服务抽象基类

    定义 embedding 的统一接口，所有后端（云端 API / 本地模型）必须实现此接口。
    支持三种调用模式：
    1. embed_page  — 单页面嵌入（文本 + 可选图片），用于 PDF 页面预索引
    2. embed_text  — 纯文本嵌入，用于任务描述等查询文本
    3. embed_batch — 批量嵌入，用于一次性摄入多个页面，减少 API 调用次数
    """

    @abstractmethod
    def embed_page(self, text: str, image_base64: Optional[str]) -> list[float]:
        """
        对单个 PDF 页面进行嵌入

        输入页面的文本内容（必填）和图片 base64（可选），返回一个固定维度的浮点向量。
        注意：文本模型（如 BGE 系列）不支持图片输入，此时 image_base64 参数被忽略，
        仅用文本进行嵌入。

        Args:
            text: 页面的文本内容（通过 fitz 提取的原始文本）
            image_base64: 页面的 base64 编码图片（JPEG），为 None 表示纯文本嵌入

        Returns:
            浮点数列表，维度取决于模型配置（BGE 默认 1024 维）
        """
        ...

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """
        纯文本嵌入（用于任务描述等查询文本）

        PageFilter 在过滤前会先将任务描述转为向量，然后与每个页面的向量计算余弦相似度。

        Args:
            text: 任务描述文本（如 "提取 FAPbI3 钝化剂参数"）

        Returns:
            浮点数列表，维度与 embed_page 一致，确保可直接比较
        """
        ...

    @abstractmethod
    def embed_batch(self, pages: list[dict]) -> list[list[float]]:
        """
        批量嵌入（用于 PDF 摄入阶段，减少 API 调用轮次）

        Args:
            pages: 页面列表，每个元素为 {"text": str, "image_base64": Optional[str]}

        Returns:
            嵌套列表，外层索引对应输入 pages 的顺序
        """
        ...


class APIEmbeddingService(EmbeddingService):
    """
    通用 OpenAI 兼容 Embedding API 实现（当前默认后端）

    支持任意 OpenAI 接口格式的云端 embedding 服务，包括但不限于：
    - SiliconFlow: https://api.siliconflow.cn/v1/embeddings
      推荐模型: BAAI/bge-large-zh-v1.5（中文优化，1024维）
               BAAI/bge-m3（多语言，1024维，支持稀疏+稠密检索）
    - DeepSeek: https://api.deepseek.com/v1/embeddings
      模型: deepseek-embedding-v1（1024维，超低价）
    - 其他兼容 OpenAI 格式的服务商

    请求格式（OpenAI 兼容）:
      POST {api_url}
      Body: {"model": "...", "input": "text" | ["t1", "t2", ...]}
      Response: {"data": [{"index": 0, "embedding": [...]}, ...]}

    注意：此实现为纯文本 embedding，不支持图片输入。
    如果页面需要图片理解，请切换到 EMBEDDING_BACKEND="jina" 使用 jina-clip-v2。

    重试策略：单次 API 调用最多重试 3 次，每次间隔 2 秒，最终失败抛 RuntimeError。
    """

    def __init__(self, api_key: str, model: str = "BAAI/bge-large-zh-v1.5",
                 api_url: str = "https://api.siliconflow.cn/v1/embeddings"):
        """
        初始化 API Embedding 服务

        Args:
            api_key: 云端服务的 API Key
            model: 模型名称，默认 BAAI/bge-large-zh-v1.5（SiliconFlow 中文模型）
            api_url: API 端点地址
        """
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        # 从 Config 读取截断长度，默认 1000（保守适应 512 token 限制）
        from core.config import Config
        self._max_chars = Config.EMBEDDING_MAX_CHARS

    def _truncate(self, text: str) -> str:
        """截断过长文本，防止 API 413 错误"""
        if len(text) > self._max_chars:
            return text[:self._max_chars]
        return text

    def _call_api(self, inputs: list[str]) -> list[list[float]]:
        """
        底层 API 调用（OpenAI 兼容格式，带重试）

        OpenAI 兼容的请求体格式：
        - input 为单个字符串时返回单个 embedding
        - input 为字符串列表时返回多个 embedding
        - 响应中 data 数组按 index 排序，取 embedding 字段

        Args:
            inputs: 文本字符串列表

        Returns:
            按 index 排序后的 embedding 向量列表

        Raises:
            RuntimeError: 3 次重试全部失败
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # OpenAI 格式：input 支持单字符串或字符串列表
        if len(inputs) == 1:
            payload = {"model": self.model, "input": inputs[0]}
        else:
            payload = {"model": self.model, "input": inputs}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url, headers=headers, json=payload, timeout=60
                )
                response.raise_for_status()
                result = response.json()
                # 单字符串输入时，返回的 data 可能只有一个元素
                data = result["data"]
                if not isinstance(data, list):
                    data = [data]
                # 按 index 排序以保证返回顺序与输入一致
                embeddings = sorted(data, key=lambda x: x["index"])
                # 强制转换为 float：API 可能返回 int 类型值（如整数 0），
                # 统一转为 float 保证下游余弦相似度计算不会因类型检查失败
                return [[float(x) for x in item["embedding"]] for item in embeddings]
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2.0)
                else:
                    raise RuntimeError(
                        f"Embedding API 调用失败（已重试 {max_retries} 次）: {e}"
                    )

    def embed_page(self, text: str, image_base64: Optional[str]) -> list[float]:
        """
        对单个页面进行嵌入

        当前 API 后端为纯文本模型，不支持图片输入，image_base64 参数会被忽略。
        如果需要多模态（文本+图片）嵌入，请切换到 EMBEDDING_BACKEND="jina"。

        Args:
            text: 页面文本内容
            image_base64: 页面截图 base64（当前后端不支持，会被忽略）

        Returns:
            浮点向量（维度取决于模型，BGE 默认 1024 维）
        """
        return self.embed_text(text)

    def embed_text(self, text: str) -> list[float]:
        """
        纯文本嵌入

        Args:
            text: 任务描述文本（自动截断到 _max_chars 避免 API 413 错误）

        Returns:
            浮点向量
        """
        results = self._call_api([self._truncate(text)])
        return results[0]

    def embed_batch(self, pages: list[dict]) -> list[list[float]]:
        """
        批量嵌入多个页面

        将多个页面的文本打包为一次 API 调用，减少网络往返次数。
        建议每批不超过 100 个页面以避免单次请求过大。

        Args:
            pages: 页面字典列表，每个字典包含 text（必填）和可选的 image_base64

        Returns:
            嵌入向量列表，长度和顺序与输入一致
        """
        texts = [self._truncate(p["text"]) for p in pages]
        return self._call_api(texts)


class JinaEmbeddingService(EmbeddingService):
    """
    Jina AI Embedding API 实现（多模态：文本 + 图片）

    使用 Jina AI 的 jina-clip-v2 模型，支持文本 + 图片多模态输入。
    与 APIEmbeddingService 的区别在于 input 格式：
    - Jina 格式: [{"text": "...", "image": "base64..."}, ...]
    - OpenAI 格式: "text" 或 ["text1", "text2"]

    使用前提：
    1. EMBEDDING_BACKEND="jina"
    2. EMBEDDING_API_KEY 设置为有效的 Jina AI API Key
    3. EMBEDDING_MODEL="jina-clip-v2"、EMBEDDING_API_URL="https://api.jina.ai/v1/embeddings"

    重试策略：单次 API 调用最多重试 3 次，每次间隔 2 秒，最终失败抛 RuntimeError。
    """

    def __init__(self, api_key: str, model: str = "jina-clip-v2",
                 api_url: str = "https://api.jina.ai/v1/embeddings"):
        """
        初始化 Jina Embedding 服务

        Args:
            api_key: Jina AI 的 API Key
            model: 模型名称，默认 jina-clip-v2（支持多模态图文输入）
            api_url: API 端点地址
        """
        self.api_key = api_key
        self.model = model
        self.api_url = api_url

    def _call_api(self, inputs: list[dict]) -> list[list[float]]:
        """
        底层 API 调用（Jina 多模态格式，带重试）

        Jina API 的 input 格式为对象列表，每个对象可包含 text 和 image 键：
          [{"text": "hello"}, {"text": "world", "image": "base64..."}]

        Args:
            inputs: 输入列表，每个元素为 {"text": str, "image"?: str}

        Returns:
            按 index 排序后的 embedding 向量列表

        Raises:
            RuntimeError: 3 次重试全部失败
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {"model": self.model, "input": inputs}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url, headers=headers, json=payload, timeout=60
                )
                response.raise_for_status()
                result = response.json()
                embeddings = sorted(result["data"], key=lambda x: x["index"])
                return [[float(x) for x in item["embedding"]] for item in embeddings]
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2.0)
                else:
                    raise RuntimeError(
                        f"Jina embedding API 调用失败（已重试 {max_retries} 次）: {e}"
                    )

    def embed_page(self, text: str, image_base64: Optional[str]) -> list[float]:
        """
        对单个页面进行多模态嵌入

        同时传入文本和可选图片，Jina 模型会将两者融合为单一语义向量。
        这是唯一支持图片输入的 embedding 后端。

        Args:
            text: 页面文本内容
            image_base64: 页面截图 base64（JPEG），为 None 时仅用文本

        Returns:
            1024 维浮点向量（jina-clip-v2）
        """
        input_item: dict = {"text": text}
        if image_base64:
            input_item["image"] = image_base64
        results = self._call_api([input_item])
        return results[0]

    def embed_text(self, text: str) -> list[float]:
        """纯文本嵌入"""
        results = self._call_api([{"text": text}])
        return results[0]

    def embed_batch(self, pages: list[dict]) -> list[list[float]]:
        """
        批量嵌入多个页面（支持图文混合）

        Args:
            pages: 页面字典列表，每个字典包含 text（必填）和可选的 image_base64

        Returns:
            嵌入向量列表，长度和顺序与输入一致
        """
        inputs = []
        for page in pages:
            item: dict = {"text": page["text"]}
            if page.get("image_base64"):
                item["image"] = page["image_base64"]
            inputs.append(item)
        return self._call_api(inputs)


class LocalEmbeddingService(EmbeddingService):
    """
    TODO: 本地模型实现（预留接口，未来实现）

    设计动机：
    - 避免 API 延迟（本地推理 < 100ms vs API 200-500ms）
    - 无 API 费用，适合大规模文档库
    - 数据不出本地，满足某些机构的合规要求

    实现计划：
    - 使用 sentence-transformers 或 FastEmbed 加载模型（如 BAAI/bge-m3）
    - 首次加载时缓存模型到内存
    - 配置切换：EMBEDDING_BACKEND="local" + LOCAL_EMBEDDING_MODEL 指定模型路径
    """

    def __init__(self, model_path: str = ""):
        self.model_path = model_path
        raise NotImplementedError("本地 embedding 模型尚未实现")

    def embed_page(self, text, image_base64):
        raise NotImplementedError

    def embed_text(self, text):
        raise NotImplementedError

    def embed_batch(self, pages):
        raise NotImplementedError


def create_embedding_service() -> EmbeddingService:
    """
    工厂函数：根据 Config 中的 EMBEDDING_BACKEND 创建对应的 EmbeddingService 实例

    支持的后端：
    - "api"   — APIEmbeddingService（通用 OpenAI 兼容接口，默认）
               读取 EMBEDDING_API_KEY / EMBEDDING_API_URL / EMBEDDING_MODEL
    - "jina"  — JinaEmbeddingService（Jina 原生多模态接口，支持图文混合输入）
               读取 EMBEDDING_API_KEY / EMBEDDING_API_URL / EMBEDDING_MODEL
    - "local" — LocalEmbeddingService（TODO，尚未实现）
               读取 LOCAL_EMBEDDING_MODEL

    设计意图：上层代码（PageFilter、PageIndexer）只依赖 EmbeddingService 抽象接口，
    切换后端只需修改配置文件中的 EMBEDDING_BACKEND 和相关参数，无需改动业务代码。

    Returns:
        EmbeddingService 的具体实现实例

    Raises:
        ValueError: EMBEDDING_BACKEND 未知，或 api 后端缺少 EMBEDDING_API_KEY
    """
    config = Config()
    backend = config.EMBEDDING_BACKEND

    if backend == "api":
        if not config.EMBEDDING_API_KEY:
            raise ValueError("EMBEDDING_BACKEND='api' 需要设置 EMBEDDING_API_KEY")
        return APIEmbeddingService(
            api_key=config.EMBEDDING_API_KEY,
            model=config.EMBEDDING_MODEL,
            api_url=config.EMBEDDING_API_URL,
        )
    elif backend == "jina":
        if not config.EMBEDDING_API_KEY:
            raise ValueError("EMBEDDING_BACKEND='jina' 需要设置 EMBEDDING_API_KEY")
        return JinaEmbeddingService(
            api_key=config.EMBEDDING_API_KEY,
            model=config.EMBEDDING_MODEL,
            api_url=config.EMBEDDING_API_URL,
        )
    elif backend == "local":
        return LocalEmbeddingService(model_path=config.LOCAL_EMBEDDING_MODEL)

    raise ValueError(f"未知的 embedding 后端: {backend}，可选值为 'api' / 'jina' / 'local'")
