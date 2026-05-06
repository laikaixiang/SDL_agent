"""
Phase 1: 页面预筛选模块
=======================

在 PDF 提取任务中，根据任务描述与页面内容的语义相似度决定是否处理某一页。
跳过与任务无关的页面，减少 LLM 调用次数，节省时间和 Token 成本。

核心流程：
1. set_task("提取FAPbI3钝化剂参数") → 将任务描述转为 embedding 向量缓存
2. should_process(pdf_path, page_num) → 逐页检查相似度是否超过阈值

相似度计算：
  使用余弦相似度（cosine similarity）：dot(A, B) / (|A| * |B|)
  值域 [-1, 1]，通常对于非负 embedding 在 [0, 1] 之间
  - 1.0 = 完全一致
  - 0.0 = 完全不相关
  - 默认阈值 0.3，偏向保守（宁可多处理，不可漏数据）

异常处理：
  - 如果某个页面尚未被索引（向量库中不存在），默认返回 True（不跳过）
  - 如果 set_task() 未被调用，默认处理所有页面

调优建议：
  - 阈值 0.3：保守，适合首次使用，几乎不漏数据
  - 阈值 0.5：中等，跳过明显不相关的页面（如纯参考文献页）
  - 阈值 0.7：激进，只处理高度相关的页面，适合精确查询但可能漏数据
"""

import math
from typing import Optional

from .config import Config
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .page_indexer import make_page_id


class PageFilter:
    """
    Phase 1 页面预筛选器

    在提取任务中逐页判断是否需要发送给 LLM 处理。
    基于 embedding 向量余弦相似度，将任务描述与每个页面的预存储向量做比较。

    使用方式：
        pf = PageFilter(embedding_service, vector_store, threshold=0.3)
        pf.set_task("提取 FAPbI3 钝化剂参数")
        for page in pages:
            if pf.should_process(pdf_path, page_num):
                # 发送此页给 LLM
            else:
                # 跳过此页
    """

    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore,
                 threshold: float = 0.3, top_k: int = 20):
        """
        初始化页面筛选器

        Args:
            embedding_service: Embedding 服务实例，用于将任务描述转为向量
            vector_store: 向量存储实例，用于获取页面预存储向量
            threshold: 余弦相似度阈值（默认 0.3），相似度 >= 此值才处理
            top_k: 最大处理页数限制（预留，Phase 1 未实现 top-k 截断）
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.threshold = threshold
        self.top_k = top_k
        # 缓存任务描述的 embedding 向量，避免每次 should_process 都重新计算
        self._query_embedding: Optional[list[float]] = None

    def set_task(self, task_description: str):
        """
        设置当前提取任务

        将任务描述文本转为 embedding 向量并缓存。
        必须在第一次调用 should_process() 之前调用。
        可以多次调用以更换任务。

        Args:
            task_description: 任务描述文本（如 "提取 FAPbI3 钝化剂参数"）
        """
        self._query_embedding = self.embedding_service.embed_text(task_description)

    def should_process(self, pdf_path: str, page_num: int) -> bool:
        """
        判断某个页面是否应该被处理（发送给 LLM 提取数据）

        计算流程：
        1. 通过 make_page_id 获取页面 ID
        2. 从 VectorStore 获取该页面的预存储 embedding
        3. 如果页面尚未索引，保守处理 → 返回 True
        4. 计算任务描述向量与页面向量之间的余弦相似度
        5. 相似度 >= threshold → 返回 True（处理），否则 → False（跳过）

        Args:
            pdf_path: PDF 文件路径
            page_num: 页码（从 0 开始）

        Returns:
            True 表示应该处理此页，False 表示可以跳过
        """
        if self._query_embedding is None:
            # 未设置任务，保守处理：不跳过任何页面
            return True

        page_id = make_page_id(pdf_path, page_num)
        page_embedding = self.vector_store.get_embedding(page_id)

        if page_embedding is None:
            # 页面尚未索引，无法判断相关性，保守处理：不跳过
            return True

        similarity = self._cosine_similarity(self._query_embedding, page_embedding)
        return similarity >= self.threshold

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """
        计算两个向量的余弦相似度

        余弦相似度 = 向量点积 / (向量模长之积)
        - 值域：[-1, 1]（对于非负 embedding 在 [0, 1] 之间）
        - 1.0 = 方向完全一致（高度相关）
        - 0.0 = 正交（不相关）
        - -1.0 = 完全相反

        Args:
            a: 向量 A
            b: 向量 B

        Returns:
            余弦相似度值
        """
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
