"""
向量存储抽象层
==============

提供统一的向量存储接口，解耦上层业务逻辑与底层存储实现。
当前实现：ChromaVectorStore（基于 ChromaDB，使用余弦距离）
预留接口：PgvectorVectorStore（PostgreSQL + pgvector，面向大规模部署）

设计原则：
- 所有上层代码（PageFilter、PageIndexer）只依赖 VectorStore 抽象接口
- 切换后端只需修改 config.VECTOR_STORE_BACKEND，无需改动业务代码
- ChromaDB 适合当前阶段（< 50 万条向量），pgvector 为未来扩展预留
"""

from abc import ABC, abstractmethod


class VectorStore(ABC):
    """
    向量存储抽象基类

    定义向量的 CRUD 和搜索操作的标准接口。
    Phase 1-3 当前使用 ChromaDB 实现，pgvector 为大规模部署的迁移路径。

    核心操作：
    - add_embeddings：批量写入向量 + 元数据（upsert 语义，同 ID 覆盖）
    - search：按余弦距离搜索最相似的向量
    - get_embedding：按 ID 获取单个向量（PageFilter 逐页检查时使用）
    - exists / delete / count：基础管理操作
    """

    @abstractmethod
    def add_embeddings(self, ids: list[str], embeddings: list[list[float]],
                       metadatas: list[dict]) -> None:
        """
        批量添加（或更新）向量及其元数据

        使用 upsert 语义：如果某个 id 已存在，则用新的 embedding 和 metadata 覆盖旧记录；
        如果 id 不存在，则新增。这保证了重复索引同一页面不会产生重复记录。

        Args:
            ids: 页面唯一标识符列表，通常通过 make_page_id() 生成
            embeddings: 对应的 embedding 向量列表，每个为浮点数列表
            metadatas: 对应的元数据列表，每个为字典，包含 pdf_path、page_num、content_hash 等
        """
        ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 20,
               where: dict = None) -> list[dict]:
        """
        按向量相似度搜索（余弦距离）

        将查询文本（任务描述）的 embedding 向量与存储中的所有向量做相似度比较，
        返回距离最近的 top_k 条记录。距离越小表示越相似（ChromaDB 使用余弦距离）。

        主要用于 Phase 3 的语义搜索功能，Phase 1 中暂不直接使用。
        Phase 1 的 PageFilter 使用 get_embedding + 自行计算余弦相似度的方式。

        Args:
            query_embedding: 查询文本的 embedding 向量
            top_k: 返回的最相似记录数
            where: 可选的元数据过滤条件（如 {"pdf_path": "xxx"}），ChromaDB 支持

        Returns:
            结果列表，每项格式: {"id": str, "metadata": dict, "distance": float}
            按 distance 升序排列（越小越相似）
        """
        ...

    @abstractmethod
    def get_embedding(self, id: str) -> list[float] | None:
        """
        按 ID 获取单个 embedding 向量

        PageFilter 在 should_process() 中逐页检查时，通过此方法获取每个页面
        的预存储向量，然后与任务描述向量计算余弦相似度。

        Args:
            id: 页面唯一标识符（page_id）

        Returns:
            浮点数列表的 embedding 向量，如果 ID 不存在则返回 None
        """
        ...

    @abstractmethod
    def exists(self, id: str) -> bool:
        """
        检查某个页面是否已被索引

        Args:
            id: 页面唯一标识符

        Returns:
            True 表示该页面已经在向量库中
        """
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """
        批量删除指定的 embedding 记录

        用于清理过期数据或重新索引某些页面时删除旧记录。

        Args:
            ids: 要删除的页面标识符列表
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """
        获取向量库中存储的 embedding 总数

        Returns:
            已存储的向量数量
        """
        ...


class ChromaVectorStore(VectorStore):
    """
    ChromaDB 向量存储实现（当前默认后端）

    基于 ChromaDB 的 PersistentClient，数据持久化到本地磁盘。
    使用余弦距离（hnsw:space = cosine）作为相似度度量。

    初始化时自动创建或打开名为 page_embeddings 的集合。
    数据存储在 config.CHROMADB_PERSIST_DIR 目录下（默认 "dialogue data/vector_store"）。

    适用场景：
    - 当前规模的 PDF 文献库（< 5 万篇 PDF，< 50 万页）
    - 单机部署
    - 无需外部数据库依赖

    不适用场景：
    - 超大规模（> 50 万条向量），此时应考虑迁移到 pgvector
    - 多进程并发写入（ChromaDB 不支持，需外部加锁）
    """

    def __init__(self, persist_dir: str, collection_name: str = "page_embeddings"):
        """
        初始化 ChromaDB 客户端和集合

        Args:
            persist_dir: ChromaDB 数据持久化目录，应使用 config.CHROMADB_PERSIST_DIR
            collection_name: 集合名称，默认 "page_embeddings"
                           可通过不同名称创建多个集合以隔离不同用途
        """
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_dir)
        # get_or_create_collection：如果集合已存在则打开，否则自动创建
        # hnsw:space = cosine 指定使用余弦距离
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_embeddings(self, ids, embeddings, metadatas):
        """
        批量写入或更新 embedding

        使用 ChromaDB 的 upsert 方法：传入的 id 如果已存在，则更新其 embedding
        和 metadata；如果不存在，则新增。这保证了重复索引操作的幂等性。

        Args:
            ids: 页面 ID 列表
            embeddings: embedding 向量列表
            metadatas: 元数据字典列表，包含 pdf_path、page_num、content_hash
        """
        self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def search(self, query_embedding, top_k=20, where=None):
        """
        按余弦距离搜索最相似的向量

        使用 ChromaDB 的 query 方法，query_embeddings 传入查询向量，
        n_results 控制返回条数，where 可附加元数据过滤。

        距离含义：余弦距离 = 1 - 余弦相似度
        - distance = 0 表示完全相同
        - distance = 1 表示完全不相关
        - PageFilter 的阈值比较时会转换为相似度（1 - distance）

        Args:
            query_embedding: 查询向量
            top_k: 返回的最相似记录数
            where: 元数据过滤条件，如 {"pdf_path": "/path/to/file.pdf"}

        Returns:
            结果列表，每项 {"id": str, "metadata": dict, "distance": float}
            按 distance 从小到大排列
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )
        formatted = []
        # ChromaDB 返回格式：results['ids'] 是嵌套列表 [[id1, id2, ...]]
        if results['ids'] and results['ids'][0]:
            for i, id_ in enumerate(results['ids'][0]):
                formatted.append({
                    "id": id_,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0.0
                })
        return formatted

    def get_embedding(self, id):
        """
        按 ID 获取单个 embedding 向量

        这是 PageFilter.should_process() 的核心依赖：获取页面的预存储向量后，
        与任务描述向量做余弦相似度计算，判断是否超过阈值。

        Args:
            id: 页面唯一标识符

        Returns:
            embedding 向量（浮点数列表），如果 ID 不存在返回 None
        """
        result = self.collection.get(ids=[id], include=["embeddings"])
        if result['ids']:
            emb = result['embeddings'][0]
            if hasattr(emb, 'tolist'):
                return emb.tolist()
            return list(emb)
        return None

    def exists(self, id):
        """
        检查指定 ID 是否已存在于向量库中

        用于 PageIndexer 判断是否需要重新索引或跳过。

        Args:
            id: 页面唯一标识符

        Returns:
            True 表示已存在
        """
        result = self.collection.get(ids=[id])
        return len(result['ids']) > 0

    def delete(self, ids):
        """
        批量删除 embedding 记录

        用于清理过期数据或 PDF 文件被删除后同步向量库。

        Args:
            ids: 要删除的页面 ID 列表
        """
        self.collection.delete(ids=ids)

    def count(self):
        """
        获取当前集合中的总记录数

        Returns:
            已索引的页面总数
        """
        return self.collection.count()


class PgvectorVectorStore(VectorStore):
    """
    TODO: PostgreSQL + pgvector 实现（预留接口，大规模部署时使用）

    迁移时机：
    - ChromaDB 中向量数量超过 50 万条
    - 需要多进程/多服务并发访问同一向量库
    - 已有 PostgreSQL 基础设施需要统一管理

    迁移步骤（未来实施）：
    1. 安装 pgvector 扩展：CREATE EXTENSION vector;
    2. 创建表结构（page_id, embedding vector(1024), metadata jsonb）
    3. 实现 PgvectorVectorStore 的所有抽象方法
    4. 设置 VECTOR_STORE_BACKEND="pgvector" 并配置 PG_HOST 等连接参数
    5. 运行数据迁移脚本 scripts/migrate_chromadb_to_pgvector.py

    注意：上层业务代码（PageFilter、PageIndexer）无需任何改动，
    因为它们只依赖 VectorStore 抽象接口。
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("pgvector 后端尚未实现")

    def add_embeddings(self, ids, embeddings, metadatas):
        raise NotImplementedError

    def search(self, query_embedding, top_k=20, where=None):
        raise NotImplementedError

    def get_embedding(self, id):
        raise NotImplementedError

    def exists(self, id):
        raise NotImplementedError

    def delete(self, ids):
        raise NotImplementedError

    def count(self):
        raise NotImplementedError
