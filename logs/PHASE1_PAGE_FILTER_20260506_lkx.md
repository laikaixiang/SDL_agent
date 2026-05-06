# Phase 1 页面预筛选实现日志

**日期**: 2026-05-06
**作者**: lkx
**版本**: v1.0

---

## 1. 概述

实现 RAG 增强文献提取方案 Phase 1 —— 基于多模态 embedding 的 PDF 页面预筛选。

目标是减少 LLM 调用次数：仅将与提取任务语义相关的页面发送给 LLM，跳过无关页面（参考文献、致谢等），节省 Token 消耗和处理时间。

---

## 2. 新增文件

| 文件 | 行数（约） | 说明 |
|------|-----------|------|
| `core/embedding_service.py` | 250 | Embedding 服务抽象层：ABC 基类 + JinaEmbeddingService（Jina AI 云端 API）+ LocalEmbeddingService 占位 + 工厂函数 |
| `core/vector_store.py` | 270 | 向量存储抽象层：ABC 基类 + ChromaVectorStore（ChromaDB 持久化 + 余弦距离）+ PgvectorVectorStore 占位 |
| `core/page_indexer.py` | 200 | PDF 页面预索引：make_page_id() / compute_content_hash() + PageIndexer（SQLite 元数据 + 内容 hash 去重） |
| `core/page_filter.py` | 150 | 页面预筛选：PageFilter.set_task() 缓存任务向量 + should_process() 逐页余弦相似度比较 |

## 3. 修改文件

| 文件 | 变更内容 |
|------|----------|
| `core/config.py` | 新增 16 个配置项：Embedding（6）、VectorStore（2）、PageFilter（3）、FewShot（2 flag）、SemanticSearch（1 flag） |
| `core/extraction_engine.py` | 新增 _init_page_filter_services() 方法、process_pdf_library() 增加预索引步骤、_process_single_pdf() 页面循环中插入 page_filter 检查 |
| `requirements.txt` | 添加 `chromadb` 依赖 |

---

## 4. 架构设计

```
ExtractionEngine.process_pdf_library()
    │
    ├── _init_page_filter_services()   ← 按需初始化（优雅降级）
    │       ├── create_embedding_service()  → JinaEmbeddingService
    │       ├── ChromaVectorStore(persist_dir)
    │       ├── PageIndexer(embedding, vector_store, sqlite)  ← 一次性预索引
    │       └── PageFilter(embedding, vector_store, threshold=0.3)
    │
    ├── page_indexer.index_all_pdfs()  ← 增量索引（跳过未变更页面）
    │
    ├── page_filter.set_task(task_description)  ← 缓存任务向量
    │
    └── for each PDF → for each page:
            if page_filter.should_process(pdf_path, page_num):
                _process_single_page()  → LLM 调用
            else:
                skip  ← 节省 Token
```

### 关键设计决策

- **延迟初始化**: page_filter / page_indexer 在 process_pdf_library 中按需创建，不影响 ExtractionEngine 的正常实例化
- **优雅降级**: 任何初始化步骤失败（API key 缺失、ChromaDB 不可写等）→ page_filter = None，后续不进行筛选，正常处理所有页面
- **保守阈值**: 默认 0.3，宁可多处理也不漏数据；用户可根据效果调整到 0.5~0.7
- **增量索引**: 通过 content_hash 检测页面变化，未变更页面跳过重复索引

---

## 5. 实测结果

### 测试环境
- API: Jina AI `jina-clip-v2`
- Embedding 维度: 1024
- 相似度度量: 余弦相似度
- 阈值: 0.3

### 测试数据

| 页面类型 | 内容 | 相似度 | 判定 |
|----------|------|--------|------|
| 相关（制备） | 使用反溶剂法在DMF中制备FAPbI3钙钛矿薄膜... | 0.6388 | PASS |
| 相关（结果） | 实验结果表明PbI2与FAI的摩尔比为1:1时器件效率最高... | 0.4667 | PASS |
| 边界（引用） | 参考文献 [1] Smith et al. Nature Materials 2019... | 0.3297 | PASS |
| 无关（致谢） | 致谢：本研究得到国家自然科学基金资助... | 0.2624 | SKIP |

查询任务：**"提取FAPbI3钙钛矿太阳能电池的制备参数"**

### 结论

- 阈值 0.3 能有效区分完全无关内容（致谢），但参考文献类内容可能刚好过线（含少量领域术语）
- 建议根据实际场景微调阈值到 0.35 以避免过多参考文献页被送入 LLM
- 最相关的制备参数页面得分 0.64，远高于无关页面，区分度明显

---

## 6. 后续工作

### 调优建议
1. 使用几轮实际提取任务，收集「正确项/误筛项」数据，确定最优阈值
2. 可选：实现 image-only embedding（仅用页面截图，不依赖文本提取质量）

### Phase 2 预留
- `FEW_SHOT_ENABLED` / `FEW_SHOT_TOP_K` 配置已加，代码未实现
- 计划在 LLM 调用时从历史提取结果中检索相似案例作为 few-shot 示例

### Phase 3 预留
- `SEMANTIC_SEARCH_ENABLED` 配置已加，代码未实现
- 计划提供 `/api/semantic_search` 路由供前端语义搜索

---

## 7. 推荐 Embedding 替代方案

### 方案 A: SiliconFlow BGE-M3（推荐，零额外成本）
- 端点: `https://api.siliconflow.cn/v1/embeddings`
- 模型: `BAAI/bge-m3`
- 优势: 与现有 LLM API key 共用，中文优化，MIT 开源
- 实现难度: 低（仅需新增 SiliconFlowEmbeddingService 类）

### 方案 B: DeepSeek Embedding
- 端点: `https://api.deepseek.com/v1/embeddings`
- 模型: `deepseek-embedding-v1`
- 优势: 超低价格（~$0.0002/1K tokens），中文优化
- 劣势: 需单独申请 API key，纯文本不支持多模态

### 方案 C: 本地部署（零费用，最大隐私）
- 推荐模型: `BAAI/bge-m3` (MIT, ~2GB) 或 `Qwen/Qwen3-Embedding-0.6B` (Apache 2.0)
- 使用 sentence-transformers 或 FastEmbed 加载
- 优势: 无 API 费用，数据不出本地
- 劣势: 需 GPU（推荐）或 CPU 推理，首次加载较慢

---

## 8. 文件变更清单

```
新建:
  core/embedding_service.py
  core/vector_store.py
  core/page_indexer.py
  core/page_filter.py
  logs/PHASE1_PAGE_FILTER_20260506_lkx.md

修改:
  core/config.py          (+28 行)
  core/extraction_engine.py  (+45 行)
  requirements.txt         (+1 行)
```
