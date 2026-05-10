# 提取审查功能 — 设计文档

**日期**：2026-05-10
**状态**：待实施

---

## 问题定义

| 问题 | 含义 | 检测方式 |
|------|------|---------|
| **稀疏** | 某条记录大部分字段为空，只有 1-2 个字段有值，但 LLM 仍然把它作为一条结果返回了 | 计算每条记录的字段填充率，低于阈值则标记/删除 |
| **重复** | 两条记录完全一样（所有非空字段值相同）；或记录 B 的所有非空字段的值都在记录 A 中出现（B 是 A 的子集），保留 A | 逐对比较所有非空字段 |
| **不准确** | 提取的值是否正确？这不能靠 LLM 自己判断。解决办法：每条结果记录它在 PDF 中的来源位置，提供 PDF 预览 API 让用户/agent 对照原文核实 | 位置追踪 + 预览接口 |

---

## Part 1: 稀疏 + 重复检测器（确定性规则，无需 LLM）

### 新增文件: `extract/quality_checker.py`

```python
class QualityChecker:
    """
    确定性规则的质量检测器。不调用 LLM，可放心在生产环境自动删除。
    
    依赖: 无外部依赖，纯 Python 逻辑。
    """
    
    @staticmethod
    def field_fill_rate(record: dict, fields: list[str]) -> float:
        """计算字段填充率: 非空字段数 / 总字段数
        
        "空" 的定义:
          - None
          - 空字符串 ""
          - 仅含空白字符的字符串
          - 字符串 "无"、"未提及"、"N/A"
        """
        non_empty = 0
        for f in fields:
            val = record.get(f)
            if val and str(val).strip() and str(val).strip() not in ("无", "未提及", "N/A", "-", "--"):
                non_empty += 1
        return non_empty / len(fields) if fields else 0.0
    
    @staticmethod
    def check_sparsity(records: list[dict], fields: list[str], 
                       threshold: float = 0.3) -> list[int]:
        """检测稀疏记录，返回应删除的索引列表
        
        阈值 0.3 表示：10 个字段中少于 3 个有值 → 标记为稀疏
        """
        deleted = []
        for i, record in enumerate(records):
            rate = QualityChecker.field_fill_rate(record, fields)
            if rate < threshold:
                deleted.append(i)
        return deleted
    
    @staticmethod
    def records_equal(record_a: dict, record_b: dict, fields: list[str]) -> bool:
        """两条记录在所有非空字段上是否完全一致"""
        for f in fields:
            va = str(record_a.get(f, "")).strip()
            vb = str(record_b.get(f, "")).strip()
            if va and vb and va != vb:
                return False
        return True
    
    @staticmethod
    def record_contains(record_a: dict, record_b: dict, fields: list[str]) -> bool:
        """记录 A 是否包含记录 B（A 在每个非空字段上 ≥ B）
        
        即 B 的所有非空字段值都在 A 中出现且完全一致，且 A 至少比 B 多一个字段有值。
        这意味着 A 的信息量大于等于 B，保留 A，删除 B。
        """
        a_richer = False
        for f in fields:
            va = str(record_a.get(f, "")).strip()
            vb = str(record_b.get(f, "")).strip()
            if vb and not va:
                return False   # B 有值但 A 没有 → A 不包含 B
            if va and not vb:
                a_richer = True  # A 有值但 B 没有 → A 更丰富
            if va and vb and va != vb:
                return False   # 都有值但不同 → 不是包含关系
        return a_richer  # 至少有一个字段 A 比 B 多
    
    @staticmethod
    def check_duplicates(records: list[dict], fields: list[str]) -> list[int]:
        """检测重复记录，返回应删除的索引列表
        
        规则:
          1. 两条记录完全一致 → 删除后出现的
          2. A 包含 B（A 是 B 的超集）→ 删除 B
        """
        n = len(records)
        deleted = set()
        
        for i in range(n):
            if i in deleted:
                continue
            for j in range(i + 1, n):
                if j in deleted:
                    continue
                
                if QualityChecker.records_equal(records[i], records[j], fields):
                    # 完全一致 → 删除 j
                    deleted.add(j)
                elif QualityChecker.record_contains(records[i], records[j], fields):
                    # i 包含 j → 删除 j
                    deleted.add(j)
                elif QualityChecker.record_contains(records[j], records[i], fields):
                    # j 包含 i → 删除 i
                    deleted.add(i)
                    break  # i 被删了，不用继续跟其他 j 比
        
        return sorted(deleted)
    
    @staticmethod
    def run_all_checks(records: list[dict], fields: list[str],
                       sparse_threshold: float = 0.3) -> dict:
        """运行全部检查，返回报告"""
        sparse_deleted = QualityChecker.check_sparsity(records, fields, sparse_threshold)
        
        # 先排除稀疏记录，再做重复检测
        remaining = [r for i, r in enumerate(records) if i not in sparse_deleted]
        dup_deleted_in_remaining = QualityChecker.check_duplicates(remaining, fields)
        
        # 将剩余数组的索引映射回原始数组的索引
        remaining_indices = [i for i in range(len(records)) if i not in sparse_deleted]
        dup_deleted_original = [remaining_indices[i] for i in dup_deleted_in_remaining]
        
        return {
            "sparse_deleted": sparse_deleted,
            "duplicate_deleted": dup_deleted_original,
            "total_deleted": len(sparse_deleted) + len(dup_deleted_original),
            "sparse_rate": {
                i: QualityChecker.field_fill_rate(records[i], fields)
                for i in sparse_deleted
            },
        }
```

### 集成方式

在 `extraction_engine.py` 的 `process_pdf_library()` 中，`_save_extraction_results()` 之前插入：

```python
# 质量检查（不调用 LLM，纯规则判断）
from extract.quality_checker import QualityChecker

qc_result = QualityChecker.run_all_checks(all_extracted_data, fields)

if qc_result["sparse_deleted"] or qc_result["duplicate_deleted"]:
    all_extracted_data = [
        r for i, r in enumerate(all_extracted_data)
        if i not in qc_result["sparse_deleted"] 
        and i not in qc_result["duplicate_deleted"]
    ]
    self.task_manager.put_message("info",
        f"质量检查: 删除 {len(qc_result['sparse_deleted'])} 条稀疏记录, "
        f"{len(qc_result['duplicate_deleted'])} 条重复记录"
    )
```

### 配置项

```python
# core/config.py
QUALITY_CHECK_ENABLED: bool = True
QUALITY_SPARSE_THRESHOLD: float = 0.3    # 字段填充率低于此值视为稀疏
QUALITY_AUTO_DELETE: bool = True         # 是否自动删除（稀疏+重复检测是确定性的，可放心开）
```

---

## Part 2: 提取结果位置追踪

### 目标

每条提取结果记录以下来源信息，让用户/agent 能回溯到 PDF 原文：

```
{
    "钝化剂名称": "PEAI",
    "PCE(%)": "22.1",
    ...
    "_source_doc": "nature_2024.pdf",        // 已有
    "_source_page": 5,                        // 新增：PDF 页码（从 1 开始）
}
```

### 实现方式

当前 `_process_with_vision()` 和 `_process_with_text()` 已经知道每个结果来自哪个 `pdf_path` 和 `page_num`。只需在写入 `_source_doc` 的同时加一行 `_source_page`。

修改位置：`extract/extraction_engine.py`

- `_process_with_vision()` 第 428 行附近，`item_dict['_source_doc'] = doc_id` 之后加一行：
  ```python
  item_dict['_source_page'] = page_num + 1
  ```

- `_process_with_text()` 第 496 行附近，同样位置加一行：
  ```python
  item_dict['_source_page'] = page_num + 1
  ```

改动量：两行，无破坏性。

---

## Part 3: PDF 预览 API

### 目标

给定 PDF 路径 + 页码 + 可选高亮关键词，返回页面图片 + 文本，让前端或 agent 能对照原文核验提取结果。

### 新增路由

```python
# app.py

@app.route('/api/page_preview', methods=['GET'])
def page_preview():
    """
    参数:
      doc:  PDF 文件名或路径
      page: 页码（从 1 开始）
      query: 可选，要高亮的关键词（逗号分隔）
    
    返回 JSON:
      {
        "doc": "nature_2024.pdf",
        "page": 5,
        "total_pages": 12,
        "image_base64": "data:image/jpeg;base64,...",
        "text": "页面文本内容...",
        "highlights": [        // 如果传了 query
          {"keyword": "PEAI", "line": 12, "context": "...PEAI was used as..."},
          ...
        ]
      }
    """
    doc = request.args.get("doc", "")
    page = int(request.args.get("page", 1))
    query = request.args.get("query", "")
    
    pdf_path = _resolve_pdf_path(doc)
    if not pdf_path or not os.path.isfile(pdf_path):
        return jsonify({"error": "文档不存在"}), 404
    
    # 提取页面内容
    page_content = pdf_processor.extract_page_content(pdf_path, page - 1)
    
    # 生成图像
    img_base64 = pdf_processor.render_page_image(pdf_path, page - 1)
    
    # 搜索高亮位置
    highlights = []
    if query:
        keywords = [k.strip() for k in query.split(",") if k.strip()]
        text_lines = page_content.get("markdown_text", "").split("\n")
        for kw in keywords:
            for line_idx, line in enumerate(text_lines):
                if kw in line:
                    highlights.append({
                        "keyword": kw,
                        "line": line_idx + 1,
                        "context": line.strip()[:200],
                    })
    
    return jsonify({
        "doc": doc,
        "page": page,
        "total_pages": page_content.get("total_pages", 0),
        "image_base64": f"data:image/jpeg;base64,{img_base64}",
        "text": page_content.get("markdown_text", ""),
        "highlights": highlights,
    })
```

### Agent 调用接口

除了 `GET /api/page_preview`，还提供 `POST /api/page_context` 供 agent 批量阅读：

```python
@app.route('/api/page_context', methods=['POST'])
def page_context():
    """
    供 agent 读取提取结果所在页面的上下文
    
    请求体:
      {
        "results": [
          {"doc": "nature_2024.pdf", "page": 5, "query": "PEAI"},
          {"doc": "nature_2024.pdf", "page": 7, "query": "22.1%"}
        ]
      }
    
    返回:
      {
        "contexts": [
          {"doc": "...", "page": 5, "text": "...", "matches": [...]},
          ...
        ]
      }
    """
    data = request.get_json()
    contexts = []
    
    for item in data.get("results", []):
        pdf_path = _resolve_pdf_path(item["doc"])
        page_num = item["page"] - 1
        query = item.get("query", "")
        
        content = pdf_processor.extract_page_content(pdf_path, page_num)
        text = content.get("markdown_text", "")
        
        # 在文本中查找关键词位置
        matches = []
        if query:
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if query in line:
                    matches.append({"line": i + 1, "text": line.strip()[:300]})
        
        contexts.append({
            "doc": item["doc"],
            "page": item["page"],
            "text": text[:3000],        # 截断到 3000 字符
            "matches": matches[:10],    # 最多 10 条匹配
        })
    
    return jsonify({"contexts": contexts})
```

---

## Part 4: 前端 PDF 预览（概要设计）

### 交互流程

```
提取结果表格（CSV 展示）
  │
  ├── 每条结果行旁有一个 "来源" 按钮 / 图标
  │    点击 →
  │      1. 读取该行的 _source_doc + _source_page
  │      2. 调用 GET /api/page_preview?doc=...&page=...
  │      3. 打开 PDF 预览面板，展示页面图片
  │      4. 根据返回的 highlights，在图片下方/侧边标注匹配行
  │
  └── 问题标记：用户发现不准确的条目 →
        在 UI 中标记 + 记录 → 反馈给 prompt 优化流程
```

### UI 布局

```
┌──────────────────────┬──────────────────────┐
│   提取结果表格        │   PDF 预览面板        │
│                      │   (默认隐藏/可收起)    │
│  ┌───────────────┐  │                      │
│  │ PEAI │ 22.1%  │  │  ┌────────────────┐  │
│  │ [查看来源 ▸]  │  │  │  页面图片       │  │
│  └───────────────┘  │  │  (高亮行标记)    │  │
│                      │  └────────────────┘  │
│  ┌───────────────┐  │                      │
│  │ FAI  │ 20.3%  │  │  匹配文本:           │
│  │ [查看来源 ▸]  │  │  L12: PEAI was used  │
│  └───────────────┘  │  L28: PEAI exhibited │
│                      │                      │
└──────────────────────┴──────────────────────┘
```

前端实现不做详细展开，核心是上述 API 支撑。

---

## 整体架构

```
                       PDF pages
                          │
               ┌──────────▼──────────┐
               │   LLM 提取           │
               │   (已有流程)          │
               │   每页生成多条记录    │
               └──────────┬──────────┘
                          │
               all_extracted_data (每条带 _source_doc + _source_page)
                          │
               ┌──────────▼──────────┐
               │   QualityChecker     │  ← 新增：确定性规则
               │   - 稀疏检测          │
               │   - 重复检测          │
               └──────────┬──────────┘
                          │
               all_extracted_data (已剔除稀疏 + 重复)
                          │
               ┌──────────▼──────────┐
               │   dedup (已有)       │  ← 字符串级去重
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │   写入 CSV           │
               │   (含 _source_page)  │
               └─────────────────────┘

   用户/agent 核验不准确:
   ┌─────────────────────────────────┐
   │  点击结果 → /api/page_preview   │
   │  → 展示 PDF 原文 + 高亮          │
   │  → agent 调 /api/page_context   │
   │  → 对照原文判断准确性            │
   └─────────────────────────────────┘
```

---

## 实施步骤

| 步骤 | 文件 | 内容 | 依赖 |
|------|------|------|------|
| 1 | `extract/quality_checker.py` | QualityChecker 类（稀疏+重复检测） | 无 |
| 2 | `extraction_engine.py` | 两处加 `_source_page` | 无 |
| 3 | `extraction_engine.py` | 集成 QualityChecker 调用 | 步骤 1 |
| 4 | `app.py` | `/api/page_preview` + `/api/page_context` 路由 | 步骤 2 |
| 5 | `core/config.py` | 新增 `QUALITY_CHECK_*` 配置项 | 无 |
| 6 | `platform_init/test/prompt/test_quality_checker.py` | QualityChecker 单元测试 | 步骤 1 |
| 7 | `frontend/` (Vue) | 结果行的来源按钮 + PDF 预览面板 | 步骤 4 |

步骤 1、2、5、6 可并行。
步骤 3 依赖 1。
步骤 4 依赖 2。
步骤 7 依赖 4。
