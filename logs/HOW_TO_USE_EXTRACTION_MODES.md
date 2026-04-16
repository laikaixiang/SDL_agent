# PDF提取模式调用方式

## 三种调用方式

### 方式1：全局配置（最简单，推荐）

直接修改配置文件，所有提取任务都使用这个模式：

```python
# 编辑 core/config.py 第75行
EXTRACTION_MODE: str = "hybrid"  # 可选: "vision", "text", "hybrid"
```

然后正常使用提取功能：
```
帮我搜寻：FAPbI3钙钛矿的钝化剂
```

系统会自动应用配置的模式。

---

### 方式2：通过Web界面切换（最直观）

1. **启动Flask应用**：
   ```bash
   python app.py
   ```

2. **访问模式设置页面**：
   ```
   http://127.0.0.1:5000/extraction_mode
   ```

3. **点击按钮切换模式**：
   - 🎯 混合模式 (推荐) - 智能判断，节省60-80%成本
   - 🖼️ 纯视觉模式 - 准确但成本高
   - 📝 纯文本模式 - 快速便宜但可能丢失图表

4. **切换后立即生效**，下次提取任务会使用新模式

---

### 方式3：通过API调用（适合程序化控制）

#### 获取当前模式
```bash
curl http://127.0.0.1:5000/api/extraction_mode
```

返回：
```json
{
  "mode": "hybrid",
  "available_modes": {
    "vision": "纯视觉模式（准确但贵）",
    "text": "纯文本模式（快速便宜）",
    "hybrid": "混合模式（推荐）"
  }
}
```

#### 切换模式
```bash
# 切换到混合模式
curl -X POST http://127.0.0.1:5000/api/extraction_mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "hybrid"}'

# 切换到视觉模式
curl -X POST http://127.0.0.1:5000/api/extraction_mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "vision"}'

# 切换到文本模式
curl -X POST http://127.0.0.1:5000/api/extraction_mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "text"}'
```

返回：
```json
{
  "success": true,
  "mode": "hybrid",
  "message": "已切换到 混合模式"
}
```

#### Python代码示例
```python
import requests

# 切换到混合模式
response = requests.post(
    'http://127.0.0.1:5000/api/extraction_mode',
    json={'mode': 'hybrid'}
)
print(response.json())
# {'success': True, 'mode': 'hybrid', 'message': '已切换到 混合模式'}
```

---

## 三种模式详解

### 🎯 Hybrid模式（推荐）

**工作流程：**
1. 提取PDF页面文本为Markdown
2. 检测是否包含复杂内容（化学式、图表、实验数据）
3. 复杂页面 → Vision API
4. 简单页面 → 文本API

**适用场景：**
- 大部分科学文献提取
- 成本和准确性都重要的场景
- 不确定PDF内容复杂度时

**成本：** 约38%（相比纯Vision）

---

### 🖼️ Vision模式

**工作流程：**
- 所有页面都转成图片
- 使用Vision API分析

**适用场景：**
- 包含大量图表、化学结构式的文献
- 扫描版PDF
- 对准确性要求极高的场景

**成本：** 100%（基准）

---

### 📝 Text模式

**工作流程：**
- 提取PDF文本为Markdown
- 使用文本API分析

**适用场景：**
- 纯文字描述的文献（如综述、理论部分）
- 成本敏感的场景
- 文本质量好的PDF

**成本：** 约20%（相比纯Vision）

**注意：** 会丢失图表和化学结构式信息

---

## 快速测试

### 测试脚本
```bash
python test_hybrid_simple.py
```

### 测试输出示例
```
测试 1: 提取PDF文本为Markdown
✓ 成功提取文本
✓ 文本长度: 3109 字符

测试 2: 检测复杂内容
✓ 检测到复杂内容: False
  → 原因: 纯文字描述，无复杂内容
  → 建议: 使用文本API节省成本
```

---

## 实际使用示例

### 场景1：日常文献提取（推荐hybrid）

```python
# core/config.py
EXTRACTION_MODE = "hybrid"
```

在Web界面输入：
```
帮我搜寻：FAPbI3钙钛矿的钝化剂
```

系统输出：
```
📄 使用文本模式处理第 1 页（节省成本）
🖼️ 使用视觉模式处理第 3 页（包含图表）
📄 使用文本模式处理第 4 页（节省成本）
...
```

### 场景2：高精度提取（使用vision）

```python
# core/config.py
EXTRACTION_MODE = "vision"
```

所有页面都用Vision API，保证最高准确性。

### 场景3：快速筛选（使用text）

```python
# core/config.py
EXTRACTION_MODE = "text"
```

快速提取大量文献的文字信息，用于初步筛选。

---

## 成本对比

假设一篇10页的钙钛矿文献：

| 模式 | Vision页数 | Text页数 | 相对成本 | 准确性 |
|------|-----------|---------|---------|--------|
| Vision | 10 | 0 | 100% | ⭐⭐⭐⭐⭐ |
| Hybrid | 3 | 7 | 38% | ⭐⭐⭐⭐ |
| Text | 0 | 10 | 20% | ⭐⭐⭐ |

**推荐：** 使用Hybrid模式，在成本和准确性之间取得最佳平衡。

---

## 常见问题

**Q: 切换模式后需要重启服务吗？**
A: 不需要，通过API或Web界面切换后立即生效。

**Q: 如何知道某一页用的是哪种模式？**
A: Hybrid模式下，系统会在任务进度中显示：
- "📄 使用文本模式处理第 X 页（节省成本）"
- "🖼️ 使用视觉模式处理第 X 页"

**Q: 可以针对不同任务使用不同模式吗？**
A: 目前是全局配置，但可以在提取前通过API切换模式。

**Q: Hybrid模式的检测准确吗？**
A: 检测规则针对科学文献优化，会识别化学式、图表引用、实验关键词等。如果不确定，建议使用Vision模式。

---

## 相关文件

- `core/config.py` - 配置文件
- `core/pdf_to_markdown.py` - PDF转Markdown核心功能
- `core/pdf_processor.py` - PDF处理器
- `core/extraction_engine.py` - 提取引擎
- `templates/extraction_mode.html` - Web设置页面
- `test_hybrid_simple.py` - 测试脚本
- `HYBRID_EXTRACTION_GUIDE.md` - 详细技术文档
