# 数据分析模式优化总结

## 问题分析

1. **API URL 重复拼接问题**：`software/auto_analyze.py` 和 `prompt_template.py` 中将 `Config.API_URL` 再次拼接 `/chat/completions`，导致 404 错误
2. **文件查找失败**：只检查指定路径，没有智能搜索功能
3. **缺少容错机制**：读取失败后直接放弃，没有尝试其他方法
4. **算法生成器入口不明显**：虽然有实现但调用不方便

## 优化方案

### 1. 修复 API URL 配置 ✅

**修改文件**：
- `core/config.py`：添加注释说明 API_URL 已包含完整路径
- `software/auto_analyze.py`：移除重复拼接 `/chat/completions`
- `software/algorithms/extra_algorithms_fromProjects/prompt_template.py`：移除重复拼接

**关键代码**：
```python
# core/config.py
API_URL: str = "https://api.longcat.chat/openai/v1/chat/completions"
# 注意：API_URL 是完整的 endpoint，已包含 /chat/completions 路径
# 直接使用即可，不需要再拼接 /chat/completions

# software/auto_analyze.py 和 prompt_template.py
_API_URL = Config.API_URL  # 已包含完整路径，不需要拼接
```

### 2. 智能文件查找 ✅

**修改文件**：`software/auto_analyze.py`

**新增功能**：
- 添加 `_find_csv_file()` 函数，支持多目录搜索
- 搜索顺序：直接路径 → temporal/ → results/ → extract/ → 当前目录
- 自动匹配 `extraction.csv` 作为默认文件
- 提供详细的错误信息，告知用户搜索了哪些目录

**关键代码**：
```python
def _find_csv_file(csv_path: str) -> str:
    """智能查找 CSV 文件，支持多种路径格式和自动搜索"""
    # 1. 直接路径存在
    if os.path.exists(csv_path):
        return os.path.abspath(csv_path)
    
    # 2. 在常见目录中查找
    search_dirs = ["temporal", "results", "extract", "."]
    for search_dir in search_dirs:
        candidate = os.path.join(search_dir, os.path.basename(csv_path))
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    
    # 3. 优先匹配 extraction.csv
    if os.path.exists("temporal"):
        csv_files = [f for f in os.listdir("temporal") if f.endswith('.csv')]
        if 'extraction.csv' in csv_files:
            return os.path.abspath(os.path.join("temporal", 'extraction.csv'))
```

### 3. 容错和重试机制 ✅

**修改文件**：`software/auto_analyze.py`

**新增功能**：
- 添加 `_try_alternative_readers()` 函数
- 当指定的读取函数失败时，自动尝试备用读取方法
- 备用顺序：`read_numeric_columns` → `read_as_columns_dict`
- 提供详细的错误信息和重试日志

**关键代码**：
```python
def _try_alternative_readers(csv_path: str, failed_reader: str, read_params: dict):
    """当指定的读取函数失败时，尝试其他读取函数"""
    fallback_order = [
        ("read_numeric_columns", {}),
        ("read_as_columns_dict", {}),
    ]
    for reader_name, params in fallback_order:
        if reader_name == failed_reader:
            continue
        try:
            reader_fn = READER_REGISTRY[reader_name]
            data = reader_fn(csv_path, **params)
            if data:
                return data, reader_name
        except Exception:
            continue
    return None, None
```

### 4. 添加算法生成器接口 ✅

**修改文件**：
- `software/software_controller.py`：添加 `generate_algorithm()` 方法
- `app.py`：添加 `handle_generate_algorithm()` 处理函数
- `templates/index.html`：添加数据分析子菜单

**新增功能**：
- 在 SoftwareController 中直接调用算法生成器
- 生成成功后自动重新加载算法注册表
- 前端添加"数据分析模式"子菜单：
  - 📊 分析数据（使用现有算法）
  - 🔧 生成新算法

**关键代码**：
```python
# software/software_controller.py
def generate_algorithm(self, user_description: str) -> dict:
    """使用 LLM 根据用户自然语言描述自动生成新算法"""
    result = _generate(user_description, verbose=False)
    if result.get("success"):
        # 重新加载算法
        self._registry.clear()
        self._discover_algorithms()
    return result

# app.py
def handle_generate_algorithm(user_message: str) -> Response:
    """处理算法生成请求"""
    description = user_message.replace("生成算法：", "").strip()
    result = software_manager.generate_algorithm(description)
    return jsonify({'type': 'system', 'reply': reply})
```

### 5. 文件选择界面 ✅

**修改文件**：`templates/index.html`

**新增功能**：
- 添加 CSV 文件选择按钮（📂）
- 仅在"分析数据"模式下显示
- 支持点击选择文件，自动填充路径
- 优先使用选择的文件路径

**UI 改进**：
```html
<!-- 文件选择按钮 -->
<label for="csv-file-input" id="csv-file-btn" class="tool-btn" 
       style="display:none;" title="选择CSV文件">📂</label>
<input type="file" id="csv-file-input" accept=".csv" style="display:none;">
```

**交互逻辑**：
```javascript
// 数据分析子菜单
<div class="mode-menu" id="analyze-submenu">
    <div class="mode-item" onclick="setMode('analyze', '数据分析：', '📊 分析数据')">
        📊 分析数据
    </div>
    <div class="mode-item" onclick="setMode('generate_algo', '生成算法：', '🔧 生成算法')">
        🔧 生成新算法
    </div>
</div>
```

## 使用说明

### 数据分析模式

1. **分析现有数据**：
   - 点击左下角 `+` → 📊 数据分析模式 → 📊 分析数据
   - 点击 📂 选择 CSV 文件，或直接输入文件路径
   - 系统会智能查找文件并自动选择合适的算法
   - 支持容错重试，读取失败会自动尝试其他方法

2. **生成新算法**：
   - 点击左下角 `+` → 📊 数据分析模式 → 🔧 生成新算法
   - 用自然语言描述算法功能，例如：
     ```
     对数值列表做移动平均，窗口大小可配置，默认5
     ```
   - 系统会自动生成算法代码并注册
   - 生成后可立即在"分析数据"中使用

### 文件查找优先级

1. 用户指定的完整路径
2. `temporal/` 目录
3. `results/` 目录
4. `extract/` 目录
5. 当前目录
6. 自动匹配 `temporal/extraction.csv`

## 测试建议

1. **测试 API 修复**：
   ```bash
   python software/algorithms/extra_algorithms_fromProjects/prompt_template.py
   ```

2. **测试文件查找**：
   - 只输入文件名（如 `extraction.csv`）
   - 输入相对路径（如 `temporal/extraction.csv`）
   - 输入不存在的文件，查看错误提示

3. **测试算法生成**：
   - 在前端选择"生成新算法"模式
   - 输入算法描述并提交
   - 检查生成的算法文件和注册状态

4. **测试容错机制**：
   - 使用格式不标准的 CSV 文件
   - 观察系统是否自动尝试其他读取方法

## 注意事项

1. API_URL 配置已修正，不要再手动拼接 `/chat/completions`
2. 文件查找会自动搜索多个目录，无需提供完整路径
3. 算法生成后会自动重新加载，无需重启服务
4. 所有优化都保持向后兼容，不影响现有功能
