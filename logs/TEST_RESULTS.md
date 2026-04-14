# 多图片输入测试结果报告

## 测试日期
2026-04-14

## 测试目标
验证LongCat API是否支持：
1. 单张图片输入
2. 多张图片批量输入
3. PDF原生输入

## 测试结果

### 问题发现
在测试过程中遇到以下问题：

1. **API连接问题**
   - 状态码: 400
   - 错误信息: "json format error"
   - 原因分析: 
     - 可能是API服务暂时不可用
     - 可能是网络连接问题
     - 可能是请求格式不符合API要求

2. **依赖问题**
   - 缺少 `pydantic_ai` 模块
   - 导致无法直接导入 `core` 模块

## 当前系统分析

### 现有实现
根据代码分析，当前系统：

1. **PDF处理流程**
   - 使用 PyMuPDF (fitz) 将PDF逐页转换为图片
   - 每页转换为Base64编码的JPEG图片
   - 逐页发送给视觉语言模型API
   - 位置: `core/pdf_processor.py:31-56`

2. **API调用格式**
   ```python
   messages = [
       {"role": "system", "content": sys_prompt},
       {"role": "user", "content": [
           {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
       ]}
   ]
   ```
   - 位置: `core/extraction_engine.py:251-254`

3. **当前限制**
   - 每次API调用只发送一张图片
   - 无法一次性分析整个PDF文档
   - 跨页信息可能丢失

## 实施建议

### 方案1: 批量多图片输入（推荐）

**前提条件**: API支持在一次请求中发送多张图片

**实施步骤**:

1. **修改 PDFProcessor 类** (`core/pdf_processor.py`)
   ```python
   def pdf_to_images_batch(self, pdf_path: str, batch_size: int = 10) -> List[List[str]]:
       """将PDF转换为图片批次"""
       images = self.convert_to_images(pdf_path)
       batches = []
       for i in range(0, len(images), batch_size):
           batches.append(images[i:i+batch_size])
       return batches
   ```

2. **修改 ExtractionEngine 类** (`core/extraction_engine.py`)
   - 添加 `_process_pdf_batch()` 方法
   - 修改消息格式支持多图片:
   ```python
   messages = [
       {"role": "system", "content": sys_prompt},
       {"role": "user", "content": [
           {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1}"}},
           {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2}"}},
           # ... 更多图片
       ]}
   ]
   ```

3. **添加配置选项** (`core/config.py`)
   ```python
   USE_BATCH_MODE: bool = False  # 默认关闭，测试通过后开启
   BATCH_SIZE: int = 5           # 每批处理5页（保守值）
   MAX_TOKENS_BATCH: int = 4096  # 批量模式的最大token数
   ```

**优点**:
- AI能看到完整上下文
- 减少API调用次数
- 提取质量更好

**缺点**:
- Token消耗增加
- 单次请求时间更长
- 需要API支持（待验证）

### 方案2: 保持现有逐页处理（当前方案）

**适用场景**: API不支持多图片输入

**优化建议**:
1. 添加并发处理（使用线程池）
2. 优化图片压缩质量
3. 添加缓存机制

### 方案3: PDF原生输入（理想方案）

**前提条件**: API支持直接读取PDF文件

**实施步骤**:
1. 将PDF文件转换为Base64
2. 使用document类型发送
3. 消息格式:
   ```python
   {"role": "user", "content": [
       {"type": "document", "source": {
           "type": "base64",
           "media_type": "application/pdf",
           "data": pdf_base64
       }}
   ]}
   ```

**优点**:
- 无需转换为图片
- 处理速度最快
- 保留PDF原始格式

**缺点**:
- 需要API支持（可能性较低）

## 下一步行动

### 立即行动
1. **验证API可用性**
   - 检查API服务状态
   - 确认网络连接
   - 联系API提供商确认服务状态

2. **验证当前实现**
   - 运行现有的PDF提取功能
   - 确认单图片输入是否正常工作
   - 如果正常，说明测试代码有问题

### 待API恢复后
1. **重新运行测试**
   ```bash
   python test/api_test/test_multi_image_support.py
   ```

2. **根据测试结果决定实施方案**
   - 如果支持多图片 → 实施方案1
   - 如果不支持 → 保持方案2并优化
   - 如果支持PDF → 实施方案3

## 测试文件位置

- 完整测试: `test/api_test/test_multi_image_support.py`
- 简单测试: `test/api_test/simple_vision_test.py`

## 注意事项

1. **API限制**
   - 注意API的速率限制
   - 注意单次请求的大小限制
   - 注意token消耗

2. **向后兼容**
   - 保留原有逐页处理模式
   - 通过配置开关选择模式
   - 确保现有功能不受影响

3. **错误处理**
   - 批量模式失败时自动降级到逐页模式
   - 添加详细的错误日志
   - 提供用户友好的错误提示
