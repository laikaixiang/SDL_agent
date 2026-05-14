# API测试工具

独立的API配置测试工具，无需启动app.py即可验证API配置是否正确。

## 功能

- ✅ 验证配置文件是否完整
- ✅ 测试API连接是否正常
- ✅ 测试对话模型是否可用
- ✅ 测试流式响应是否正常
- ✅ 验证视觉模型配置
- ✅ 检查文件路径是否存在

## 使用方法

### 从项目根目录运行

```bash
python test/api_test/api_test.py
```

## 测试内容

1. Configuration Validation - 检查配置项
2. API Connection Test - 测试API连接
3. Talk Model Test - 测试对话模型
4. Streaming Response Test - 测试流式响应（前端显示的关键）
5. Vision-Language Model Test - 验证视觉模型配置
6. File Paths Test - 检查文件路径

## 故障排除

如果 Streaming Response Test 失败，可能导致前端无法显示文字。



# 自适应流式响应功能 - 实施完成

## 已完成的工作

### 1. 创建核心模块
✅ **文件**: `core/adaptive_stream.py`
- 实现了 `AdaptiveStreamHandler` 类
- 自动检测API流式支持
- 智能缓存检测结果（1小时）
- 支持流式和非流式自动切换
- 不支持流式时模拟流式输出

### 2. 修改核心导出
✅ **文件**: `core/__init__.py`
- 添加了 `AdaptiveStreamHandler` 的导出

### 3. 集成到应用
✅ **文件**: `app.py`
- 导入 `AdaptiveStreamHandler`
- 初始化自适应处理器
- 修改 `handle_normal_chat` 函数使用自适应处理器
- 添加了两个新的API端点：
  - `/api/streaming_status` - 查询流式支持状态
  - `/api/streaming_recheck` - 强制重新检测

### 4. 创建测试工具
✅ **文件**: `test/api_test/test_adaptive_stream.py`
- 独立测试脚本
- 测试流式检测功能
- 测试流式和非流式响应

✅ **文件**: `test/api_test/run_test.bat`
- Windows批处理脚本
- 自动激活conda环境并运行测试

## 使用方法

### 方式1: 直接运行app.py

```bash
conda activate SDL_agent
python app.py
```

启动后，系统会自动检测API是否支持流式响应，并在控制台输出：
- `[自适应流式] ✓ API支持流式响应` 或
- `[自适应流式] × API不支持流式响应，将使用非流式模式`

### 方式2: 运行测试脚本

**Windows:**
```bash
cd test/api_test
run_test.bat
```

**或手动运行:**
```bash
conda activate SDL_agent
python test/api_test/test_adaptive_stream.py
```

### 方式3: 使用API端点

启动app.py后，可以通过以下端点查询状态：

**查询流式支持状态:**
```bash
curl http://127.0.0.1:5000/api/streaming_status
```

返回示例：
```json
{
  "streaming_support": false,
  "last_check_time": 1234567890,
  "check_interval": 3600,
  "time_until_recheck": 3500
}
```

**强制重新检测:**
```bash
curl -X POST http://127.0.0.1:5000/api/streaming_status
```

## 工作原理

```
用户发送消息
    ↓
app.py 调用 adaptive_handler.generate_response()
    ↓
首次调用？
    ↓ 是
发送测试请求检测流式支持 → 缓存结果
    ↓ 否
使用缓存结果（1小时内有效）
    ↓
支持流式？
    ↓ 是                              ↓ 否
使用真正的流式响应          使用非流式请求 + 模拟流式输出
    ↓                                  ↓
逐块返回给前端                  逐字返回给前端（10ms/字符）
```

## 优势

1. **自动适配**: 无需手动配置，系统自动检测
2. **性能优化**: 检测结果缓存1小时，避免重复检测
3. **用户体验**: 即使不支持流式，也能模拟流式效果
4. **易于调试**: 提供状态查询和强制重检接口
5. **向后兼容**: 不影响现有功能，前端无需修改

## 前端体验

无论API是否支持流式响应，前端都能看到：
- 文字逐步显示（流式效果）
- 实时响应
- 无需任何修改

## 故障排除

如果遇到问题：
1. 确保已激活SDL_agent环境
2. 检查 `core/config.py` 中的API配置
3. 运行 `python test/api_test/api_test.py` 验证API连接
4. 查看app.py控制台输出的 `[自适应流式]` 日志
