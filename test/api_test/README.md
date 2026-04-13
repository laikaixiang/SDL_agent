# API 测试工具

这个文件夹包含用于测试API和前端功能的工具。

## 测试文件

### 1. test_api.py
Python脚本，用于测试后端API是否正常工作。

**使用方法：**
```bash
# 确保app.py正在运行
python test/test_api.py
```

**测试内容：**
- 服务器连接测试
- 普通聊天接口（流式响应）
- 带前缀的聊天

### 2. test_frontend.html
HTML页面，用于测试前端JavaScript是否能正确接收API响应。

**使用方法：**
1. 确保app.py正在运行
2. 在浏览器中打开：http://127.0.0.1:5000/static/test_frontend.html
   或者直接用浏览器打开这个HTML文件

**测试内容：**
- 流式聊天响应
- JSON响应
- 模拟前端完整流程

## 常见问题排查

### 如果API测试通过但前端无法显示：
1. 检查浏览器控制台是否有JavaScript错误
2. 检查网络请求是否成功（F12 -> Network标签）
3. 检查前端代码中的流式响应处理逻辑

### 如果API测试失败：
1. 确认app.py正在运行
2. 检查API配置（core/config.py）
3. 检查API密钥是否有效
4. 查看app.py的控制台输出
