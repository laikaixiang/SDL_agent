# 前端无法显示文字的诊断清单

## 问题描述
app.py运行后，无法把文字推送到前端，大概率是前端的问题（最近只修改了前端）

## 诊断步骤

### 1. 启动服务器
```bash
python app.py
```

### 2. 运行API测试
```bash
python test/test_api.py
```

如果API测试通过，说明后端正常，问题在前端。

### 3. 浏览器控制台检查

打开浏览器（http://127.0.0.1:5000），按F12打开开发者工具：

#### 检查点1: Console标签
- 是否有JavaScript错误？
- 是否有网络请求失败？

#### 检查点2: Network标签
- 发送消息后，是否有 `/api/chat` 请求？
- 请求的状态码是什么？（应该是200）
- Response Headers 中的 Content-Type 是什么？
  - 普通聊天应该是：`text/plain; charset=utf-8`
  - 提取模式应该是：`application/json`
- Response 标签中是否有内容？

#### 检查点3: Elements标签
- 发送消息后，`<div id="chat-box">` 中是否有新的消息元素？
- 消息元素的结构是否正确？

## 常见问题和解决方案

### 问题1: 流式响应不显示

**症状：** API返回数据，但前端不显示

**可能原因：**
1. `createMessageDiv()` 没有正确创建元素
2. `msgDiv.textContent` 没有触发DOM更新
3. CSS隐藏了消息

**解决方案：** 检查 index.html 第254行的 `createMessageDiv('ai')` 是否正常执行

### 问题2: Content-Type判断错误

**症状：** 应该是流式响应，但被当作JSON处理（或反之）

**可能原因：** 后端返回的 Content-Type 不正确

**解决方案：** 检查 app.py 第396行的 `content_type='text/plain; charset=utf-8'`

### 问题3: 流式读取失败

**症状：** 请求成功，但读取流时出错

**可能原因：** `response.body.getReader()` 失败

**解决方案：** 检查浏览器控制台是否有 `TypeError` 或 `ReadableStream` 相关错误

## 快速修复建议

如果确认是前端问题，可以尝试以下修复：

### 修复1: 添加调试日志

在 index.html 的 `sendMessage()` 函数中添加 console.log：

```javascript
// 第254行之后添加
const msgDiv = createMessageDiv('ai');
console.log('Created message div:', msgDiv);
console.log('Chat box:', chatBox);
```

### 修复2: 检查流式读取

在第256-261行的循环中添加日志：

```javascript
while (true) {
    const { done, value } = await reader.read();
    console.log('Read chunk:', done, value);
    if (done) break;
    const decoded = decoder.decode(value, {stream: true});
    console.log('Decoded:', decoded);
    msgDiv.textContent += decoded;
    scrollToBottom();
}
```

### 修复3: 简化测试

创建一个最简单的测试，直接在浏览器控制台运行：

```javascript
fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'chat', message: '测试' })
}).then(async response => {
    console.log('Response:', response);
    console.log('Content-Type:', response.headers.get('content-type'));
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        console.log('Chunk:', decoder.decode(value, {stream: true}));
    }
});
```

## 下一步

1. 启动 app.py
2. 运行 `python test/test_api.py` 确认API正常
3. 打开浏览器，按F12，查看Console和Network标签
4. 发送一条测试消息
5. 根据上述检查点定位问题
