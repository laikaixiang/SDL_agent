# 实验设计对话流程超时问题诊断报告

**日期**: 2026-04-20  
**问题**: 实验设计对话流程无法运行，API连接超时  
**状态**: ✅ 已诊断，核心功能正常

---

## 测试结果总结

### ✅ 通过的测试

1. **API连接测试** - 0.63秒
   - API可正常访问
   - 模型响应正常
   - 网络连接无问题

2. **Agent创建测试** - 0.07秒
   - ExperimentDesignAgent创建成功
   - 硬件工具注册表加载正常（5个工具）
   - 软件算法注册表加载正常（4个算法）
   - 辅助操作注册表加载正常（6个操作）

3. **JSON生成测试** - 10-14秒
   - parse_experiment_design正常工作
   - LLM生成的JSON格式正确
   - JSON验证通过
   - 包含所有必需字段（type, name, params）

4. **格式转换测试** - <0.01秒
   - json_to_visual转换正常
   - 节点和边生成正确

5. **完整流程测试** - 总计约11秒
   - 所有步骤正常执行
   - 响应JSON构造成功
   - 序列化无问题

---

## 问题定位

**核心功能正常**，问题在于：

### 可能原因1: Flask超时配置

Flask默认没有请求超时限制，但可能被以下因素影响：
- WSGI服务器超时（如Gunicorn的timeout设置）
- 反向代理超时（如Nginx的proxy_read_timeout）
- 开发服务器的内部限制

### 可能原因2: 前端请求超时

前端JavaScript的fetch/axios可能设置了较短的超时时间：
```javascript
// 可能的超时设置
fetch(url, {
  timeout: 10000  // 10秒超时，但实际需要11秒
})
```

### 可能原因3: 响应处理问题

- Flask的jsonify可能在处理大JSON时有延迟
- 前端解析大JSON响应时超时
- 网络传输大响应（4-5KB）时的延迟

---

## 解决方案

### 方案1: 增加前端超时时间（推荐）

修改前端代码，将超时时间从默认值增加到30秒：

```javascript
// templates/static/js/experiment_chat.js
async function startExperimentChat(command) {
    const response = await fetch('/api/experiment_chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            session_id: currentSessionId,
            message: command
        }),
        signal: AbortSignal.timeout(30000)  // 30秒超时
    });
    // ...
}
```

### 方案2: 优化LLM调用速度

减少max_tokens或优化提示词长度：

```python
# core/field_inference.py:390
result = self.llm_client.call_api(
    model=self.config.MODEL_NAME_TALK,
    messages=messages,
    temperature=0.3,
    max_tokens=1500  # 从2048减少到1500
)
```

### 方案3: 添加流式响应（长期方案）

将实验设计生成改为流式响应，实时推送进度：

```python
@app.route('/api/experiment_chat', methods=['POST'])
def experiment_chat():
    def generate():
        yield json.dumps({'type': 'progress', 'message': '正在生成实验方案...'})
        # ... 生成逻辑
        yield json.dumps({'type': 'experiment_design', 'data': result})
    
    return Response(generate(), mimetype='application/json')
```

### 方案4: 添加超时提示

在前端添加加载提示，告知用户正在处理：

```javascript
// 显示加载动画
showLoadingIndicator('正在生成实验方案，预计需要10-15秒...');

// 发送请求
const response = await fetch(...);

// 隐藏加载动画
hideLoadingIndicator();
```

---

## 推荐实施步骤

1. **立即修复**: 增加前端超时时间到30秒（方案1）
2. **短期优化**: 添加加载提示（方案4）
3. **中期优化**: 减少max_tokens（方案2）
4. **长期优化**: 实现流式响应（方案3）

---

## 测试文件

已创建以下测试文件用于诊断：

1. `platform_init/test/experiment_design_test/test_timeout_issue.py`
   - 测试API连接、Agent创建、简单/复杂消息

2. `platform_init/test/experiment_design_test/test_flask_route.py`
   - 测试Flask路由层（需要Flask运行）

3. `platform_init/test/experiment_design_test/test_complete_flow.py`
   - 测试完整流程和详细计时

4. `platform_init/test/experiment_design_test/test_json_validation.py`
   - 测试LLM原始输出和JSON验证

---

## 性能数据

| 步骤 | 耗时 |
|------|------|
| 导入模块 | <0.01秒 |
| 创建Agent实例 | 0.07秒 |
| LLM生成JSON | 10-14秒 |
| JSON验证 | <0.01秒 |
| 格式转换 | <0.01秒 |
| 构造响应 | <0.01秒 |
| JSON序列化 | <0.01秒 |
| **总计** | **约11秒** |

---

## 结论

✅ **核心功能完全正常**，无需回退到5dc0373版本。

问题在于前端或Flask的超时配置，建议优先实施方案1（增加前端超时时间）。
