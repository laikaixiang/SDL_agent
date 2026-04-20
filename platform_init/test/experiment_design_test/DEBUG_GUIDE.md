# 实验设计对话调试指南

## 修复内容总结

### 1. 后端修复（app.py）
- **修改文件**: `app.py:803-920`
- **修改内容**: `/api/experiment_chat` 路由从使用 `field_inference.ExperimentDesignAgent` 切换到使用全局 `experiment_agent`
- **关键变化**:
  - 使用异步调用 `experiment_agent.run()`
  - 通过事件回调获取实验设计 JSON
  - 支持会话管理和多轮对话

### 2. 前端调试日志（experiment_chat.js）
- **修改文件**: `templates/static/js/experiment/experiment_chat.js`
- **添加内容**: 详细的 console.log 调试信息
- **目的**: 追踪前端调用流程

### 3. 后端调试日志（app.py）
- **修改文件**: `app.py:803-830`
- **添加内容**: 详细的打印日志
- **目的**: 追踪后端接收和处理流程

## 测试步骤

### 步骤1：启动应用
```bash
cd D:\PycharmProjects\SDL_agent
python app.py
```

**预期输出**:
```
[会话管理] 应用启动，会话时间戳: 20260420_XXXXXX
[会话管理] 数据保存路径: dialogue data\20260420_XXXXXX
 * Running on http://127.0.0.1:5000
```

### 步骤2：打开浏览器
1. 访问 http://127.0.0.1:5000
2. 打开浏览器开发者工具（F12）
3. 切换到 Console 标签

### 步骤3：发送实验设计请求
在聊天输入框中输入：
```
实验设计：设计一个旋涂实验，转速3000rpm，加速度1000rpm/s，持续时间30秒，使用PbI2试剂，体积50µl
```

### 步骤4：检查浏览器控制台日志

**预期看到的日志**:
```javascript
[ExperimentChat] startExperimentChat 被调用
[ExperimentChat] command: 设计一个旋涂实验，转速3000rpm...
[ExperimentChat] sessionId: exp_1713600000000
[ExperimentChat] 发送请求到 /api/experiment_chat
[ExperimentChat] 响应状态: 200
[ExperimentChat] 响应数据: {type: 'experiment_design', experiment_json: {...}, ...}
[ExperimentChat] 类型: experiment_design
[ExperimentChat] 加载实验设计 JSON
[ExperimentChat] experiment_json: {experiment_name: '钙钛矿层旋涂实验', steps: [...]}
```

**如果没有看到日志**:
- 检查是否有 JavaScript 错误
- 检查 `startExperimentChat` 是否被调用
- 检查网络请求是否发送（Network 标签）

### 步骤5：检查服务器控制台日志

**预期看到的日志**:
```
============================================================
[实验设计] /api/experiment_chat 被调用
============================================================

[实验设计] 接收到的数据:
  - session_id: exp_1713600000000
  - message: 设计一个旋涂实验，转速3000rpm...

============================================================
[实验设计] 开始生成实验方案
[实验设计] Session ID: exp_1713600000000
[实验设计] 用户需求: 设计一个旋涂实验...
============================================================

[ExperimentAgent] 开始处理会话 exp_1713600000000
[ExperimentAgent] 用户消息: 设计一个旋涂实验...
[ExperimentAgent] 无关联PDF
[ExperimentAgent] 开始调用 Approach 2 Agent...
[ExperimentAgent] Agent调用完成
[ExperimentAgent] 会话历史已更新，共 2 条消息
[实验设计] 事件: experiment_design_generated
[实验设计] Agent 返回: ✅ 已生成实验设计方案：钙钛矿层旋涂实验

共 4 个步骤。
[实验设计] 捕获事件数: 1

============================================================
[实验设计] ✅ 生成成功
[实验设计] 实验名称: 钙钛矿层旋涂实验
[实验设计] 步骤数量: 4

[实验设计] 完整JSON:
{
  "experiment_name": "钙钛矿层旋涂实验",
  "steps": [...]
}
============================================================

[实验设计] 已转换为前端可视化格式
[实验设计] 节点数量: 4
[实验设计] 边数量: 3
```

**如果没有看到日志**:
- 检查 `/api/experiment_chat` 是否被调用
- 检查前端是否正确发送请求

## 常见问题排查

### 问题1：前端没有调用 startExperimentChat

**症状**: 浏览器控制台没有 `[ExperimentChat]` 日志

**排查步骤**:
1. 检查 `/api/chat` 是否返回 `experiment_design_mode` 类型
2. 检查 `chat.js` 是否正确处理该类型
3. 检查 `experiment_chat.js` 是否正确加载

**测试命令**:
```javascript
// 在浏览器控制台执行
startExperimentChat("测试命令");
```

### 问题2：后端没有收到请求

**症状**: 服务器控制台没有 `[实验设计] /api/experiment_chat 被调用` 日志

**排查步骤**:
1. 检查浏览器 Network 标签，查看请求是否发送
2. 检查请求 URL 是否正确（应该是 `/api/experiment_chat`）
3. 检查请求方法是否为 POST
4. 检查请求体是否包含 `session_id` 和 `message`

### 问题3：后端返回错误

**症状**: 响应类型为 `error`

**排查步骤**:
1. 查看服务器控制台的错误堆栈
2. 检查 `experiment_agent` 是否正确初始化
3. 检查 LLM API 是否可用

**常见错误**:
- `消息不能为空` - 前端没有正确传递 message
- `实验设计生成失败` - LLM 调用失败或返回格式错误

### 问题4：前端没有显示实验设计

**症状**: 后端成功返回，但前端没有显示

**排查步骤**:
1. 检查浏览器控制台是否有 `experiment_json` 日志
2. 检查 `loadExperimentFromJSON` 是否被调用
3. 检查是否有 JavaScript 错误
4. 检查实验设计面板是否打开

**测试命令**:
```javascript
// 在浏览器控制台执行
loadExperimentFromJSON({
  experiment_name: "测试实验",
  steps: [
    {type: "tool", name: "spin_coating", params: {spin_speed: 3000}, description: "测试"}
  ]
});
```

## 端到端测试

### 自动化测试
```bash
# 确保 Flask 应用正在运行
python app.py

# 新开终端，运行端到端测试
python platform_init/test/experiment_design_test/test_e2e.py
```

**预期结果**:
```
步骤1 (/api/chat): [PASS]
步骤2 (/api/experiment_chat): [PASS]

[OK] 所有测试通过 - 端到端流程正常
```

### 手动测试清单

- [ ] 启动应用成功
- [ ] 浏览器打开页面成功
- [ ] 输入 "实验设计：xxx" 后发送
- [ ] `/api/chat` 返回 `experiment_design_mode`
- [ ] 前端调用 `startExperimentChat`
- [ ] 前端发送请求到 `/api/experiment_chat`
- [ ] 后端收到请求并打印日志
- [ ] 后端调用 `experiment_agent.run()`
- [ ] 后端生成实验设计 JSON
- [ ] 后端返回 `experiment_design` 类型
- [ ] 前端收到响应并打印日志
- [ ] 前端调用 `loadExperimentFromJSON`
- [ ] 实验设计面板显示实验流程
- [ ] 显示成功通知

## 相关文件

### 后端
- `app.py:204-352` - `/api/chat` 路由（识别实验设计请求）
- `app.py:803-920` - `/api/experiment_chat` 路由（生成实验设计）
- `core/experiment_agent.py` - 交互式 ExperimentDesignAgent

### 前端
- `templates/static/js/chat/chat.js:87-90` - 处理 `experiment_design_mode`
- `templates/static/js/experiment/experiment_chat.js` - 调用 `/api/experiment_chat`
- `templates/static/js/experiment/experiment_design.js:395-420` - `loadExperimentFromJSON`

### 测试
- `platform_init/test/experiment_design_test/test_experiment_chat.py` - 单元测试
- `platform_init/test/experiment_design_test/test_api_request.py` - API 测试
- `platform_init/test/experiment_design_test/test_e2e.py` - 端到端测试

## 下一步

如果所有测试通过但前端仍然无法显示：
1. 清除浏览器缓存（Ctrl+Shift+Delete）
2. 硬刷新页面（Ctrl+F5）
3. 检查浏览器控制台的完整日志
4. 检查 Network 标签的完整请求/响应
5. 提供具体的错误信息或日志
