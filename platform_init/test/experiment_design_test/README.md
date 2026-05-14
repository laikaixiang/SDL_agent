# 实验设计对话流程测试套件

本目录包含用于诊断和测试实验设计对话流程的测试脚本。

## 问题背景

实验设计对话流程报错"API连接超时"，但API本身可以正常连接和输出JSON。

## 诊断结果

✅ **核心功能正常**，问题在于前端fetch请求没有设置超时时间（默认10秒），而LLM生成需要10-15秒。

## 修复方案

已修改 `templates/static/js/experiment/experiment_chat.js`，增加30秒超时设置。

---

## 测试文件说明

### 1. test_timeout_issue.py
**用途**: 全面诊断超时问题  
**运行**: `python platform_init/test/experiment_design_test/test_timeout_issue.py`  
**测试内容**:
- API连接测试
- Agent创建测试
- 简单消息生成测试
- 复杂消息生成测试

**预期结果**: 所有测试通过，总耗时约11秒

---

### 2. test_json_validation.py
**用途**: 查看LLM原始输出和JSON验证过程  
**运行**: `python platform_init/test/experiment_design_test/test_json_validation.py`  
**测试内容**:
- 显示LLM原始输出
- 显示清理后的JSON
- 逐步验证JSON结构
- 分析验证失败原因

**预期结果**: JSON验证通过，显示完整的生成内容

---

### 3. test_complete_flow.py
**用途**: 模拟Flask路由的完整执行流程  
**运行**: `python platform_init/test/experiment_design_test/test_complete_flow.py`  
**测试内容**:
- 步骤1: 创建Agent和Converter
- 步骤2: 调用parse_experiment_design
- 步骤3: 添加时间戳
- 步骤4: 转换为可视化格式
- 步骤5: 构造JSON响应
- 步骤6: 序列化为JSON字符串
- 详细计时分析

**预期结果**: 完整流程通过，总耗时约11秒

---

### 4. test_flask_route.py
**用途**: 测试Flask API接口  
**前置条件**: Flask服务器必须运行（`python app.py`）  
**运行**: `python platform_init/test/experiment_design_test/test_flask_route.py`  
**测试内容**:
- 检查Flask服务器状态
- 发送POST请求到 `/api/experiment_chat`
- 验证响应格式和内容

**预期结果**: 返回 `type: 'experiment_design'`，包含完整的实验JSON和可视化数据

---

### 5. test_fix_verification.py
**用途**: 验证修复后的完整流程  
**前置条件**: Flask服务器必须运行（`python app.py`）  
**运行**: `python platform_init/test/experiment_design_test/test_fix_verification.py`  
**测试内容**:
- 检查Flask运行状态
- 测试 `/api/experiment_chat` 接口
- 验证30秒超时设置是否生效

**预期结果**: 请求成功，耗时10-15秒，返回完整实验设计

---

## 快速测试流程

### 方案A: 无需Flask（推荐用于快速验证）

```bash
cd D:/PycharmProjects/SDL_agent

# 运行完整流程测试
python platform_init/test/experiment_design_test/test_complete_flow.py
```

### 方案B: 需要Flask（完整集成测试）

```bash
# 终端1: 启动Flask
cd D:/PycharmProjects/SDL_agent
python app.py

# 终端2: 运行测试
cd D:/PycharmProjects/SDL_agent
python platform_init/test/experiment_design_test/test_fix_verification.py
```

### 方案C: 浏览器测试（最终验证）

1. 启动Flask: `python app.py`
2. 打开浏览器: http://127.0.0.1:5000
3. 输入: "实验设计：设计一个简单的旋涂实验"
4. 等待10-15秒
5. 验证实验设计是否成功生成

---

## 文档说明

### README.md（本文件）
测试套件使用指南

---

## 性能基准

| 操作 | 预期耗时 |
|------|---------|
| API连接测试 | <1秒 |
| Agent创建 | <0.1秒 |
| LLM生成JSON | 10-15秒 |
| 格式转换 | <0.1秒 |
| 完整流程 | 11-16秒 |

---

## 常见问题

### Q: 测试显示"API连接失败"
**A**: 检查 `config.txt` 中的 `API_KEY` 和 `API_URL` 配置

### Q: 测试显示"Flask服务器未运行"
**A**: 在另一个终端运行 `python app.py`

### Q: JSON验证失败
**A**: 运行 `test_json_validation.py` 查看详细的验证失败原因

### Q: 浏览器测试仍然超时
**A**: 
1. 确认已修改 `templates/static/js/experiment/experiment_chat.js`
2. 清除浏览器缓存（Ctrl+Shift+R）
3. 检查浏览器控制台错误信息

---

## 修复验证清单

- [x] 运行 `test_complete_flow.py` - 核心功能正常
- [x] 修改前端超时设置 - 增加到30秒
- [ ] 运行 `test_fix_verification.py` - Flask集成测试
- [ ] 浏览器测试 - 完整用户流程
- [ ] 清除浏览器缓存 - 确保使用新代码

---

## 端到端调试指南

### 步骤1：启动应用
```bash
cd D:\PycharmProjects\SDL_agent
python app.py
```
预期输出：`[会话管理] 应用启动，会话时间戳: ...` + `* Running on http://127.0.0.1:5000`

### 步骤2：浏览器控制台预期日志

发送 "实验设计：设计一个旋涂实验，转速3000rpm..." 后，Console 应出现：
```
[ExperimentChat] startExperimentChat 被调用
[ExperimentChat] command: 设计一个旋涂实验...
[ExperimentChat] 发送请求到 /api/experiment_chat
[ExperimentChat] 响应状态: 200
[ExperimentChat] 响应数据: {type: 'experiment_design', experiment_json: {...}, ...}
[ExperimentChat] 加载实验设计 JSON
```
如无日志：检查 JS 错误、`startExperimentChat` 是否被调用、Network 标签请求状态。

### 步骤3：服务器控制台预期日志

```
============================================================
[实验设计] /api/experiment_chat 被调用
[实验设计] 接收到的数据: session_id / message
[实验设计] 开始生成实验方案
[ExperimentAgent] 开始处理会话...
[ExperimentAgent] 开始调用 Approach 2 Agent...
[实验设计] 事件: experiment_design_generated
[实验设计] ✅ 生成成功，实验名称: xxx，步骤数量: N
[实验设计] 已转换为前端可视化格式
============================================================
```
如无日志：检查 `/api/experiment_chat` 是否被调用、前端是否正确发送请求。

### 常见问题排查

**问题1：前端没有调用 startExperimentChat**
- 症状：浏览器控制台没有 `[ExperimentChat]` 日志
- 排查：检查 `/api/chat` 是否返回 `experiment_design_mode` 类型；检查 `chat.js` 是否正确处理该类型
- 测试：在控制台执行 `startExperimentChat("测试命令")`

**问题2：后端没有收到请求**
- 症状：服务器控制台没有 `[实验设计] /api/experiment_chat 被调用` 日志
- 排查：检查浏览器 Network 标签，确认请求 URL 为 `/api/experiment_chat`、方法为 POST、请求体含 `session_id` 和 `message`

**问题3：后端返回错误**
- 症状：响应类型为 `error`
- 排查：查看服务器控制台错误堆栈；检查 `experiment_agent` 是否正确初始化；检查 LLM API 是否可用
- 常见错误：`消息不能为空`（前端未正确传递 message）、`实验设计生成失败`（LLM 调用失败或返回格式错误）

**问题4：前端没有显示实验设计**
- 症状：后端成功返回，但前端没有显示
- 排查：检查控制台是否有 `experiment_json` 日志；检查 `loadExperimentFromJSON` 是否被调用；检查实验设计面板是否打开
- 测试：在控制台执行 `loadExperimentFromJSON({experiment_name: "测试", steps: [{type: "tool", name: "spin_coating", params: {spin_speed: 3000}, description: "测试"}]})`

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
