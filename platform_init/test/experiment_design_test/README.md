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

### DIAGNOSIS_REPORT.md
完整的诊断报告，包含：
- 测试结果总结
- 问题定位分析
- 多种解决方案对比
- 性能数据
- 推荐实施步骤

### SUMMARY.md
修复总结文档，包含：
- 问题诊断过程
- 修复内容（代码对比）
- 测试文件说明
- 验证步骤
- 性能数据
- 后续优化建议

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

## 联系信息

如有问题，请查看：
- `DIAGNOSIS_REPORT.md` - 详细诊断过程
- `SUMMARY.md` - 修复总结
- Flask控制台日志 - 运行时错误信息
- 浏览器控制台 - 前端错误信息
