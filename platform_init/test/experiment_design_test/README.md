# 实验设计测试套件

本目录包含实验设计功能的完整测试代码。

## 测试文件

### 1. test_experiment_design_v2.py
验证方案2（JSON + 提示词）的基础功能：
- ExperimentDesignAgent 初始化
- 系统提示词生成
- JSON 验证功能
- 模拟实验生成

### 2. test_experiment_chat.py（新增）
诊断 `/api/experiment_chat` 接口问题：
- 测试交互式 ExperimentDesignAgent（core/experiment_agent.py）
- 测试单次解析版本（core/field_inference.py）
- 检查 app.py 中的 experiment_agent 配置
- 验证所有必需方法存在

### 3. test_api_request.py（新增）
API 集成测试：
- 模拟前端请求 `/api/experiment_chat`
- 验证完整的请求-响应流程
- 需要 Flask 应用运行

## 问题诊断与修复

### 问题描述
用户报告点击"实验设计对话"后输入无法正确调用 ExperimentDesignAgent。

### 根本原因
`/api/experiment_chat` 路由使用的是 `core.field_inference.ExperimentDesignAgent`（单次解析版本），而不是已初始化的 `experiment_agent`（交互式版本）。

**测试结果：**
- ✅ `app.experiment_agent` 是交互式版本（有 run, set_pdf_path, submit_response 方法）
- ✅ 交互式版本工作正常（成功生成4步实验设计）
- ❌ field_inference 版本失败（返回"生成的JSON格式不符合要求"）

### 修复方案
修改 `app.py:803-920` 的 `/api/experiment_chat` 路由，使用已初始化的 `experiment_agent`（交互式版本）。

**关键变化：**
1. 使用全局 `experiment_agent` 实例（不再创建新实例）
2. 通过 asyncio 调用异步 `run()` 方法
3. 通过事件回调获取实验设计 JSON
4. 支持会话管理和多轮对话

## 运行测试

### 测试1：单元测试（不需要启动 Flask）

```bash
cd D:\PycharmProjects\SDL_agent
python platform_init/test/experiment_design_test/test_experiment_chat.py
```

**预期结果：**
```
[PASS] - methods          # ExperimentDesignAgent 方法检查
[PASS] - app_import       # app.py 导入检查
[PASS] - field_inference  # field_inference 版本测试
[PASS] - interactive      # 交互式版本测试
```

### 测试2：API 集成测试（需要启动 Flask）

**步骤1：启动 Flask 应用**
```bash
cd D:\PycharmProjects\SDL_agent
python app.py
```

**步骤2：运行 API 测试**
```bash
# 新开一个终端
cd D:\PycharmProjects\SDL_agent
python platform_init/test/experiment_design_test/test_api_request.py
```

**预期结果：**
```
[测试] 响应状态码: 200
[测试] 响应类型: experiment_design
[测试] [OK] 实验设计生成成功
[测试] 实验名称: 钙钛矿层旋涂实验
[测试] 步骤数: 4
```

### 测试3：前端手动测试

1. 启动应用：`python app.py`
2. 打开浏览器：http://127.0.0.1:5000
3. 点击"实验设计对话"
4. 输入：`设计一个旋涂实验，转速3000rpm，加速度1000rpm/s，持续时间30秒，使用PbI2试剂，体积50µl`
5. 点击发送

**预期结果：**
- ✅ 控制台输出实验设计 JSON
- ✅ 前端显示成功消息
- ✅ 实验流程画布显示4个节点

## ExperimentDesignAgent 架构

### 版本1：field_inference.ExperimentDesignAgent
- **位置**: `core/field_inference.py`
- **类型**: 同步，单次解析
- **方法**: `parse_experiment_design(user_input) -> (bool, dict)`
- **用途**: 简单的一次性解析
- **问题**: 当前实现有 JSON 格式验证问题

### 版本2：experiment_agent.ExperimentDesignAgent（推荐）
- **位置**: `core/experiment_agent.py`
- **类型**: 异步，交互式
- **方法**:
  - `async run(session_id, user_message, send_event) -> str`
  - `set_pdf_path(session_id, pdf_path)`
  - `submit_response(request_id, response)`
  - `wait_for_response(request_id, timeout)`
- **用途**: 多轮对话、会话管理、PDF 关联、用户确认
- **优势**:
  - ✅ 支持会话历史
  - ✅ 支持 PDF 上传关联
  - ✅ 支持用户确认流程
  - ✅ 事件驱动架构
  - ✅ 基于 Approach 2（内部使用 field_inference）

### 修复后的架构
- `/api/experiment_chat` → 使用 `experiment_agent`（版本2）
- `/api/experiment_confirm` → 使用 `experiment_agent.submit_response()`
- `/api/experiment_upload` → 使用 `experiment_agent.set_pdf_path()`

## 方案2说明

方案2使用JSON + 提示词的方式实现实验设计：
- **优势**: 不需要Function Calling支持，任何LLM都可使用
- **实现**: `core/field_inference.py:ExperimentDesignAgent`
- **注册表**: 
  - 硬件工具: `hardware/tools/REGISTRY.json`
  - 软件算法: 通过 `SoftwareController` 动态加载
  - 辅助操作: 内置在 `ExperimentDesignAgent` 中

## 相关文件

- `app.py:803-920` - `/api/experiment_chat` 路由（已修复）
- `core/experiment_agent.py` - 交互式 ExperimentDesignAgent
- `core/field_inference.py` - 单次解析版本
- `hardware/tools/REGISTRY.json` - 硬件工具注册表
- `software/algorithms/` - 软件算法目录
- `CLAUDE.md` - 完整文档
