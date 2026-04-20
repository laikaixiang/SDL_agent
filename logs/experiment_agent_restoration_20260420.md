# ExperimentDesignAgent 恢复报告

## 恢复时间
2026-04-20

## 背景
用户指出 `app.py` 需要使用 `ExperimentDesignAgent` 来接入"实验设计对话"功能，包括 `/api/experiment_confirm` 路由。

## 问题
之前的清理工作错误地移除了 `ExperimentDesignAgent` 的引用，导致以下功能失效：
1. PDF 上传后无法关联到实验设计会话
2. `/api/experiment_confirm` 路由无法处理用户确认响应

## 修复内容

### 1. 恢复 core/__init__.py 导出（第10行）

**修改前：**
```python
from .field_inference import FieldInference, AlgorithmParser, ExperimentDesignAgent
# from .experiment_agent import ExperimentDesignAgent  # Deprecated
```

**修改后：**
```python
from .field_inference import FieldInference, AlgorithmParser
from .experiment_agent import ExperimentDesignAgent  # Interactive experiment design agent (Approach 2 based)
```

**说明：**
- `field_inference.ExperimentDesignAgent` - 纯解析器，无交互功能
- `experiment_agent.ExperimentDesignAgent` - 交互式版本，支持会话管理、PDF 关联、用户确认

### 2. 恢复 app.py 导入（第36行）

**修改前：**
```python
from core import (
    ...
    SoftwareManager,
    AdaptiveStreamHandler,
)
```

**修改后：**
```python
from core import (
    ...
    SoftwareManager,
    AdaptiveStreamHandler,
    ExperimentDesignAgent,
)
```

### 3. 恢复 app.py 初始化（第54行）

**修改前：**
```python
adaptive_handler = AdaptiveStreamHandler(config, llm_client)  # 自适应流式响应处理器
```

**修改后：**
```python
adaptive_handler = AdaptiveStreamHandler(config, llm_client)  # 自适应流式响应处理器
experiment_agent = ExperimentDesignAgent()  # 实验设计智能体（Approach 2）
```

### 4. 恢复 PDF 上传路由（第897-899行）

**修改前：**
```python
file.save(path)

# TODO: 方案2不支持交互式PDF读取，保留路径供未来扩展
# Note: experiment_agent removed - using Approach 2 directly in /api/experiment_chat
return jsonify({'filename': safe_name, 'path': path})
```

**修改后：**
```python
file.save(path)

# 关联 PDF 到实验设计会话
experiment_agent.set_pdf_path(session_id, path)
return jsonify({'filename': safe_name, 'path': path})
```

### 5. 恢复实验确认路由（第903-935行）

**修改前：**
```python
@app.route('/api/experiment_confirm', methods=['POST'])
def experiment_confirm():
    """处理实验确认响应（方案2暂不支持交互式确认）"""
    # ...
    # TODO: 方案2不支持交互式确认，保留接口供未来扩展
    # Note: experiment_agent removed - using Approach 2 directly in /api/experiment_chat
    return jsonify({'status': 'success', 'message': '方案2暂不支持交互式确认'})
```

**修改后：**
```python
@app.route('/api/experiment_confirm', methods=['POST'])
def experiment_confirm():
    """处理实验确认响应"""
    data = request.json
    request_id = data.get('request_id')
    session_id = data.get('session_id')
    action = data.get('action')
    params = data.get('params', {})

    if not request_id or not session_id:
        return jsonify({'error': 'Missing request_id or session_id'}), 400

    # Submit response to the agent's queue
    response = {
        "action": action,
        "params": params
    }
    experiment_agent.submit_response(request_id, response)

    return jsonify({'status': 'success'})
```

## 验证结果

```bash
✅ app.py imports successfully
✅ experiment_agent type: ExperimentDesignAgent
✅ Available methods: ['clear_session', 'config', 'create_response_queue', 
                       'get_active_sessions', 'run', 'set_pdf_path', 
                       'submit_response', 'wait_for_response']
```

## 架构说明

### ExperimentDesignAgent 的两个版本

**版本1：field_inference.ExperimentDesignAgent**
- 位置：`core/field_inference.py`
- 功能：纯实验设计解析器
- 方法：`parse_experiment_design(user_input)` → 返回 JSON
- 用途：单次解析，无状态

**版本2：experiment_agent.ExperimentDesignAgent**（当前使用）
- 位置：`core/experiment_agent.py`
- 功能：交互式实验设计智能体
- 方法：
  - `run(session_id, user_message, send_event)` - 异步对话
  - `set_pdf_path(session_id, pdf_path)` - 关联 PDF
  - `submit_response(request_id, response)` - 处理用户确认
  - `wait_for_response(request_id, timeout)` - 等待用户响应
  - `clear_session(session_id)` - 清除会话
  - `get_active_sessions()` - 获取活跃会话列表
- 用途：多轮对话，支持会话管理和用户交互
- 实现：基于 Approach 2（内部使用 field_inference.ExperimentDesignAgent）

### 为什么需要两个版本？

- **版本1（field_inference）**：适用于 `/api/experiment_chat` 单次生成场景
- **版本2（experiment_agent）**：适用于需要多轮对话、PDF 关联、用户确认的复杂场景

## 总结

成功恢复了 `ExperimentDesignAgent` 在 `app.py` 中的使用：
- ✅ 恢复了 `core/__init__.py` 的正确导出
- ✅ 恢复了 `app.py` 的导入和初始化
- ✅ 恢复了 PDF 上传路由的会话关联功能
- ✅ 恢复了实验确认路由的完整实现
- ✅ 所有交互式方法可用（set_pdf_path, submit_response, wait_for_response）

现在 `app.py` 可以正确使用 `ExperimentDesignAgent` 进行实验设计对话和用户交互。
