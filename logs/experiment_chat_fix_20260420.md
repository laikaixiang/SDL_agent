# /api/experiment_chat 接口修复报告

## 修复时间
2026-04-20

## 问题描述
用户报告点击"实验设计对话"后输入无法正确调用 ExperimentDesignAgent。

## 问题诊断

### 测试结果
运行 `platform_init/test/experiment_design_test/test_experiment_chat.py` 发现：

1. ✅ **app.experiment_agent 配置正确**
   - 类型：`core.experiment_agent.ExperimentDesignAgent`
   - 方法：`run`, `set_pdf_path`, `submit_response` 全部存在
   - 是交互式版本

2. ✅ **交互式版本工作正常**
   - 成功生成实验设计：钙钛矿层旋涂实验
   - 包含4个步骤：set_temperature → WAIT → spin_coating → WAIT
   - 事件驱动架构正常工作

3. ❌ **field_inference 版本失败**
   - 错误：`生成的JSON格式不符合要求`
   - 原因：JSON 格式验证逻辑有问题

### 根本原因
`/api/experiment_chat` 路由（app.py:826-829）使用的是 `core.field_inference.ExperimentDesignAgent`（失败的版本），而不是已初始化的 `experiment_agent`（成功的版本）。

**问题代码：**
```python
# 使用ExperimentDesignAgent生成JSON
from core.field_inference import ExperimentDesignAgent
from experiment.format import ExperimentFormatConverter

agent = ExperimentDesignAgent()  # 错误：创建了新的 field_inference 版本
converter = ExperimentFormatConverter()

success, result = agent.parse_experiment_design(user_message)  # 失败
```

## 修复方案

### 修改文件：app.py

**位置：** 第803-920行

**修改前：**
```python
@app.route('/api/experiment_chat', methods=['POST'])
def experiment_chat():
    """实验设计对话 - 使用自然语言生成实验设计JSON"""
    
    # 使用ExperimentDesignAgent生成JSON
    from core.field_inference import ExperimentDesignAgent
    agent = ExperimentDesignAgent()
    
    success, result = agent.parse_experiment_design(user_message)
    
    if success:
        # 转换为前端可视化格式
        visual_data = converter.json_to_visual(result)
        return jsonify({
            'type': 'experiment_design',
            'experiment_json': result,
            'visual_data': visual_data,
            'reply': f"✅ 已生成实验设计方案..."
        })
```

**修改后：**
```python
@app.route('/api/experiment_chat', methods=['POST'])
def experiment_chat():
    """实验设计对话 - 使用交互式 ExperimentDesignAgent"""
    
    # 使用交互式 ExperimentDesignAgent
    import asyncio
    
    events = []
    
    # 事件收集回调
    async def collect_event(event):
        events.append(event)
        print(f"[实验设计] 事件: {event.get('type')}")
    
    # 运行异步 agent
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result_text = loop.run_until_complete(
            experiment_agent.run(session_id, user_message, collect_event)
        )
        loop.close()
        
        # 检查是否有实验设计生成事件
        experiment_event = None
        for event in events:
            if event.get('type') == 'experiment_design_generated':
                experiment_event = event
                break
        
        if experiment_event:
            experiment_json = experiment_event.get('experiment_json', {})
            
            # 转换为前端可视化格式
            visual_data = converter.json_to_visual(experiment_json)
            
            return jsonify({
                'type': 'experiment_design',
                'experiment_json': experiment_json,
                'visual_data': visual_data,
                'reply': result_text
            })
```

### 关键变化

1. **使用全局 experiment_agent**
   - 不再创建新的 `field_inference.ExperimentDesignAgent` 实例
   - 使用已初始化的 `experiment_agent`（交互式版本）

2. **异步调用**
   - 使用 `asyncio.new_event_loop()` 创建事件循环
   - 调用 `experiment_agent.run(session_id, user_message, collect_event)`
   - 通过 `loop.run_until_complete()` 等待完成

3. **事件驱动**
   - 通过 `collect_event` 回调收集事件
   - 从 `experiment_design_generated` 事件中提取 JSON
   - 支持未来扩展更多事件类型

4. **会话管理**
   - 支持 `session_id` 参数
   - 支持多轮对话历史
   - 支持 PDF 关联（通过 `experiment_agent.set_pdf_path()`）

## 测试验证

### 创建的测试文件

1. **platform_init/test/experiment_design_test/test_experiment_chat.py**
   - 单元测试，不需要启动 Flask
   - 测试4个方面：方法检查、app导入、field_inference版本、交互式版本
   - 运行：`python platform_init/test/experiment_design_test/test_experiment_chat.py`

2. **platform_init/test/experiment_design_test/test_api_request.py**
   - API 集成测试，需要启动 Flask
   - 模拟前端请求 `/api/experiment_chat`
   - 运行：先启动 `python app.py`，再运行测试脚本

3. **platform_init/test/experiment_design_test/README.md**
   - 完整的测试文档
   - 包含问题诊断、修复方案、测试方法
   - 包含架构说明和相关文件列表

### 测试结果

**单元测试（test_experiment_chat.py）：**
```
[PASS] - methods          # ExperimentDesignAgent 方法检查
[PASS] - app_import       # app.py 导入检查
[FAIL] - field_inference  # field_inference 版本失败（预期）
[PASS] - interactive      # 交互式版本成功
```

**交互式版本输出：**
```
[测试] [OK] 调用成功
[测试] 返回结果: ✅ 已生成实验设计方案：钙钛矿层旋涂实验

共 4 个步骤。
[测试] 捕获事件数: 1
[测试] 事件 1: experiment_design_generated
```

**生成的实验设计：**
```json
{
  "experiment_name": "钙钛矿层旋涂实验",
  "description": "设计并执行一次旋涂实验...",
  "steps": [
    {
      "type": "tool",
      "name": "set_temperature",
      "params": {"target": 100},
      "description": "设置加热台温度为100℃，预热基底。"
    },
    {
      "type": "WAIT",
      "name": "WAIT",
      "params": {"duration": 3000},
      "description": "等待3秒，确保温度稳定。"
    },
    {
      "type": "tool",
      "name": "spin_coating",
      "params": {
        "spin_speed": 3000,
        "spin_acc": 1000,
        "spin_dur": 30000,
        "reagent": "PbI2",
        "volume": 50
      },
      "description": "执行旋涂实验..."
    },
    {
      "type": "WAIT",
      "name": "WAIT",
      "params": {"duration": 5000},
      "description": "等待5秒，确保旋涂完成后薄膜稳定。"
    }
  ]
}
```

## ExperimentDesignAgent 架构说明

### 两个版本对比

| 特性 | field_inference 版本 | experiment_agent 版本 |
|------|---------------------|----------------------|
| 位置 | `core/field_inference.py` | `core/experiment_agent.py` |
| 类型 | 同步，单次解析 | 异步，交互式 |
| 方法 | `parse_experiment_design()` | `run()`, `set_pdf_path()`, `submit_response()` |
| 会话管理 | ❌ 无 | ✅ 支持 |
| PDF 关联 | ❌ 无 | ✅ 支持 |
| 用户确认 | ❌ 无 | ✅ 支持 |
| 事件驱动 | ❌ 无 | ✅ 支持 |
| 当前状态 | ❌ 有 bug | ✅ 工作正常 |

### 修复后的路由映射

- `/api/experiment_chat` → `experiment_agent.run()` - 实验设计对话
- `/api/experiment_confirm` → `experiment_agent.submit_response()` - 用户确认
- `/api/experiment_upload` → `experiment_agent.set_pdf_path()` - PDF 上传

## 相关文件

### 修改的文件
- `app.py:803-920` - `/api/experiment_chat` 路由

### 新增的文件
- `platform_init/test/experiment_design_test/test_experiment_chat.py` - 单元测试
- `platform_init/test/experiment_design_test/test_api_request.py` - API 集成测试
- `platform_init/test/experiment_design_test/README.md` - 测试文档（更新）

### 相关文件
- `core/experiment_agent.py` - 交互式 ExperimentDesignAgent（使用此版本）
- `core/field_inference.py` - 单次解析版本（有 bug，不使用）
- `core/__init__.py` - 导出交互式版本

## 后续建议

1. **修复 field_inference 版本的 bug**
   - 问题：JSON 格式验证逻辑
   - 位置：`core/field_inference.py:ExperimentDesignAgent.parse_experiment_design()`
   - 优先级：低（因为已有可用的交互式版本）

2. **API 集成测试**
   - 启动 Flask 应用
   - 运行 `test_api_request.py`
   - 验证完整的请求-响应流程

3. **前端手动测试**
   - 在浏览器中测试"实验设计对话"功能
   - 验证 UI 交互和数据展示
   - 测试 PDF 上传和关联功能

## 总结

成功修复了 `/api/experiment_chat` 接口，从使用有问题的 `field_inference.ExperimentDesignAgent` 切换到已验证工作正常的交互式 `experiment_agent`。

**修复内容：**
- ✅ 使用全局 `experiment_agent` 实例
- ✅ 通过 asyncio 调用异步 `run()` 方法
- ✅ 通过事件回调获取实验设计 JSON
- ✅ 支持会话管理和多轮对话
- ✅ 支持 PDF 关联功能

**测试验证：**
- ✅ 单元测试通过（交互式版本生成4步实验设计）
- ⏳ API 集成测试（需要启动 Flask）
- ⏳ 前端手动测试（需要启动 Flask）
