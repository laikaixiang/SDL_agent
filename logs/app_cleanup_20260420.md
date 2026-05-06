# app.py 清理报告

## 清理时间
2026-04-20

## 清理内容

### 移除未使用的 ExperimentDesignAgent 引用

**第33行** - 移除导入
```python
# 修改前
from core import (
    ...
    ExperimentDesignAgent,  # Deprecated PydanticAI version
    ...
)

# 修改后
from core import (
    ...
    # ExperimentDesignAgent 已移除 - 直接在 /api/experiment_chat 中使用 Approach 2
    ...
)
```

**第53行** - 移除注释掉的初始化
```python
# 修改前
# experiment_agent = ExperimentDesignAgent()  # Deprecated PydanticAI version

# 修改后
# 完全移除此行
```

**第897行** - 更新注释
```python
# 修改前
# experiment_agent.set_pdf_path(session_id, path)

# 修改后
# Note: experiment_agent removed - using Approach 2 directly in /api/experiment_chat
```

**第931行** - 更新注释
```python
# 修改前
# experiment_agent.submit_response(request_id, response)

# 修改后
# Note: experiment_agent removed - using Approach 2 directly in /api/experiment_chat
```

## 当前实验设计实现

### /api/experiment_chat 路由（第803-870行）

```python
@app.route('/api/experiment_chat', methods=['POST'])
def experiment_chat():
    """实验设计对话 - 使用自然语言生成实验设计JSON"""
    
    # 直接导入并使用 Approach 2
    from core.field_inference import ExperimentDesignAgent
    from experiment.format import ExperimentFormatConverter
    
    agent = ExperimentDesignAgent()  # 每次请求创建新实例
    converter = ExperimentFormatConverter()
    
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

## 架构说明

### 实验设计的两种使用方式

**方式1：直接使用（当前 app.py 使用）**
- 路由：`/api/experiment_chat`
- 实现：每次请求创建 `ExperimentDesignAgent` 实例
- 优点：无状态，简单直接
- 适用：单次实验设计生成

**方式2：交互式使用（可选，未启用）**
- 文件：`core/experiment_agent.py`
- 实现：维护会话状态，支持多轮对话
- 优点：支持 PDF 关联、对话历史
- 适用：需要多轮交互的复杂场景

## 验证结果

✅ **app.py 导入成功**  
✅ **ExperimentDesignAgent 引用已完全移除**  
✅ **实验设计路由正常工作**（使用 Approach 2）  
✅ **无未使用的导入**  
✅ **注释已更新说明**

## 总结

成功清理了 `app.py` 中未使用的 `ExperimentDesignAgent` 引用：
- 移除了导入语句
- 移除了注释掉的初始化代码
- 更新了相关注释说明
- 保持了 `/api/experiment_chat` 路由的正常功能（直接使用 Approach 2）

`core/experiment_agent.py` 作为可选的交互式版本保留，但不在 `app.py` 中引用。
