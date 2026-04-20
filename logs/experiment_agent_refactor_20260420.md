# core/experiment_agent.py 重构报告

## 重构时间
2026-04-20

## 重构目标
将 `core/experiment_agent.py` 从 PydanticAI 实现（Approach 1）完全重构为 Approach 2（JSON + 提示词）

## 重构前问题

1. **依赖 PydanticAI**：需要模型支持 Function Calling
2. **Prompt 传递失败**：试图访问不存在的 `EXPERIMENT_AGENT_SYSTEM_PROMPT` 属性
3. **架构混淆**：混用两种实现方式

## 重构后架构

### 核心变化

**移除的依赖：**
- ❌ `pydantic_ai` 
- ❌ `OpenAIChatModel`
- ❌ `OpenAIProvider`
- ❌ `Deps` 依赖注入容器
- ❌ `hardware.tools` 的 PydanticAI 工具函数

**新的实现：**
- ✅ 完全使用 `core.field_inference.ExperimentDesignAgent`（Approach 2）
- ✅ 不依赖 Function Calling
- ✅ 支持任何 LLM
- ✅ 保留会话管理和多轮对话功能

### 代码结构

```python
class ExperimentDesignAgent:
    """基于 Approach 2 的交互式版本"""
    
    def __init__(self):
        self.config = Config()
        self._agent = FieldInferenceAgent()  # 使用 Approach 2
        self._sessions = {}
        self._response_queues = {}
    
    async def run(self, session_id, user_message, send_event):
        """使用 Approach 2 生成实验设计"""
        success, result = self._agent.parse_experiment_design(full_input)
        
        if success:
            # 返回实验设计 JSON
            return f"✅ 已生成实验设计方案：{result['experiment_name']}"
        else:
            # 返回错误信息
            return f"❌ 实验设计生成失败：{result}"
```

## 功能保留

✅ **会话管理**：`_sessions` 字典存储每个会话的历史和 PDF 路径  
✅ **多轮对话**：支持对话历史记录  
✅ **PDF 关联**：`set_pdf_path()` 方法  
✅ **用户确认**：`wait_for_response()` 和 `submit_response()` 方法（保留接口）  
✅ **事件推送**：通过 `send_event` 回调推送事件到前端

## 验证测试

### 测试1：初始化测试
```bash
✅ Agent 类型: ExperimentDesignAgent
✅ 使用 Approach 2: True
✅ System prompt 长度: 2280
✅ Hardware registry: 5 个工具
✅ Software registry: 4 个算法
✅ 无 PydanticAI 依赖: 成功
```

### 测试2：依赖检查
```bash
✅ pydantic_ai 引用: 0
✅ PydanticAI 引用: 0
✅ OpenAIChatModel 引用: 0
✅ Deps 引用: 0
```

## 文件变更

- `core/experiment_agent.py` - 完全重写（186行 → 210行）
- `core/experiment_agent_backup.py` - 旧版本备份（PydanticAI 实现）

## 向后兼容性

✅ **API 接口不变**：
- `ExperimentDesignAgent()` 构造函数
- `run(session_id, user_message, send_event)` 方法
- `set_pdf_path(session_id, pdf_path)` 方法
- `clear_session(session_id)` 方法
- `get_active_sessions()` 方法

✅ **现有代码无需修改**：
- `app.py` 中的调用无需改动
- 其他模块的引用保持兼容

## PydanticAI 版本保留

PydanticAI 交互式版本已备份至：
- `core/experiment_agent_backup.py`
- `experiment/agent.py`（原始 Approach 1 实现）

这些文件保留用于参考，但不在任何交互下引用。

## 优势

1. **无 Function Calling 依赖**：支持任何 LLM
2. **统一架构**：完全使用 Approach 2
3. **Prompt 正确传递**：使用动态生成的 system_prompt（2280字符）
4. **更简洁**：移除了复杂的 PydanticAI 工具函数依赖
5. **易于维护**：单一实现路径

## 总结

成功将 `core/experiment_agent.py` 从 PydanticAI 实现重构为完全基于 Approach 2 的实现，解决了 prompt 传递问题，同时保持了所有交互式功能和向后兼容性。
