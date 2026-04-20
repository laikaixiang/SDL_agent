# 实验生成功能修复报告

## 问题描述
用户报告无法生成实验，怀疑是 `ExperimentDesignAgent.EXPERIMENT_AGENT_SYSTEM_PROMPT` 这个 prompt 没有传进去。

## 根本原因
`core/experiment_agent.py` 第60行试图访问 `ExperimentDesignAgent.EXPERIMENT_AGENT_SYSTEM_PROMPT`，但存在以下问题：

1. **命名冲突**：第17行导入了 `from .field_inference import ExperimentDesignAgent`
2. **属性不存在**：`core/field_inference.py` 中的 `ExperimentDesignAgent` 类（Approach 2）没有 `EXPERIMENT_AGENT_SYSTEM_PROMPT` 静态属性
3. **架构混淆**：代码混用了两种实现方式
   - `core/field_inference.py:ExperimentDesignAgent` - Approach 2（JSON + 提示词，使用 `self.system_prompt`）
   - `experiment/agent.py:ExperimentDesignAgent` - Approach 1（PydanticAI，已弃用）

## 修复方案

### 修改文件：`core/experiment_agent.py`

**1. 修复导入（第17行）**
```python
# 修改前
from .field_inference import ExperimentDesignAgent

# 修改后
from .field_inference import ExperimentDesignAgent as FieldInferenceAgent
```

**2. 修复 system_prompt 获取（第49-69行）**
```python
def _create_agent(self) -> Agent:
    """创建 PydanticAI Agent，绑定 API 和实验工具"""
    model = OpenAIChatModel(
        self.config.EXPERIMENT_MODEL_NAME,
        provider=OpenAIProvider(
            base_url=self.config.API_URL.rsplit('/chat/completions', 1)[0],
            api_key=self.config.API_KEY,
        ),
    )

    # 使用 field_inference 中的 ExperimentDesignAgent 生成的 system_prompt
    # 注意：这里创建一个临时实例只是为了获取动态生成的 system_prompt
    field_agent = FieldInferenceAgent()

    return Agent(
        model,
        system_prompt=field_agent.system_prompt,  # 使用动态生成的提示词
        deps_type=Deps,
        tools=[read_pdf, save_experiment_step, start_experiment, get_all_reagents],
    )
```

## 验证测试

### 测试1：FieldInference ExperimentDesignAgent 初始化
```bash
✅ 初始化成功
✅ System prompt 长度: 2280
✅ Hardware registry: 5 个工具
✅ Software registry: 4 个算法
```

### 测试2：ExperimentDesignAgent 初始化
```bash
✅ 初始化成功
✅ Agent 实例创建成功
✅ 配置加载正常: Qwen/Qwen3-VL-30B-A3B-Instruct
```

### 测试3：导入测试
```bash
✅ from core.experiment_agent import ExperimentDesignAgent
✅ from core.field_inference import ExperimentDesignAgent
✅ 无导入错误
```

## 架构说明

### 当前实验设计有两种实现：

**Approach 1（已弃用）** - `experiment/agent.py:ExperimentDesignAgent`
- 基于 PydanticAI Function Calling
- 需要模型支持 Function Calling
- 代码保留用于参考

**Approach 2（当前使用）** - `core/field_inference.py:ExperimentDesignAgent`
- 基于 JSON + 提示词
- 不需要 Function Calling 支持
- 动态生成 system_prompt（从 hardware/tools/REGISTRY.json 和 software/algorithms/ 加载）
- `app.py` 第826行使用此实现

### PydanticAI 交互式实验（可选）
`core/experiment_agent.py:ExperimentDesignAgent` 提供交互式实验设计：
- 支持读取 PDF
- 支持用户确认
- 支持多轮对话
- 使用 Approach 2 的动态生成的 system_prompt

## 相关文件

- `core/experiment_agent.py` - PydanticAI 交互式实验智能体（已修复）
- `core/field_inference.py` - Approach 2 实验设计解析器（无需修改）
- `experiment/agent.py` - Approach 1 实验设计智能体（已弃用，保留参考）
- `app.py` - 使用 Approach 2 生成实验（无需修改）

## 总结

修复了 `core/experiment_agent.py` 中的两个问题：
1. 导入命名冲突
2. 错误的 system_prompt 访问方式

现在 `ExperimentDesignAgent` 可以正确初始化，并使用动态生成的 system_prompt（包含5个硬件工具和4个软件算法的描述）。
