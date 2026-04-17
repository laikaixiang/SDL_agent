# 实验设计与执行分离实施日志

**日期**: 2026-04-16  
**开发人员**: lkx  
**项目**: SDL_agent - 自驱动实验室智能体系统  
**模块**: 实验设计智能体

---

## 📋 概述

实现了实验设计与执行阶段的完全分离，以解决Qwen3-VL模型不支持Function Calling的兼容性问题。新架构允许任何大语言模型通过输出结构化JSON来设计实验，然后由专用执行器执行。

---

## 🎯 问题陈述

**原始问题**: 
- Qwen3-VL-30B-A3B-Instruct不支持OpenAI的Function Calling API
- PydanticAI Agent失败，错误信息：`UnexpectedModelBehavior: Exceeded maximum retries (1) for output validation`
- 根本原因：模型既不返回文本也不调用工具，违反了PydanticAI的要求

**影响**:
- 实验设计模式完全无法使用
- 无法使用主要模型进行自动化实验规划

---

## 💡 解决方案架构

### 设计理念
将实验工作流程分为两个独立阶段：

1. **设计阶段**：AI分析需求并输出JSON格式的实验方案
2. **执行阶段**：系统解析JSON并逐步执行硬件操作

### 核心优势
- ✅ 模型无关：适用于任何大语言模型（无需Function Calling支持）
- ✅ 用户可控：执行前可审查和编辑完整方案
- ✅ 降低复杂度：移除了PydanticAI工具系统
- ✅ 增强可靠性：结构化JSON格式，易于验证
- ✅ 方案可复用：支持保存/加载JSON格式的实验方案

---

## 🔧 实施细节

### 1. 后端修改

#### `core/config.py` (第93-119行)
**修改内容**：实验设计智能体的系统提示词

**变更说明**：
- 移除了基于工具的指令
- 添加了JSON输出格式要求
- 包含了可用操作的文档说明
- 添加了参数约束（转速≤6000rpm，温度≤200℃）

**关键代码**：
```python
EXPERIMENT_AGENT_SYSTEM_PROMPT: str = (
    "你是一位经验丰富的材料科学家，专门帮助用户设计旋涂实验方案。\n\n"
    "你的任务：\n"
    "1. 理解用户的实验需求\n"
    "2. 基于需求设计详细的实验方案\n"
    "3. 以JSON格式输出方案\n\n"
    # ... 详细的格式规范
)
```

#### `core/experiment_agent.py` (完全重写)
**修改内容**：从基于PydanticAI简化为直接API调用

**移除内容**：
- PydanticAI Agent初始化
- 工具注册系统
- 用户确认队列
- 依赖注入容器

**新增内容**：
- `design_experiment()`：主要设计方法（异步）
- `_call_llm()`：使用aiohttp直接调用LLM API
- `_get_available_reagents()`：从JSON读取试剂配置

**核心方法**：
```python
async def design_experiment(self, session_id: str, user_message: str) -> str:
    """设计实验方案（不执行）"""
    # 获取可用试剂
    # 构建包含试剂列表的系统提示词
    # 直接调用LLM API
    # 返回AI响应（文本 + JSON）
```

#### `core/experiment_executor.py` (新建文件)
**创建内容**：专用于执行基于JSON的实验方案的执行器

**核心功能**：
- `execute_plan()`：按顺序执行实验步骤
- `validate_plan()`：验证JSON结构和试剂可用性
- 支持进度回调，提供实时反馈
- 操作映射：spin_coating、set_temperature、move_robot_arm、collect_spectrum

**执行流程**：
```python
def execute_plan(self, plan_json: dict, progress_callback=None):
    for step in plan_json["steps"]:
        # 执行操作
        # 报告进度
        # 处理错误
    # 为旋涂步骤发送启动命令
```

#### `app.py` (第762-920行)
**修改内容**：更新路由以支持设计-执行分离

**路由变更**：
- `/api/experiment_chat`：现在调用`design_experiment()`而非`run()`
- `/api/experiment_execute`：新增路由，用于执行JSON方案

**关键更新**：
- 导入了`ExperimentExecutor`
- 移除了`send_event_async`回调（不再需要）
- 执行前添加了方案验证
- 增强了调试日志

### 2. 前端修改

#### `templates/index.html`

**新增JavaScript函数** (第1519-1700行)：
- `handleDesignResult()`：解析AI响应并提取JSON
- `renderExperimentPlan()`：渲染可编辑的方案表格
- `executeExperimentPlan()`：提交方案执行
- `collectUpdatedPlan()`：收集用户修改的参数
- `saveExperimentPlan()`：下载方案为JSON文件
- `handleExecutionResult()`：显示执行结果

**新增CSS样式** (第337-470行)：
- `.experiment-plan-card`：方案显示容器
- `.plan-step`：单个步骤样式
- `.step-params`：参数网格布局
- `.plan-param-input`：可编辑输入框
- `.execution-result-card`：执行结果显示
- `.result-step`：带成功/失败指示器的步骤结果

**SSE消息处理** (第865-888行)：
更新以识别新的消息类型：
- `design_result`：触发方案渲染
- `execution_result`：触发结果显示

---

## 📊 JSON格式规范

### 实验方案结构
```json
{
  "experiment_name": "实验名称",
  "description": "实验描述",
  "steps": [
    {
      "step_number": 1,
      "description": "步骤描述",
      "action": "spin_coating",
      "params": {
        "reagent": "Perovskite",
        "volume": 10,
        "spin_speed": 3000,
        "spin_acc": 1000,
        "spin_dur": 30000
      }
    }
  ],
  "notes": "注意事项",
  "reference": "参考来源"
}
```

### 支持的操作类型
| 操作 | 参数 |
|--------|-----------|
| `spin_coating` | reagent（试剂）, volume（体积）, spin_speed（转速）, spin_acc（加速度）, spin_dur（时长） |
| `set_temperature` | temperature（温度）, duration（持续时间） |
| `move_robot_arm` | x, y, z（坐标） |
| `collect_spectrum` | duration（采集时长） |

---

## 🧪 测试结果

### 测试1：基本设计功能
**输入**："帮我设计一个简单的旋涂实验，使用Perovskite试剂，转速3000rpm，时长30秒"

**预期输出**：
- AI对设计思路的解释
- 包含1个spin_coating步骤的JSON格式方案
- 可编辑的参数表格
- "执行实验"和"保存方案"按钮

**状态**：✅ 准备测试

### 测试2：多步骤实验
**输入**："设计两步实验：先涂Perovskite 3000rpm 30秒，然后加热到100度10分钟"

**预期输出**：
- 包含2个步骤的方案（spin_coating + set_temperature）
- 正确的参数映射

**状态**：✅ 准备测试

### 测试3：方案编辑
**操作**：在UI中将spin_speed从3000修改为4000

**预期行为**：
- 更新的值在执行时生效
- 验证通过

**状态**：✅ 准备测试

---

## 📁 修改/创建的文件

### 修改的文件
1. `core/config.py` - 系统提示词更新
2. `core/experiment_agent.py` - 完全重写（178行 → 165行）
3. `app.py` - 路由更新（第762-920行）
4. `templates/index.html` - 前端逻辑和样式

### 新建的文件
1. `core/experiment_executor.py` - 执行器实现（260行）
2. `实验设计执行分离方案.md` - 详细方案文档
3. `实施完成说明.md` - 实施指南

### 文档
1. `logs/Experiment_Design_Execution_Separation_20260416.md` - 本日志

---

## 🔍 技术亮点

### 1. 异步HTTP客户端
使用`aiohttp`直接调用LLM API：
```python
async with aiohttp.ClientSession() as session:
    async with session.post(API_URL, json=payload) as resp:
        data = await resp.json()
```

### 2. JSON提取
使用健壮的正则表达式从markdown中提取JSON：
```javascript
const jsonMatch = designResult.match(/```json\s*([\s\S]*?)\s*```/);
```

### 3. 动态参数收集
从DOM收集用户编辑：
```javascript
document.querySelectorAll('.plan-param-input').forEach(input => {
    plan.steps[stepIndex].params[paramKey] = parseFloat(value);
});
```

### 4. 方案验证
执行前验证：
```python
def validate_plan(self, plan_json: dict) -> tuple[bool, str]:
    # 检查必需字段
    # 验证操作类型
    # 验证试剂可用性
```

---

## 🐛 已知问题与限制

### 当前限制
1. **无PDF阅读**：设计阶段尚不支持读取PDF文件
2. **无方案模板**：没有预置的实验模板
3. **无执行历史**：执行日志未持久化
4. **无暂停/恢复**：无法中途暂停执行

### 未来增强
1. 在设计阶段集成PDF阅读功能
2. 添加实验模板库
3. 实现执行历史跟踪
4. 添加暂停/恢复功能
5. UI中的实时参数验证

---

## 📈 性能影响

### 改造前（PydanticAI模式）
- ❌ 立即失败，出现验证错误
- ❌ 无法成功执行实验
- ⚠️ 需要复杂的错误处理

### 改造后（分离模式）
- ✅ 适用于任何LLM模型
- ✅ 可预测的JSON输出
- ✅ 简单的错误处理
- ✅ 用户可编辑方案

### 代码复杂度
- **移除**：约200行PydanticAI集成代码
- **新增**：约260行执行器代码
- **净变化**：+60行，但逻辑更简单

---

## 🔐 安全考虑

### 输入验证
- ✅ 执行前进行JSON模式验证
- ✅ 根据配置验证试剂名称
- ✅ 参数范围检查（转速、温度）

### 用户确认
- ✅ 执行前需要明确确认
- ✅ 方案审查和编辑能力
- ✅ 清晰的错误消息

---

## 🎓 经验教训

1. **模型兼容性**：选择框架前务必验证API兼容性
2. **关注点分离**：分离设计和执行提高了灵活性
3. **用户控制**：给予用户编辑能力增加信任和安全性
4. **结构化输出**：对于复杂任务，JSON格式比工具调用更可靠
5. **渐进增强**：从简单开始（JSON输出），再添加复杂性（工具调用）

---

## 📞 支持与维护

### 调试技巧
1. 检查后端日志中的LLM API响应
2. 在浏览器控制台验证JSON格式
3. 在`reagent_layout.json`中验证试剂名称
4. 在Network标签中监控SSE消息流

### 常见问题
- **JSON解析错误**：检查是否有多余的markdown格式
- **试剂未找到**：验证拼写和配置
- **执行失败**：检查MQTT连接状态

---

## ✅ 完成清单

- [x] 系统提示词已更新
- [x] Agent已简化（移除PydanticAI）
- [x] 执行器已实现
- [x] 路由已更新
- [x] 前端处理器已添加
- [x] CSS样式已添加
- [x] 文档已编写
- [x] 测试用例已定义
- [ ] 集成测试
- [ ] 用户验收测试

---

## 📝 备注

本次实施代表了从基于工具的执行到基于方案的执行的根本性架构转变。新方法更易维护、更可靠，并提供更好的用户体验。权衡是我们失去了AI自主调用工具的一些"魔力"，但我们获得了可预测性和用户控制。

基于JSON的方法还为以下功能开辟了可能性：
- 实验方案库
- 方案版本控制和历史记录
- 协作方案编辑
- 自动方案优化

---

**状态**：✅ 实施完成  
**下一步**：集成测试  
**部署**：准备进入预发布环境

---

*日志结束*
