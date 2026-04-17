# 实验设计模块重构日志

**日期**: 2026-04-17  
**作者**: lkx  
**类型**: 架构重构 + 功能增强

---

## 概述

重构实验设计对话模块，统一JSON格式，移除PydanticAI交互模式，添加JSON与图形化格式的双向转换功能。

---

## 主要改动

### 1. 提示词迁移 (core/config.py → core/field_inference.py)

**变更**:
- 从`Config.EXPERIMENT_AGENT_SYSTEM_PROMPT`移至`ExperimentDesignParser.EXPERIMENT_AGENT_SYSTEM_PROMPT`
- 更新`experiment_agent.py`引用新位置

**原因**:
- 字段推断模块更适合管理提示词
- 减少config.py的职责范围
- 提示词与解析器逻辑内聚

**影响文件**:
- `core/config.py` - 移除EXPERIMENT_AGENT_SYSTEM_PROMPT
- `core/field_inference.py` - 新增ExperimentDesignParser类
- `core/experiment_agent.py` - 更新导入

---

### 2. 统一实验设计JSON格式

**新格式**:
```json
{
  "experiment_name": "实验名称",
  "description": "实验描述",
  "steps": [
    {
      "type": "tool",           // "tool" 或 "helper"
      "name": "spin_coating",   // 操作名称
      "params": {...},          // 参数
      "description": "步骤描述"
    },
    {
      "type": "helper",
      "name": "WAIT",
      "params": {"duration": 5000},
      "description": "等待5秒"
    }
  ],
  "notes": "注意事项",
  "created_at": "2026-04-17T..."
}
```

**兼容性**:
- 执行器同时支持旧格式`action`字段和新格式`type+name`字段
- 前端自动转换为统一格式

---

### 3. 文件重命名

**变更**: `core/experiment_executor.py` → `core/experiment_manager.py`

**原因**:
- 不仅执行实验，还负责格式转换和验证
- "Manager"更准确描述其职责范围

**类名**: `ExperimentExecutor` → `ExperimentManager`

---

### 4. 新增格式转换功能 (core/experiment_manager.py)

#### 4.1 JSON → 图形化格式

**方法**: `json_to_visual(experiment_json) -> dict`

**输入**: 标准实验JSON
**输出**: 前端可视化格式
```json
{
  "nodes": [
    {
      "id": "node_1",
      "type": "spin_coating",
      "label": "旋涂",
      "params": {...},
      "description": "..."
    }
  ],
  "edges": [
    {"from": "node_1", "to": "node_2"}
  ]
}
```

**特性**:
- 自动生成节点ID
- 根据操作类型生成中文标签
- 构建节点间的连接关系

#### 4.2 图形化格式 → JSON

**方法**: `visual_to_json(visual_data) -> dict`

**输入**: 前端可视化格式（nodes + edges）
**输出**: 标准实验JSON

**特性**:
- 拓扑排序确定执行顺序
- 处理环检测（回退到原始顺序）
- 自动识别工具操作和辅助操作

---

### 5. 简化API路由 (app.py)

**变更**: `/api/experiment_chat`

**移除**:
- PydanticAI Agent交互模式
- `use_agent_mode`参数
- SSE流式推送逻辑

**保留**:
- 直接JSON生成模式（默认且唯一）

**新增功能**:
1. **控制台打印**:
   ```
   ============================================================
   [实验设计] ✅ 生成成功
   [实验设计] 实验名称: 旋涂实验_v1
   [实验设计] 步骤数量: 3
   
   [实验设计] 完整JSON:
   {...}
   ============================================================
   ```

2. **双格式返回**:
   ```json
   {
     "type": "experiment_design",
     "experiment_json": {...},    // 标准格式
     "visual_data": {...},        // 可视化格式
     "reply": "✅ 已生成...已推送到实验流程画布。"
   }
   ```

---

### 6. 前端推送功能 (templates/index.html)

#### 6.1 响应处理

**位置**: `experiment_chat()`函数

**新增逻辑**:
```javascript
else if (data.type === 'experiment_design') {
    appendMessage(data.reply, 'ai');
    
    if (data.experiment_json) {
        loadExperimentFromJSON(data.experiment_json);
        showNotification('✅ 实验设计已生成并推送到画布', 'success');
    }
}
```

#### 6.2 新增函数

**`loadExperimentFromJSON(json)`**:
- 更新`experimentName`全局变量
- 转换`steps`为前端格式
- 调用`renderExperimentSteps()`渲染画布
- 调用`updateExperimentJSON()`更新代码区

**`showNotification(message, type)`**:
- 显示浮动通知（右上角）
- 支持success/error/info三种类型
- 3秒后自动消失
- 带滑入滑出动画

#### 6.3 CSS动画

**新增**:
```css
@keyframes slideIn {
    from { transform: translateX(400px); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}
@keyframes slideOut {
    from { transform: translateX(0); opacity: 1; }
    to { transform: translateX(400px); opacity: 0; }
}
```

---

### 7. 辅助操作支持

**新增**: `WAIT`辅助操作

**实现** (core/experiment_manager.py):
```python
def _execute_wait(self, params: dict) -> str:
    duration_ms = params.get("duration", 1000)
    duration_s = duration_ms / 1000.0
    time.sleep(duration_s)
    return f"✅ 等待 {duration_s} 秒完成"
```

**用途**: 多步实验间的等待时间

---

### 8. 模块导出更新 (core/__init__.py)

**新增导出**:
- `ExperimentDesignParser` - 实验设计解析器
- `ExperimentManager` - 实验管理器（替代ExperimentExecutor）

---

## 工作流程

### 用户视角

1. 输入: `"实验设计：设计一个三步旋涂实验"`
2. AI生成JSON（后端控制台打印完整JSON）
3. 前端接收JSON并推送到画布
4. 上半部分显示步骤卡片（可拖拽排序）
5. 下半部分显示完整JSON代码
6. 弹出绿色成功通知

### 技术流程

```
用户输入
  ↓
ExperimentDesignParser.parse_experiment_design()
  ↓
生成标准JSON (打印到控制台)
  ↓
ExperimentManager.json_to_visual()
  ↓
返回 {experiment_json, visual_data}
  ↓
前端 loadExperimentFromJSON()
  ↓
更新 experimentSteps + experimentName
  ↓
renderExperimentSteps() (画布)
updateExperimentJSON() (代码区)
  ↓
显示成功通知
```

---

## 兼容性

### 向后兼容

**执行器**:
- 同时支持`action`字段（旧格式）
- 同时支持`type+name`字段（新格式）

**验证器**:
- 自动识别格式类型
- 统一验证逻辑

### 前端兼容

**手动设计**:
- 前端手动添加步骤 → 生成新格式JSON
- 可与AI生成的JSON无缝混合

**JSON编辑**:
- 直接编辑代码区JSON
- 保存后自动同步到画布

---

## 测试建议

### 功能测试

1. **基础生成**:
   - 输入: "实验设计：单步旋涂"
   - 验证: JSON生成、控制台打印、前端推送

2. **多步实验**:
   - 输入: "实验设计：三步旋涂，每步间隔5秒"
   - 验证: WAIT步骤正确插入

3. **格式转换**:
   - 手动添加步骤 → 保存 → 导出JSON
   - 验证: JSON格式正确

4. **混合编辑**:
   - AI生成 → 手动修改 → 执行
   - 验证: 修改生效

### 边界测试

1. **空步骤**: 生成0步骤实验
2. **大量步骤**: 生成20+步骤实验
3. **特殊字符**: 实验名称包含特殊字符
4. **缺失字段**: JSON缺少可选字段

---

## 已知限制

1. **PydanticAI模式已移除**:
   - 不再支持交互式工具调用
   - 不再支持PDF读取工具
   - 如需恢复，参考git历史

2. **图形化格式限制**:
   - 仅支持线性流程（顺序执行）
   - 不支持分支/循环结构
   - 边的拓扑排序假设无环

3. **前端同步**:
   - 代码区手动编辑JSON后需点击"保存"才同步到画布
   - 画布修改自动同步到代码区

---

## 文件清单

### 新增文件
- `core/experiment_manager.py` - 实验管理器（重命名自experiment_executor.py）
- `logs/Experiment_Design_Refactor_20260417.md` - 本文档

### 修改文件
- `core/config.py` - 移除EXPERIMENT_AGENT_SYSTEM_PROMPT
- `core/field_inference.py` - 新增ExperimentDesignParser类
- `core/experiment_agent.py` - 更新提示词引用
- `core/__init__.py` - 更新导出列表
- `app.py` - 简化/api/experiment_chat路由
- `templates/index.html` - 新增loadExperimentFromJSON和showNotification

### 删除文件
- 无（experiment_executor.py通过git mv重命名）

---

## 后续优化建议

1. **提示词优化**:
   - 根据实际使用情况调整ExperimentDesignParser.EXPERIMENT_AGENT_SYSTEM_PROMPT
   - 添加更多实验类型示例

2. **格式验证增强**:
   - 添加JSON Schema验证
   - 前端实时验证JSON语法

3. **可视化增强**:
   - 支持分支结构（条件执行）
   - 支持循环结构（重复步骤）
   - 节点间添加连线动画

4. **错误处理**:
   - 更详细的错误提示
   - 部分失败时的回滚机制

5. **性能优化**:
   - 大量步骤时的虚拟滚动
   - JSON解析缓存

---

## 参考资料

- 原PydanticAI实现: `git log --grep="实验设计"`
- 格式转换算法: 拓扑排序（Kahn算法）
- 前端框架: 原生JavaScript + CSS Grid

---

**变更总结**: 本次重构简化了实验设计流程，统一了数据格式，增强了前后端交互体验，为后续功能扩展奠定了基础。
