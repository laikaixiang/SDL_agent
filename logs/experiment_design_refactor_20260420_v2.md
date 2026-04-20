# 实验流程生成重构 - 2026/04/20

## 概述

重构实验设计流程，统一使用 `core/experiment_manager.py` 进行 JSON 解析和格式转换，简化前后端交互流程。

## 变更文件

| 文件路径 | 变更类型 | 说明 |
|---------|---------|------|
| `app.py` | 修改 | `/api/experiment_chat` 路由重构，使用 ExperimentManager |
| `templates/static/js/experiment/experiment_chat.js` | 修改 | 更新前端处理逻辑，增强日志和通知 |
| `templates/static/js/experiment/experiment_design.js` | 修改 | `loadExperimentFromJSON` 加载前清空原有数据 |
| `CLAUDE.md` | 更新 | 更新实验设计工作流和架构说明 |

## 重构流程

### 旧流程（已废弃）

```
用户输入 → /api/experiment_chat
         ↓
    experiment_agent.run() (生成 JSON)
         ↓
    ExperimentFormatConverter.json_to_visual() (转换格式)
         ↓
    返回 { experiment_json, visual_data }
         ↓
    前端 loadExperimentFromJSON()
```

### 新流程（当前实现）

```
用户输入 → /api/experiment_chat
         ↓
    experiment_agent.run() (生成 JSON)
         ↓
    ExperimentManager.validate_plan() (验证)
         ↓
    返回 { experiment_json }
         ↓
    前端 loadExperimentFromJSON() (清空 + 加载)
         ↓
    更新画布和 JSON 编辑器
```

## 核心变更

### 1. 后端 - app.py

**变更点：**
- 移除 `ExperimentFormatConverter` 导入
- 使用 `ExperimentManager` 进行验证和管理
- 简化返回格式，只返回 `experiment_json`（不再返回 `visual_data`）
- 前端自行处理 JSON 到画布的渲染

**代码片段：**
```python
from core.experiment_manager import ExperimentManager

experiment_manager = ExperimentManager(software_manager=software_manager)

# 验证实验方案
is_valid, error_msg = experiment_manager.validate_plan(experiment_json)
if not is_valid:
    print(f"[实验设计] ⚠️ 验证警告: {error_msg}")

# 只返回 JSON，不返回 visual_data
return jsonify({
    'type': 'experiment_design',
    'experiment_json': experiment_json,
    'reply': result_text
})
```

### 2. 前端 - experiment_chat.js

**变更点：**
- 增强日志输出，便于调试
- 改进通知消息，显示实验名称和步骤数量
- 统一错误处理，所有错误都显示通知

**代码片段：**
```javascript
if (data.experiment_json) {
    console.log('[ExperimentChat] 收到实验设计 JSON');
    console.log('[ExperimentChat] 实验名称:', data.experiment_json.experiment_name);
    console.log('[ExperimentChat] 步骤数量:', data.experiment_json.steps?.length || 0);

    // 加载实验设计到画布和 JSON 编辑器
    loadExperimentFromJSON(data.experiment_json);

    // 显示成功通知
    const stepCount = data.experiment_json.steps?.length || 0;
    const expName = data.experiment_json.experiment_name || '未命名实验';
    showNotification(`✅ 实验设计已生成：${expName}（${stepCount} 个步骤）`, 'success');
}
```

### 3. 前端 - experiment_design.js

**变更点：**
- `loadExperimentFromJSON()` 加载前清空原有数据
- 防止新旧实验数据混合显示
- 增强日志输出

**代码片段：**
```javascript
function loadExperimentFromJSON(json) {
    try {
        // 清空原有数据
        experimentSteps = [];
        experimentName = '未命名实验';

        // 加载新数据
        experimentName  = json.experiment_name || '未命名实验';
        experimentSteps = json.steps.map(step => { /* ... */ });

        // 刷新画布和 JSON 显示
        renderExperimentSteps();
        updateExperimentJSON();

        console.log('[ExperimentDesign] 已加载实验设计:', experimentName);
        console.log('[ExperimentDesign] 步骤数量:', experimentSteps.length);
    } catch (e) {
        console.error('[ExperimentDesign] 加载失败:', e);
        alert('加载实验设计失败: ' + e.message);
    }
}
```

## 架构优势

### 1. 统一管理
- 所有实验相关操作（验证、执行、编译、格式转换）集中在 `ExperimentManager`
- 避免多个模块重复实现相同功能

### 2. 简化流程
- 后端只负责生成和验证 JSON
- 前端负责 JSON 到 UI 的渲染
- 职责分离更清晰

### 3. 易于维护
- 格式转换逻辑集中在一处（`ExperimentManager`）
- 修改格式只需更新一个文件
- `experiment/format.py` 标记为 legacy，未来可移除

### 4. 数据清洁
- 加载新实验前自动清空旧数据
- 避免 UI 显示混乱
- 用户体验更好

## 兼容性

### 保持兼容
- JSON 格式未变更，仍使用统一格式
- 支持旧格式 `action` 字段和新格式 `type+name` 字段
- 前端 `loadExperimentFromJSON()` 兼容两种格式

### 废弃组件
- `experiment/format.py:ExperimentFormatConverter` - 标记为 legacy
- 建议未来迁移到 `ExperimentManager` 的方法

## 测试建议

### 手动测试流程

1. **启动应用**
   ```bash
   python app.py
   ```

2. **测试实验设计生成**
   - 打开浏览器 → 选择"实验设计"模式
   - 输入：`实验设计：制备钙钛矿薄膜，包括旋涂、退火、光谱采集`
   - 验证：
     - 控制台输出完整 JSON
     - 前端画布显示步骤卡片
     - 底部 JSON 编辑器显示代码
     - 显示成功通知

3. **测试数据清空**
   - 生成第一个实验设计
   - 再次输入新的实验描述
   - 验证：旧实验步骤被清空，只显示新实验

4. **测试错误处理**
   - 输入无效描述（如空字符串）
   - 验证：显示错误通知，不影响现有数据

### 验证点

- [ ] 后端正确生成 JSON
- [ ] 后端验证 JSON 结构
- [ ] 前端正确接收 JSON
- [ ] 前端清空旧数据
- [ ] 前端正确渲染画布
- [ ] 前端正确更新 JSON 编辑器
- [ ] 通知消息正确显示
- [ ] 控制台日志完整

## 后续优化建议

1. **移除 legacy 代码**
   - 确认所有调用都迁移到 `ExperimentManager` 后
   - 删除 `experiment/format.py`

2. **增强验证**
   - 在 `ExperimentManager.validate_plan()` 中增加更多检查
   - 验证试剂是否存在
   - 验证算法是否可用

3. **错误恢复**
   - 如果加载失败，保留原有数据
   - 提供"撤销"功能

4. **性能优化**
   - 大型实验方案的渲染优化
   - 虚拟滚动支持

## 相关文档

- `CLAUDE.md` - 项目架构和工作流说明
- `core/experiment_manager.py` - 实验管理器实现
- `core/experiment_agent.py` - 实验设计智能体
- `experiment/COMPILER.md` - 编译器实现细节

## 作者

- laikaixiang
- 2026/04/20
