# 实验设计方案2重构日志

**日期**: 2026-04-20  
**修改人**: lkx  
**版本标签**: v2.0-experiment-design-refactor

## 概述

将实验设计功能从方案1（PydanticAI Function Calling）重构为方案2（JSON + 提示词），并统一类命名为 `ExperimentDesignAgent`。

## 修改文件清单

| 文件路径 | 修改类型 | 说明 |
|---------|---------|------|
| `core/field_inference.py` | 重构 | 类名从 `ExperimentDesignParser` 改为 `ExperimentDesignAgent`，添加绝对路径支持 |
| `core/__init__.py` | 更新 | 更新导入和导出，注释说明弃用的PydanticAI版本 |
| `app.py` | 更新 | 更新导入和实例化，使用新的类名 |
| `hardware/tools/REGISTRY.json` | 新增 | 硬件工具注册表，包含5个工具定义 |
| `test/experiment_design_v2/test_experiment_design_v2.py` | 新增 | 方案2的完整测试套件 |
| `test/experiment_design_v2/README.md` | 新增 | 测试说明文档 |
| `CLAUDE.md` | 更新 | 翻译中文为英文，更新类名引用 |
| `experiment/agent.py` | 标记 | 添加弃用注释，保留用于参考 |

## 技术变更

### 1. 类命名统一

**变更前**:
- `core/field_inference.py`: `ExperimentDesignParser` (方案2)
- `experiment/agent.py`: `ExperimentDesignAgent` (方案1)

**变更后**:
- `core/field_inference.py`: `ExperimentDesignAgent` (方案2，当前使用)
- `experiment/agent.py`: `ExperimentDesignAgent` (方案1，已弃用)

**原因**: 统一命名，通过文件路径区分不同实现

### 2. 注册表驱动架构

**新增注册表**:
```json
hardware/tools/REGISTRY.json
{
  "spin_coating": {...},
  "set_temperature": {...},
  "move_robot_arm": {...},
  "collect_spectrum": {...},
  "start_experiment": {...}
}
```

**动态加载机制**:
- 硬件工具: 从 `REGISTRY.json` 读取
- 软件算法: 通过 `SoftwareController` 动态扫描
- 辅助操作: 内置在 `ExperimentDesignAgent` 中

**优势**:
- 添加新工具无需修改代码
- 提示词自动更新
- 支持任何LLM（无需Function Calling）

### 3. 路径处理改进

**问题**: 相对路径在子目录运行时失败

**解决方案**:
```python
# 使用绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
registry_path = os.path.join(project_root, "hardware", "tools", "REGISTRY.json")
```

**影响**: 测试可以从任何目录运行

### 4. 文档国际化

**CLAUDE.md 翻译**:
- "方案1/方案2" → "Approach 1/Approach 2"
- "已弃用" → "deprecated"
- "手动维护" → "manually maintained"

**原因**: 统一技术文档语言为英文

## 测试验证

### 测试覆盖

1. **初始化测试**: 验证注册表加载
   - 硬件工具: 5个
   - 软件算法: 4个
   - 辅助操作: 6个

2. **提示词生成测试**: 验证所有工具包含在提示词中
   - 提示词长度: 2333字符
   - 所有工具检查通过

3. **JSON验证测试**: 验证格式检查功能
   - 有效JSON: 通过
   - 无效JSON: 正确拒绝

4. **模拟生成测试**: 验证完整实验流程
   - 包含硬件、软件、辅助操作
   - JSON格式正确

### 运行测试

```bash
cd test/experiment_design_v2
python test_experiment_design_v2.py
```

### 测试结果

```
✓ 硬件工具数量: 5
✓ 软件算法数量: 4
✓ 辅助操作数量: 6
✓ 系统提示词长度: 2333 字符
✓ 所有检查通过
```

## 方案对比

| 特性 | 方案1 (PydanticAI) | 方案2 (JSON + 提示词) |
|------|-------------------|---------------------|
| Function Calling | 必需 | 不需要 |
| 模型兼容性 | 仅OpenAI格式 | 任何LLM |
| 交互式设计 | 支持 | 不支持 |
| 工具扩展 | 修改代码 | 修改JSON |
| 提示词维护 | 自动 | 自动 |
| 当前状态 | 已弃用 | 使用中 |

## 迁移指南

### 从方案1迁移到方案2

1. **更新导入**:
   ```python
   # 旧
   from core import ExperimentDesignAgent  # 方案1
   
   # 新
   from core.field_inference import ExperimentDesignAgent  # 方案2
   ```

2. **API调用**:
   ```python
   # 方案2使用同步调用
   agent = ExperimentDesignAgent()
   success, result = agent.parse_experiment_design(user_description)
   ```

3. **注册表维护**:
   - 添加硬件工具: 编辑 `hardware/tools/REGISTRY.json`
   - 添加软件算法: 在 `software/algorithms/default/` 创建新文件

## 已知问题

无

## 后续工作

1. ~~解耦 `hardware/tools.py`~~ (TODO标记已添加)
2. 考虑为方案2添加PDF读取支持
3. 考虑添加交互式确认功能

## 参考资料

- `CLAUDE.md` - 完整项目文档
- `test/experiment_design_v2/README.md` - 测试说明
- `hardware/tools/REGISTRY.json` - 硬件工具注册表格式
- `software/algorithms/base.py` - 算法基类定义
