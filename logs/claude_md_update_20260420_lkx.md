# CLAUDE.md 更新日志 - 硬件工具注册表架构

**日期**: 2026-04-20  
**作者**: lkx  
**版本**: v2.0  
**类型**: 文档更新

---

## 📝 更新概述

更新 CLAUDE.md 以反映硬件工具模块的注册表架构重构（v2.0），添加了注册表系统的说明、使用方法和关键注意事项。

---

## 🔄 变更内容

### 1. 目录结构更新

**位置**: `Directory Structure` 节

**变更**:
```diff
- `hardware/` - MQTT communication, hardware tools, visualization
+ `hardware/` - MQTT communication, hardware tools, visualization
+   - `hardware/tools/` - Hardware tool registry system (decorator-based auto-discovery)
```

**原因**: 新增了 `hardware/tools/` 子目录，采用注册表架构

---

### 2. 关键文件更新

**位置**: `Key Files` 节

**变更**:
```diff
- `core/hardware_controller.py` - MQTT-based hardware control, tool execution
+ `core/hardware_controller.py` - Hardware control, uses registry from hardware/tools

- `hardware/tools.py` - Hardware tool functions (spin coating, temperature, etc.)
+ `hardware/tools/` - Hardware tool registry (decorator-based), individual tool modules
+ `hardware/tools/registry.py` - Tool registry core (ToolRegistry, @register_tool decorator)
+ `hardware/tools/registry.json` - Exported tool metadata (run `python export_registry.py` to update)
+ `hardware/tools.py` - PydanticAI async tools (read_pdf, save_experiment_step, etc.)
+ `export_registry.py` - Export hardware tool registry to JSON (run after tool changes)
```

**原因**: 
- 工具从单一文件拆分为独立模块
- 添加注册表核心文件和导出脚本
- 明确 `tools.py` 现在只用于 PydanticAI 异步工具

---

### 3. 关键注意事项更新

**位置**: `Gotchas > Critical` 节

**变更**:
```diff
+ **Hardware tool registry** - After modifying tools in `hardware/tools/`, run `python export_registry.py` to update `registry.json`
+ **Deps class location** - `Deps` class defined in `hardware/tools/__init__.py` to avoid circular import with `hardware/__init__.py`
```

**原因**:
- 用户需要知道修改工具后要手动导出注册表
- 说明 `Deps` 类的位置，避免导入问题

---

### 4. 新增硬件工具注册表专节

**位置**: 新增 `Hardware Tool Registry` 节（在 `Codebase` 之后）

**内容**:

#### 架构说明
- 装饰器注册模式
- 单例注册表管理
- 控制器使用注册表
- JSON 导出机制

#### 添加新工具流程
1. 创建工具文件
2. 在 `__init__.py` 导入
3. 运行导出脚本
4. 重启应用

#### 工具定义模式
```python
from .registry import register_tool

@register_tool(
    name="tool_name",
    description="Tool description for LLM",
    params={
        "param1": {"type": "int", "description": "...", "required": True}
    }
)
def execute_tool_name(param1: int) -> str:
    return "result"
```

#### 相关文件
- `hardware/tools/registry.py` - 核心实现
- `hardware/tools/registry.json` - 导出的元数据
- `hardware/tools/README.md` - 使用指南
- `export_registry.py` - 导出脚本

---

## 📊 影响范围

### 受影响的文档节
1. ✅ `Directory Structure` - 添加 `hardware/tools/` 说明
2. ✅ `Key Files` - 更新硬件工具相关文件
3. ✅ `Gotchas > Critical` - 添加注册表和 Deps 类注意事项
4. ✅ 新增 `Hardware Tool Registry` - 完整的注册表系统说明

### 未修改的节
- `Quick Start` - 启动流程不变
- `Architecture` - 整体架构不变
- `Configuration` - 配置方式不变
- `Testing` - 测试方法不变

---

## 🎯 更新目的

### 帮助未来的 Claude 会话

1. **理解新架构** - 知道硬件工具采用注册表模式
2. **添加工具** - 清楚添加新工具的步骤
3. **避免问题** - 了解导出注册表和 Deps 类位置
4. **查找文件** - 知道注册表相关文件的位置

### 关键学习点

1. **装饰器注册模式** - 使用 `@register_tool` 自动注册
2. **手动导出** - 修改后运行 `export_registry.py`，而非自动导出
3. **循环导入问题** - `Deps` 类在 `hardware/tools/__init__.py` 中定义
4. **元数据驱动** - 工具定义包含 LLM 可理解的描述和参数

---

## 📚 相关文档

### 硬件工具文档
- `hardware/tools/README.md` - 简明使用说明
- `hardware/tools/QUICK_REFERENCE.md` - 快速参考
- `hardware/README.md` - 完整硬件模块文档
- `EXPORT_REGISTRY_GUIDE.md` - 详细导出指南

### 重构日志
- `logs/hardware_refactor_20260419_lkx.md` - 详细重构日志
- `logs/HARDWARE_REFACTOR_SUMMARY.md` - 重构总结
- `logs/FINAL_SUMMARY_20260419.md` - 最终总结

---

## ✅ 验证清单

- [x] 目录结构更新
- [x] 关键文件列表更新
- [x] 关键注意事项添加
- [x] 新增注册表专节
- [x] 代码示例正确
- [x] 文件路径准确
- [x] 格式一致

---

## 📝 后续建议

1. **定期更新** - 当注册表系统有重大变更时更新此节
2. **添加示例** - 如果有常见的工具添加场景，可以添加更多示例
3. **故障排除** - 如果发现常见问题，可以添加到 Gotchas 节

---

**更新完成**: 2026-04-20  
**审核状态**: ✅ 已应用  
**文档版本**: CLAUDE.md v2.0
