# 实验设计对话功能开发日志

**作者**: lkx  
**日期**: 2026-04-17  
**版本**: v1.0

---

## 📋 功能概述

实现了实验设计对话模式，支持图形化拖拽编辑实验流程，并生成JSON配置文件。

---

## ✨ 新增功能

### 1. 实验设计对话模式
- **入口**: 硬件操控模式 → 实验设计对话
- **布局**: 三栏式布局（对话区35% + 单步控制22.75% + 实验设计画布42.25%）
- **特性**: 
  - 左侧对话区域保持可见，支持AI对话
  - 中间单步控制面板可折叠
  - 右侧实验设计画布支持拖拽排序

### 2. 图形化实验设计
- **添加步骤**: 双击单步控制中的工具，自动添加到右侧画布
- **参数配置**: 使用当前参数值创建步骤
- **拖拽排序**: 支持拖拽步骤卡片调整执行顺序
- **步骤操作**: 
  - ▲ 上移
  - ▼ 下移
  - ✏️ 编辑（JSON格式）
  - 🗑️ 删除

### 3. 辅助函数
在实验设计模式下提供控制结构：
- 🔁 **LOOP**: 循环执行
- 📦 **GROUP**: 步骤分组
- ⏱️ **WAIT**: 延时等待
- ❓ **CONDITION**: 条件判断

### 4. 实验管理
- 💾 **保存设计**: 保存到 `experiment_designs/` 文件夹
- ▶ **执行实验**: 按顺序执行所有步骤
- 🗑️ **清空**: 清空所有步骤
- 📤 **导出JSON**: 下载JSON配置文件

### 5. 实时代码预览
- 下半部分显示JSON代码
- 实时更新实验配置
- 支持复制和导出

---

## 🔧 技术实现

### 前端 (templates/index.html)

#### CSS样式
- `.experiment-design-panel`: 实验设计面板（65%宽度）
- `.step-control-panel`: 单步控制面板（支持折叠）
- `.exp-step-item`: 步骤卡片（支持拖拽）
- `.helper-functions`: 辅助函数栏
- `.exp-code-area`: JSON代码预览区

#### JavaScript功能
- `openExperimentDesignDialog()`: 打开实验设计模式
- `addStepToExperiment(toolName)`: 添加步骤到画布
- `addHelperFunction(fnType)`: 添加辅助函数
- `renderExperimentSteps()`: 渲染步骤列表
- `updateExperimentJSON()`: 更新JSON代码
- `saveExperimentDesign()`: 保存实验设计
- `executeExperimentDesign()`: 执行实验序列
- `exportExperimentJSON()`: 导出JSON文件

#### 拖拽功能
- `dragstart`: 记录拖拽源索引
- `dragover`: 允许放置
- `drop`: 交换步骤顺序

### 后端 (app.py)

#### 新增API路由

**1. `/api/save_experiment_design` (POST)**
```python
# 保存实验设计JSON到文件夹
# 请求体: {"experiment_name": "...", "steps": [...]}
# 返回: {"success": true, "filepath": "..."}
```

**2. `/api/execute_experiment_design` (POST)**
```python
# 执行实验设计JSON中的步骤序列
# 请求体: {"experiment_name": "...", "steps": [...]}
# 返回: {"type": "task_trigger", "reply": "..."}
```

#### 执行逻辑
- 后台线程执行实验序列
- 通过 `task_manager` 推送进度事件
- 支持工具调用和辅助函数（WAIT、LOOP等）
- 错误处理和异常捕获

---

## 🐛 问题修复

### 1. 面板冲突问题
**问题**: 打开单步控制或实验设计时，PDF面板也被触发显示

**原因**: 所有面板共用 `split-mode` CSS类

**解决方案**: 
- 使用独立的CSS类区分不同模式
  - `pdf-mode`: PDF阅读模式
  - `step-control-mode`: 单步控制模式
  - `experiment-design-mode`: 实验设计模式
- 在 `page_reading` 事件中添加模式检查，避免在实验设计模式下打开PDF面板

### 2. 布局遮挡问题
**问题**: 右侧面板覆盖对话区域

**解决方案**: 
- 实验设计模式使用三栏布局
- 单步控制模式使用左右分屏布局
- 通过 `flex` 布局实现自适应

### 3. 折叠按钮显示问题
**问题**: 单步控制单独使用时不需要折叠按钮

**解决方案**: 
- 折叠按钮默认隐藏 (`display: none`)
- 仅在实验设计模式下显示
- 退出实验设计模式时自动隐藏

---

## 📊 JSON格式

### 实验设计JSON结构
```json
{
  "experiment_name": "旋涂实验_v1",
  "created_at": "2026-04-17T12:34:56.789Z",
  "steps": [
    {
      "type": "tool",
      "name": "spin_coating",
      "description": "旋涂实验",
      "params": {
        "spin_speed": 3000,
        "spin_acc": 1000,
        "spin_dur": 30000,
        "reagent": "Perovskite",
        "volume": 10
      }
    },
    {
      "type": "helper",
      "name": "WAIT",
      "description": "等待",
      "params": {
        "duration": 5000
      }
    },
    {
      "type": "helper",
      "name": "LOOP",
      "description": "循环执行",
      "params": {
        "iterations": 3,
        "steps": []
      }
    }
  ]
}
```

---

## 🎯 使用流程

1. 点击 **+** → **硬件操控模式** → **实验设计对话**
2. 左侧对话区输入需求，AI给出建议
3. 中间单步控制面板双击工具添加到右侧画布
4. 调整参数后再次双击添加（使用当前参数值）
5. 使用辅助函数按钮添加LOOP、WAIT等控制结构
6. 拖拽步骤卡片调整顺序，或使用▲▼按钮
7. 下方实时查看JSON代码
8. 点击"保存设计"或"执行实验"

---

## 📁 文件变更

### 修改文件
- `templates/index.html`: 新增实验设计面板UI和交互逻辑
- `app.py`: 新增实验设计保存和执行API

### 新增目录
- `experiment_designs/`: 存储实验设计JSON文件

---

## 🔮 未来优化方向

1. **嵌套循环支持**: 实现LOOP和GROUP的嵌套步骤执行
2. **条件判断**: 实现CONDITION的条件分支逻辑
3. **可视化流程图**: 使用流程图展示实验步骤
4. **模板库**: 预设常用实验模板
5. **版本管理**: 实验设计的版本控制和回滚
6. **实时预览**: 执行前模拟实验流程
7. **参数验证**: 添加参数范围和类型检查
8. **批量导入**: 支持从CSV批量生成实验序列

---

## 📝 备注

- 实验设计模式与AI对话模式可以配合使用
- 所有实验步骤都会经过硬件控制器执行
- JSON文件可以手动编辑后重新导入
- 辅助函数的嵌套执行功能待完善

---

**开发完成时间**: 2026-04-17  
**测试状态**: 待测试  
**部署状态**: 待部署
