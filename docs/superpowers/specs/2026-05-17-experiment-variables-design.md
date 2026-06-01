# 实验设计变量系统 — 设计规格

**日期**: 2026-05-17
**状态**: 设计中

---

## 概述

为实验设计系统添加变量支持。所有步骤参数可使用变量名代替硬编码字面量。用户在参数框输入非数字值时自动检测，通过变量栏管理变量（声明、默认值、约束），支持 CSV 批量导入与批量执行。

---

## 一、数据模型

### 1.1 变量定义

```
VariableDefinition {
    name: string                   // 唯一标识，如 "speed1"、"temp_a"
    type: "int" | "float" | "str" | "bool"
    default_value: number | string | boolean
    constraints?: {
        min?: number               // 数值最小值
        max?: number               // 数值最大值
        required?: boolean         // 是否必填（CSV行必须提供）
        options?: string[]         // 枚举选项（字符串变量）
    }
}
```

- 默认值必须是整数，不允许小数
- 约束为空时仅做类型校验

### 1.2 步骤参数中的引用

使用变量名（无 `$` 前缀）替代字面量：

```json
{
  "type": "tool",
  "name": "spin_coating",
  "params": {
    "spin_speed": "speed1",
    "spin_dur": "duration * 1000",
    "reagent": "reagent_name",
    "spin_acc": 500,
    "volume": 60
  }
}
```

- `"speed1"` → 直接引用变量
- `"duration * 1000"` → 表达式引用（含运算符时）
- `500` → 字面量，与现有逻辑一致

### 1.3 实验 JSON 顶层新增字段

```json
{
  "experiment_name": "...",
  "variables": {
    "speed1": {
      "type": "int",
      "default_value": 3000,
      "constraints": { "min": 1000, "max": 6000 }
    },
    "duration": {
      "type": "int",
      "default_value": 30
    },
    "reagent_name": {
      "type": "str",
      "default_value": "Perovskite",
      "constraints": { "required": true }
    }
  },
  "batch_data": [
    { "speed1": 3000, "duration": 30, "reagent_name": "Perovskite" },
    { "speed1": 4000, "duration": 25, "reagent_name": "MAPbI3" }
  ],
  "batch_mode": false,
  "steps": [...]
}
```

- `variables`: 变量声明（名称 → 定义）
- `batch_data`: CSV 导入后的批量数据数组
- `batch_mode`: true 时遍历 batch_data 逐行执行

---

## 二、交互设计

### 2.1 参数输入框 blur 检测

```
用户在参数输入框中输入非数字值 → 光标离开（blur）
  ├─ 值已是已声明变量名 → 输入框正常色 + 内联显示引用指示（如 "→ 3000"）
  └─ 值未被声明 → 输入框变红 + 右侧出现 [声明] 按钮
      用户点击 [声明] → 变量栏新增一行，用户填写默认值和约束
```

### 2.2 变量栏布局

位于 CodeArea（JSON/Python 代码区）上方，ExperimentPage 内全宽，高度紧凑。

```
┌──────────────────────────────────────────────────────────┐
│ 变量   [+添加] [CSV导入] [删除] [CSV导出]    批量模式 □  │
├──────────┬──────────┬──────────┬──────────────────────────┤
│ 名称     │ 默认值    │ 约束      │ 引用步骤                  │
├──────────┼──────────┼──────────┼──────────────────────────┤
│ speed1   │ 3000     │ 1000-6000│ 步骤1: spin_speed        │
│ duration │ ?        │ 必填      │ 步骤1: spin_dur          │
│ reagent  │ "A液"    │ 必填      │ 步骤1: reagent           │
└──────────┴──────────┴──────────┴──────────────────────────┘
```

- 变量多时出现横向滚动条
- **[删除]** 默认灰置，选中某行后亮起，用于删除该变量
- **默认值为空**时显示 `?`
- **引用步骤**列展示哪个步骤的哪个参数引用了该变量

### 2.3 CSV 导入

- CSV 第一行 header 自动成为变量名
- 从第一行数据推断变量类型
- 数据行转为 `batch_data`
- 已存在的同名变量默认值被 CSV 覆盖
- CSV 独有的列自动新增变量；变量栏已有但 CSV 无的列保留

---

## 三、后端架构

### 3.1 新增模块：`core/variable_resolver.py`

```
VariableResolver
├── validate_variables(variables, steps) → ValidationResult
│     检查所有引用的变量是否已声明、类型匹配、约束满足
│
├── resolve(experiment_json) → resolved_json
│     将参数中的变量名替换为默认值，计算表达式
│
├── resolve_batch(experiment_json) → List[resolved_json]
│     遍历 batch_data，每行生成完整 resolved JSON
│
└── evaluate_expression(expr, variables) → value
      ast safe_eval: + - * / // % ** > < >= <= == != and or not ()
```

### 3.2 表达式引擎

- 使用 Python `ast` 模块 safe eval，仅白名单节点
- 不支持函数调用、import、属性访问
- 错误时返回友好提示并推送前端 `reply` 字段

### 3.3 执行流程

```
用户执行实验
  │
  ▼
experiment_json（含 variables + batch_data）
  │
  ▼
VariableResolver.validate_variables()  ← 执行前校验
  │
  ├─ batch_mode=false → resolve() → 单次解析执行
  └─ batch_mode=true  → resolve_batch() → 逐行解析执行
```

### 3.4 API 变更

| API | 变更 |
|---|---|
| `POST /api/experiment_chat` | 返回的 experiment_json 包含 variables 字段 |
| `POST /api/variables/import_csv` | 新增：接收 CSV 文件，返回解析后的变量定义 + batch_data |
| `POST /api/execute_experiment_design` | 执行前走 VariableResolver 校验+解析 |
| `POST /api/compile_experiment` | 编译时处理变量引用 |

### 3.5 错误推送

所有运行时错误（变量未声明、类型不匹配、约束越界、表达式求值失败等）由后端生成错误文本，通过 `reply` 字段推送到前端。前端不构造错误文案。

---

## 四、Prompt 变更

`prompts/experiment_design/_system.yaml` 新增变量声明规则：

- **何时使用变量**：单轮实验用默认值；多轮/优化实验用变量名；用户明确指定时优先；不确定时询问用户
- **输出格式**：顶层新增 `variables` 字段，每项含 type / default_value / constraints
- **步骤引用方式**：params 中直接用变量名（无 `$` 前缀），如 `"spin_speed": "speed1"`
- **变量命名**：英文+数字，语义化，变量名不重复

---

## 五、前端架构

### 5.1 新增组件

| 组件 | 说明 |
|------|------|
| `VariableBar.vue` | 变量管理栏，位于 CodeArea 上方，ExperimentPage 内全宽 |

### 5.2 修改文件

| 文件 | 变更 |
|------|------|
| `types/experiment.ts` | 新增 VariableDefinition interface |
| `api/experiment.ts` | ExperimentPlan 增加 variables/batch_data/batch_mode |
| `stores/experiment.ts` | 变量 CRUD、批量数据、CSV 导入方法 |
| `pages/ExperimentPage.vue` | 集成 VariableBar |
| `components/experiment/StepEditor.vue` | blur 检测 + [声明] 按钮 |

### 5.3 原则

- 前端仅负责 UI，不构造对用户的回应文本
- 所有 AI 回复、错误说明由后端 `reply` 字段返回

---

## 六、文件变更清单

### 新增

| 文件 | 说明 |
|------|------|
| `core/variable_resolver.py` | 变量解析器 |
| `frontend/src/components/experiment/VariableBar.vue` | 变量栏组件 |

### 修改

| 文件 | 说明 |
|------|------|
| `prompts/experiment_design/_system.yaml` | 变量声明规则 |
| `app.py` | `/api/variables/import_csv`；执行/编译集成 VariableResolver |
| `experiment/executor.py` | 执行前校验+解析；批量模式 |
| `experiment/compiler.py` | 编译时变量引用处理 |
| `experiment/format.py` | variables 字段透传 |
| `frontend/src/types/experiment.ts` | VariableDefinition 类型 |
| `frontend/src/api/experiment.ts` | 接口类型更新 |
| `frontend/src/stores/experiment.ts` | 变量状态管理 |
| `frontend/src/pages/ExperimentPage.vue` | 集成 VariableBar |
| `frontend/src/components/experiment/StepEditor.vue` | blur 检测 + 声明按钮 |
