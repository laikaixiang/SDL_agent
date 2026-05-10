# Prompts — 集中管理与优化

本项目所有 LLM prompt 的统一管理中心。16 个 prompt 覆盖文献提取、实验设计、硬件控制、数据分析、算法生成 5 个业务模块。配套的提取质量检查器 `extract/quality_checker.py` 提供确定性规则检测（稀疏记录 + 重复记录），无需额外 LLM 调用。

## 目录结构

```
prompts/
├── README.md                       ← 本文件
├── registry.yaml                   ← 索引：name → 文件路径 + 分类
├── manager.py                      ← PromptManager：加载 / 渲染 / 修改 / 热更新
├── optimizer.py                    ← PromptOptimizer：LLM 优化辅助
├── api.py                          ← Flask Blueprint（7 个 REST 接口）
├── __init__.py                     ← 导出 + 全局单例工厂
│
├── extraction/                     ← 文献提取
│   ├── _system_vision.yaml         ← 图片模式提取
│   ├── _system_text.yaml           ← 文本模式提取
│   └── _few_shot_block.yaml        ← 历史示例注入块
│
├── experiment_design/              ← 实验设计
│   ├── _system.yaml                ← 系统提示词（动态注入工具列表）
│   └── _user.yaml                  ← 用户需求包装
│
├── data_analysis/                  ← 数据分析
│   ├── _system.yaml                ← 算法选择 + 读取方式决策
│   └── _user.yaml                  ← CSV 信息 + 算法列表模板
│
├── algorithm_gen/                  ← 算法代码生成
│   ├── _user_guidance.yaml         ← 引导用户描述算法需求
│   ├── _spec_extraction.yaml       ← NL → 算法规格 JSON
│   ├── _code_gen_system.yaml       ← 代码生成系统提示词
│   └── _code_gen_template.yaml     ← 代码生成模板（接口规范 + 占位符）
│
├── field_inference/                ← 字段推断
│   ├── _infer_fields.yaml          ← 从任务描述推断提取字段
│   └── _filename_prefix.yaml       ← 任务 → 英文文件名前缀
│
├── hardware/                       ← 硬件控制
│   └── _command_parse.yaml         ← NL 命令 → 工具调用 JSON（安全关键）
│
├── misc/                           ← 其他
│   └── _session_title.yaml         ← 对话 → 会话标题
│
├── optimizer/                      ← 元 prompt
│   └── _meta_optimize.yaml         ← 用 LLM 优化其他 prompt 的 prompt
│
├── overrides/                      ← 运行时修改（gitignore，不提交）
│
└── reviewer_design.md              ← 提取审查功能设计文档

extract/
└── quality_checker.py              ← 确定性质量检测器（稀疏 + 重复检测）

platform_init/test/prompt/          ← 测试脚本
    ├── test_migration.py           ← 迁移验证（9 tests）
    └── test_evaluation.py          ← 量化评测（30 tests + 4 业务评测套件）
    └── test_quality_checker.py     ← 质量检测器测试（65 tests）
```

---

## Prompt 文件格式

每个 prompt 是一个独立的 YAML 文件，包含 4 个必填字段：

```yaml
# prompts/extraction/_system_vision.yaml
name: extraction_system_vision          # 唯一标识，与 registry.yaml 中的 key 一致
description: "PDF 页面图片提取的系统提示词"  # 给人看，说明用途
variables: [task_description, fields, example_json]  # 模板需要的变量列表
template: |                             # 模板正文，使用 ${var} 占位
  你是一个钙钛矿太阳能电池领域的文献数据提取专家。
  需要提取的字段：${fields}
  必须严格遵循此格式：${example_json}
```

**模板语法**：Python `string.Template` 原生语法 `${variable}`。这与 Python f-string 的 `{var}` 不同——因为 prompt 中经常包含 JSON 示例的 `{}`，用 `${}` 可以避免冲突。

**变量校验**：`variables` 列表不是装饰。调用 `pm.get(name, ...)` 时，PromptManager 会校验所有声明的变量是否已提供，缺一个就抛异常。

---

## 如何读取 Prompt（业务代码中）

**之前**：prompt 散落在各文件中作为内联 f-string 或模块常量：
```python
sys_prompt = f"你是一个专家。提取：{fields}。格式：{example_json}"
```

**现在**：通过 PromptManager 获取：
```python
from prompts import create_prompt_manager
pm = create_prompt_manager()
sys_prompt = pm.get("extraction_system_vision",
    task_description=task_description,
    fields=str(fields),
    example_json=example_json,
)
```

`create_prompt_manager()` 返回全局单例，首次调用时自动加载所有 YAML 文件。

---

## 如何修改 Prompt（三种方式）

### 方式 A：直接改 YAML 源文件（推荐，用于永久性优化）

```bash
vim prompts/extraction/_system_vision.yaml  # 改 template 字段
```

改完重启 Flask 或调一次 `POST /api/prompts/reload`，立即生效。

### 方式 B：通过 API 运行时覆盖（不修改源文件）

```bash
# 修改单个 prompt，写入 overrides/ 目录
curl -X PUT http://127.0.0.1:5000/api/prompts/extraction_system_vision \
  -H 'Content-Type: application/json' \
  -d '{"template": "新的模板文本，变量仍用 ${task_description} ${fields}"}'

# 查看修改效果（对比原始和当前）
curl http://127.0.0.1:5000/api/prompts/extraction_system_vision

# 撤销修改，回到源文件
curl -X POST http://127.0.0.1:5000/api/prompts/extraction_system_vision/reset
```

覆盖层机制：
- 源文件（`extraction/_system_vision.yaml`）永远不动
- 修改写入 `overrides/extraction/_system_vision.yaml`（仅存被修改的字段）
- 加载时源文件 + overrides 字段级合并
- `overrides/` 目录已加入 `.gitignore`，不会被提交

### 方式 C：通过代码调用

```python
from prompts import create_prompt_manager
pm = create_prompt_manager()

# 修改
pm.update("extraction_system_vision", template="新的模板...")
pm.update("extraction_system_vision", variables=["a", "b", "c"])

# 重置
pm.reset("extraction_system_vision")

# 全部重载（清所有 overrides）
pm.reload()
```

---

## 如何新增 Prompt

1. 在对应 category 子目录创建 `_your_name.yaml`：
   ```yaml
   name: your_prompt_name
   description: "用途说明"
   variables: [var1, var2]
   template: |
     这是模板正文，使用 ${var1} 和 ${var2}。
   ```

2. 在 `registry.yaml` 注册：
   ```yaml
   prompts:
     # ... 已有条目 ...
     your_prompt_name:
       file: category/_your_name.yaml
       category: your_category
       enabled: true
   ```

3. 业务代码中调用：
   ```python
   pm.get("your_prompt_name", var1="hello", var2="world")
   ```

---

## API 参考

所有接口返回 JSON，格式 `{"success": true/false, "data": ..., "error": "..."}`。

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/prompts` | GET | 列出所有 prompt 元信息。`?category=extraction` 过滤 |
| `/api/prompts/<name>` | GET | 单个 prompt 详情（当前模板 / 原始模板 / 是否被覆盖） |
| `/api/prompts/<name>` | PUT | 修改 prompt。body: `{"template": "...", "variables": [...], "description": "..."}` 三字段均可选 |
| `/api/prompts/<name>/reset` | POST | 撤销修改，回到源文件 |
| `/api/prompts/reload` | POST | 清空所有 overrides，重新加载 |
| `/api/prompts/optimize` | POST | LLM 优化建议。body: `{"name": "...", "requirements": "...", "test_inputs": [...]}` |
| `/api/prompts/test` | POST | 用测试输入跑一次 prompt。body: `{"name": "...", "variables": {...}, "user_content": "..."}` |
| `/api/page_preview` | GET | PDF 页面预览 + 关键词高亮。`?doc=...&page=...&query=...` |
| `/api/page_context` | POST | Agent 批量阅读 PDF 原文。body: `{"results": [{"doc":..., "page":..., "query":...}]}` |

### 提取结果来源追踪

每条提取结果自动携带来源信息，无需 LLM 输出：

| 字段 | 类型 | 说明 |
|------|------|------|
| `_source_doc` | str | PDF 文件名（已有） |
| `_source_page` | int | PDF 页码，从 1 开始（新增） |

配合 `/api/page_preview` 和 `/api/page_context`，用户/agent 可点击结果查看原始 PDF 页面并高亮匹配行，以此判断提取准确性。

### 提取质量检查

`extract/quality_checker.py` 提供确定性规则检测（不调用 LLM）：

| 检查项 | 规则 | 默认阈值 |
|--------|------|---------|
| 稀疏检测 | 字段填充率 < 阈值 → 删除 | 30%（10 个字段中少于 3 个有值） |
| 重复检测 | 两条记录完全一致或 A 包含 B → 保留信息量大的 | — |

在 extraction_engine 的提取循环结束、保存 CSV 之前自动执行。配置项：`QUALITY_CHECK_ENABLED`、`QUALITY_SPARSE_THRESHOLD`。

---

## 测试

```bash
# 迁移验证（确认所有 prompt 可加载、渲染、修改、重置）
python platform_init/test/prompt/test_migration.py

# 量化评测（离线模式，秒出结果）
python platform_init/test/prompt/test_evaluation.py

# 量化评测（调用 LLM，用于 A/B 对比）
python platform_init/test/prompt/test_evaluation.py --live --all

# 只评测某个 prompt
python platform_init/test/prompt/test_evaluation.py --live --prompt hardware_command_parse

# 质量检测器测试
python platform_init/test/prompt/test_quality_checker.py
```

---

## Prompt 优化记录（2026-05-10）

### 优化原则

| 维度 | 检查点 |
|------|--------|
| 角色设定 | 是否具体？是否匹配任务领域？ |
| 任务清晰度 | 目标是否明确？成功标准是否可度量？ |
| 领域上下文 | 是否包含足够的专业知识？ |
| 输出格式 | JSON 格式约束是否清晰？有无完整示例？ |
| 安全约束 | 可能产生物理影响的 prompt 是否有防护？ |
| Few-Shot | 示例是否覆盖正常/边界/异常场景？ |
| 错误处理 | 是否有歧义兜底策略？ |

### 优化清单

| 优先级 | Prompt | 主要问题 | 优化措施 |
|--------|--------|---------|---------|
| **P0** | `hardware_command_parse` | 无角色/无安全约束/仅2个简单示例 | 安全助手角色 + 3条安全守则 + 危险命令拒绝 + 5个示例（含负例） |
| **P1** | `field_inference_infer_fields` | 仅4句话/无领域上下文/无示例 | 钙钛矿领域上下文 + 4条设计原则 + 2个好的示例 + 2个差的示例 + 命名规范 |
| **P2** | `experiment_design_system` | 设计原则仅5条/无安全约束 | 安全约束层 + 步骤排序逻辑 + 5项自查清单 |
| **P2** | `experiment_design_user` | 纯拼接无引导 | Chain-of-Thought 3个自问自答 |
| **P3** | `extraction_system_vision` | 角色泛化/仅3条规则 | 领域专家角色 + 5条规则 + 3条输出质量原则 |
| **P3** | `extraction_system_text` | 同上 | 同vision版 |
| **P3** | `extraction_few_shot_block` | 无利用示例的指引 | 明确参考要点（粒度/格式/新字段处理） |
| **P4** | `algorithm_gen_code_gen_system` | 全禁令式/无正向指导 | 6条禁令→5项正向质量准则 |
| **P4** | `algorithm_gen_code_gen_template` | 94行/嵌套代码块混乱 | 精简到约70行 + 步骤化(a/b/c/d/e) + 减少嵌套 |
| **P4** | `algorithm_gen_spec_extraction` | 描述不够具体 | 参数描述的因果解释 + type/default 一致性检查 |
| **—** | `data_analysis_system` | 选择原则无优先级 | 标注优先级 + 增加多算法适用性指导 |
| **—** | `field_inference_infer_fields` | LLM 需输出来源字段 | 移除来源字段要求，系统自动附加 `_source_doc` + `_source_page` |

### 提取质量检查（2026-05-10，确定性规则）

| 文件 | 说明 |
|------|------|
| `extract/quality_checker.py` | QualityChecker 类：稀疏检测（字段填充率）+ 重复检测（相等/包含关系） |
| `app.py` | PDF 预览 API：`/api/page_preview` + `/api/page_context` |
| `extract/extraction_engine.py` | `_source_page` 追踪 + QualityChecker 集成 |

### 量化指标

| 层次 | 指标 | 适用 |
|------|------|------|
| L1（格式） | JSON 合法率 / Python 语法正确率 | 所有 prompt |
| L2（结构） | 字段覆盖度 / 工具选择准确率 / 步骤完整性 | 各业务 prompt |
| L3（语义） | 准确率/召回率/F1（需标注数据） | 提取、字段推断 |
| L4（下游） | 端到端任务成功率 | 硬件控制、实验设计 |

---

## Prompt 清单

| # | Name | Category | Variables | 说明 |
|---|------|----------|-----------|------|
| 1 | `extraction_system_vision` | extraction | task_description, fields, example_json | PDF 图片页面的提取系统提示词 |
| 2 | `extraction_system_text` | extraction | task_description, fields, example_json | PDF 文本页面的提取系统提示词 |
| 3 | `extraction_few_shot_block` | extraction | examples_text | 历史提取示例注入到 system prompt 前 |
| 4 | `field_inference_infer_fields` | field_inference | task_description, schema_str | 从任务描述推断需要的提取字段列表。**不再需要 LLM 输出来源字段**，系统自动附加 `_source_doc` + `_source_page` |
| 5 | `field_inference_filename_prefix` | field_inference | task_description | 任务描述→英文文件名前缀 |
| 6 | `experiment_design_system` | experiment_design | hardware_tools_desc, software_tools_desc, helper_tools_desc | 实验设计 Agent 系统提示词（工具列表动态注入） |
| 7 | `experiment_design_user` | experiment_design | system_prompt, user_description | 实验设计用户需求包装 |
| 8 | `hardware_command_parse` | hardware | tools_schema, user_command | NL→硬件工具调用 JSON（安全关键） |
| 9 | `algorithm_gen_user_guidance` | algorithm_gen | (none) | 引导用户如何描述算法需求 |
| 10 | `algorithm_gen_spec_extraction` | algorithm_gen | (none) | NL→结构化算法规格 JSON |
| 11 | `algorithm_gen_code_gen_system` | algorithm_gen | (none) | 代码生成的系统提示词 |
| 12 | `algorithm_gen_code_gen_template` | algorithm_gen | name, description, class_name, input_format, output_fields, params_detail | 代码生成的完整接口模板 |
| 13 | `data_analysis_system` | data_analysis | (none) | 数据分析：算法选择+数据读取方式决策 |
| 14 | `data_analysis_user` | data_analysis | csv_path, columns, algorithms_desc, functions_desc | 数据分析：CSV 信息+算法列表模板 |
| 15 | `misc_session_title` | misc | lines | 对话历史→会话标题（≤20汉字） |
| 16 | `meta_optimize` | optimizer | current_prompt, prompt_name, prompt_description, requirements, test_inputs | 优化其他 prompt 的元 prompt |
