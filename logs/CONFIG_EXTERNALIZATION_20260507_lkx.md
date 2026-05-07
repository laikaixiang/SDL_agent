# 配置外部化 —— 敏感信息从代码中解耦

**日期**: 2026-05-07
**作者**: lkx
**版本**: v1.0

---

## 1. 背景

此前所有配置参数（包括 API Key、MQTT 密码等敏感信息）都硬编码在 `core/config.py` 中，存在以下问题：

- **安全风险**：API Key 随代码一起提交到 git，泄露风险高
- **环境切换困难**：不同环境（开发/生产）需要修改同一文件，容易冲突
- **新用户上手不便**：需要编辑 Python 代码而非简单的配置文件

## 2. 解决方案

采用 **config.json + 环境变量** 双重配置机制：

```
配置优先级: 环境变量 > config.json > core/config.py 硬编码默认值
```

| 文件 | 用途 | git 追踪 |
|------|------|----------|
| `config.example.json` | 配置模板，含中文注释和占位值 | 追踪 |
| `config.json` | 用户实际配置，含真实 API Key | **不追踪** (.gitignore) |
| `utils/config_loader.py` | 配置加载工具（JSON 读取 + 环境变量合并 + 类型转换） | 追踪 |
| `core/config.py` | 配置类，敏感字段默认为空，启动时从 config.json 加载 | 追踪 |

## 3. 新建文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `config.example.json` | ~55 | 完整配置模板，所有 key 带中文 `_注释` 和占位值 |
| `config.json` | ~40 | 用户实际配置（已从旧 config.py 迁移原值），gitignore 忽略 |
| `utils/config_loader.py` | ~100 | 配置加载器：`_find_project_root()` → `_load_json_config()` → `_merge_env_vars()` |

## 4. 修改文件

| 文件 | 变更 |
|------|------|
| `core/config.py` | 重写：模块加载时读取 config.json，所有属性改为 `_external.get("KEY", default)`；敏感字段默认值置空 |
| `.gitignore` | 新增 `config.json` |

## 5. 关键技术细节

### 5.1 类型保持

JSON 只有 string/number/boolean，而 Config 类属性有 `int`/`float`/`bool`/`str` 类型。通过 `_external.get("KEY", default)` 读取时，如果 config.json 中有值则使用（JSON 解析后的 Python 类型），没有则使用 default（保持原类型）。

环境变量覆盖时也做了自动类型转换（根据 default 的类型）。

### 5.2 `_` 前缀 key 过滤

`config.example.json` 中以 `_` 开头的 key（如 `_API配置`、`_Embedding配置`）作为注释说明，`_load_json_config()` 自动过滤，不传入 Config 类。

### 5.3 向后兼容

- 无 `config.json` 时，Config 使用硬编码默认值（非敏感参数如 URL、模型名等有默认值，敏感参数如 API_KEY 默认为空字符串）
- `Config.get_config()` / `Config.set_config()` / `Config.validate_config()` 接口不变
- 所有现有测试无需修改即可通过

### 5.4 环境变量支持（CI/CD 友好）

```bash
export API_KEY="sk-xxx"
export EMBEDDING_API_KEY="sk-xxx"
export EMBEDDING_MODEL="BAAI/bge-large-en-v1.5"
python app.py
```

同名环境变量自动覆盖 config.json 中的值。

## 6. 新用户使用流程

```bash
git clone <repo>
cd SDL_agent
cp config.example.json config.json
# 编辑 config.json，填入 API Key
pip install -r requirements.txt
python app.py
```

## 7. 验证

- Phase 1 测试：10/10 pass
- Phase 2 测试：12/12 pass
- app.py 启动正常
- `git status` 确认 `config.json` 未被追踪

## 8. 文件变更总览

```
新建（3个）:
  config.example.json
  config.json
  utils/config_loader.py

修改（3个）:
  core/config.py       (重构：外部配置加载)
  .gitignore           (+config.json)
  CLAUDE.md            (Quick Start / Configuration 更新)
  README.md            (配置文件相关章节更新)

新建（1个）:
  logs/CONFIG_EXTERNALIZATION_20260507_lkx.md  (本文件)
```
