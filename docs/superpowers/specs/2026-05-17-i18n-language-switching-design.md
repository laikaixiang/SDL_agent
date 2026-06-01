# i18n 中/英文切换 — 设计文档

**日期:** 2026-05-17
**状态:** 待审阅

---

## 目标

为 SDL_agent 添加中/英文切换功能，覆盖 V2 前端 UI、后端 API 响应消息、LLM prompt 三部分。

## 架构概览

```
┌──────────────────────────────────────────────┐
│  前端 (Vue 3 + vue-i18n)                      │
│  frontend/src/locales/{en,zh}.json            │
│  ┌─────────────┐  ┌──────────────────────┐    │
│  │ LanguageSwitch│  │ $t('key') 替换硬编码  │    │
│  │ UI 组件       │  │ ~330 处, 37 个文件    │    │
│  └─────────────┘  └──────────────────────┘    │
│  语言偏好: localStorage + document.cookie      │
├──────────────────────────────────────────────┤
│  后端 (Flask + I18nHelper)                    │
│  utils/i18n.py — 读取 locates/*.json          │
│  ~80 处硬编码响应 → I18nHelper.get(key, lang)  │
│  语言检测: request.cookies.lang → 默认 zh     │
├──────────────────────────────────────────────┤
│  Prompt (按语言分目录)                         │
│  prompts/zh/ — 现有 16 个 yaml                │
│  prompts/en/ — 新建英文翻译                    │
│  PromptManager.get(key, lang='zh')            │
├──────────────────────────────────────────────┤
│  Python 工具链 (scripts/)                     │
│  i18n_sync.py — key 结构对齐                  │
│  i18n_translate.py — LLM 自动翻译             │
│  i18n_check.py — CI 一致性校验                │
└──────────────────────────────────────────────┘
```

## 1. 翻译文件

**位置:** `frontend/src/locales/en.json` (源语言) + `zh.json` (中文翻译)

**格式:** 嵌套 JSON，按功能模块组织

```json
{
  "common": {
    "save": "Save",
    "cancel": "Cancel",
    "delete": "Delete",
    "confirm": "Confirm",
    "loading": "Loading..."
  },
  "chat": {
    "placeholder": "Type a message... (Enter to send)",
    "thinking": "AI is thinking...",
    "send": "Send"
  },
  "sidebar": {
    "logo": "SDL Agent",
    "expand": "Expand sidebar",
    "collapse": "Collapse sidebar",
    "modes": {
      "chat": "Chat",
      "extraction": "Literature Extraction",
      "hardware": "Hardware Control",
      "experiment": "Experiment Design",
      "analysis": "Data Analysis"
    }
  },
  "experiment": {
    "unnamed": "Untitled Experiment",
    "steps": "Steps",
    "compile": "Compile",
    "run": "Run"
  }
}
```

- **en.json 为源语言**（新 key 先加在这里）
- zh.json 由 `i18n_sync.py` 对齐结构 + `i18n_translate.py` 自动翻译

## 2. 前端

### 2.1 依赖

- `vue-i18n` (v10+) — 官方 Vue 3 国际化库

### 2.2 初始化 (`frontend/src/i18n/index.ts`)

```typescript
import { createI18n } from 'vue-i18n'
import en from '@/locales/en.json'
import zh from '@/locales/zh.json'

export function getLanguage(): string {
  return localStorage.getItem('language')
    || navigator.language?.startsWith('en') ? 'en' : 'zh'
}

export function setLanguage(lang: string) {
  localStorage.setItem('language', lang)
  document.cookie = `lang=${lang}; path=/; max-age=${365 * 86400}`
  i18n.global.locale.value = lang
}

const i18n = createI18n({
  legacy: false,
  locale: getLanguage(),
  fallbackLocale: 'zh',
  messages: { en, zh }
})

export default i18n
```

- **legacy: false** — 使用 Composition API 模式
- **语言优先级:** localStorage > navigator.language > 默认 zh

### 2.3 Store 整合 (`frontend/src/stores/settings.ts`)

```typescript
export const useSettingsStore = defineStore('settings', () => {
  const language = ref(getLanguage())

  function switchLanguage(lang: string) {
    language.value = lang
    setLanguage(lang)  // localStorage + cookie + vue-i18n
  }

  return { language, switchLanguage }
})
```

### 2.4 语言切换 UI

- 在 TopBar 或 Sidebar 底部加一个简洁的 `zh / en` 切换按钮
- 不设完整下拉（先只做中英双语），以后可扩展

### 2.5 模板中使用

```vue
<template>
  <button>{{ $t('common.save') }}</button>
  <input :placeholder="$t('chat.placeholder')" />
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'
const { t } = useI18n()
const label = computed(() => t('experiment.unnamed'))
</script>
```

### 2.6 需迁移的文件（37 个 .vue + .ts）

| 类别 | 文件数 | 中文字符串估数 |
|------|--------|---------------|
| pages/ | 5 | ~30 |
| components/experiment/ | 6 | ~40 |
| components/layout/ | 6 | ~35 |
| components/chat/ | 4 | ~15 |
| components/search/ | 5 | ~8 |
| components/modals/ | 4 | ~10 |
| components/common/ | 5 | 0 |
| components/cards/ | 1 | 0 |
| stores/ | 4 | ~20 |
| api/ | 4 | ~5 |
| **合计** | **37** | **~165 个唯一 key** |

## 3. 后端

### 3.1 I18nHelper (`utils/i18n.py`)

```python
import json
import os

class I18nHelper:
    def __init__(self, locales_dir=None):
        if locales_dir is None:
            locales_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'frontend', 'src', 'locales'
            )
        self._translations = {}
        for lang in ['en', 'zh']:
            path = os.path.join(locales_dir, f'{lang}.json')
            with open(path, 'r', encoding='utf-8') as f:
                self._translations[lang] = json.load(f)

    def get(self, key: str, lang: str = 'zh') -> str:
        """支持点分隔路径: 'chat.placeholder'"""
        value = self._translations.get(lang, self._translations['zh'])
        for part in key.split('.'):
            if isinstance(value, dict):
                value = value.get(part, key)
            else:
                return key
        return value if isinstance(value, str) else key

    def get_lang(self, request) -> str:
        lang = request.cookies.get('lang', 'zh')
        return lang if lang in ('en', 'zh') else 'zh'
```

- 直接读 `frontend/src/locales/*.json`，和前端共用同一份翻译
- 后续如有后端特有的消息 key，追加在 JSON 中即可

### 3.2 后端迁移（app.py ~80 处）

现有模式:
```python
return jsonify({'type': 'error', 'reply': '提取任务失败，请重试'})
```

迁移后:
```python
from utils.i18n import i18n_helper
lang = i18n_helper.get_lang(request)
return jsonify({'type': 'error', 'reply': i18n_helper.get('errors.extraction_failed', lang)})
```

### 3.3 Cookie 设置

前端 `setLanguage()` 同时写 Cookie `lang=en|zh`，后端自动读取。首次访问无 Cookie 则默认 `zh`。

## 4. Prompt

### 4.1 目录结构

```
prompts/
├── zh/                       ← 现有 16 个 yaml 搬入
│   ├── registry.yaml
│   ├── extraction/
│   │   ├── _system_vision.yaml
│   │   └── ...
│   └── ...
├── en/                       ← 新建英文翻译
│   ├── registry.yaml
│   ├── extraction/
│   │   ├── _system_vision.yaml
│   │   └── ...
│   └── ...
└── manager.py                ← PromptManager 增加 lang 参数
```

### 4.2 PromptManager 改动

```python
# manager.py
def __init__(self, lang='zh'):
    self._lang = lang
    self._base_dir = os.path.join(BASE_DIR, lang)
    # 其余逻辑不变，只是从对应语言目录加载
```

调用方:
```python
pm = create_prompt_manager(lang=lang)
text = pm.get("extraction_system_vision", task_description=..., fields=...)
```

## 5. Python 工具链 (`scripts/`)

### 5.1 `scripts/i18n_sync.py` — 结构对齐

以 `en.json` 为基准：
- zh.json 缺失的 key → 补入，值为 `"[待翻译] <en 原文>"`
- zh.json 多余的 key → 删除
- 输出对齐后的 `zh.json`

### 5.2 `scripts/i18n_translate.py` — 自动翻译

- 扫描 `zh.json` 中值包含 `[待翻译]` 标记的条目
- 调用项目已有 `LLMClient` 批量翻译（每次 10-20 条，控制上下文）
- 将翻译结果写回 `zh.json`

### 5.3 `scripts/i18n_check.py` — 一致性校验

- 比较 `en.json` 和 `zh.json` 的所有 key 路径
- 不一致时打印差异并 exit(1)
- 可通过 pre-commit hook 或 CI 调用

## 6. 变更清单

| 变更类型 | 文件/目录 | 说明 |
|----------|----------|------|
| **新建** | `frontend/src/locales/en.json` | 英文翻译（源语言） |
| **新建** | `frontend/src/locales/zh.json` | 中文翻译 |
| **新建** | `frontend/src/i18n/index.ts` | vue-i18n 初始化 |
| **新建** | `frontend/src/stores/settings.ts` | 语言设置 store |
| **新建** | `utils/i18n.py` | 后端 I18nHelper |
| **新建** | `scripts/i18n_sync.py` | 结构对齐工具 |
| **新建** | `scripts/i18n_translate.py` | LLM 自动翻译工具 |
| **新建** | `scripts/i18n_check.py` | 一致性校验工具 |
| **新建** | `prompts/en/` | 英文 prompt（16个 yaml） |
| **移动** | `prompts/*.yaml` → `prompts/zh/` | 现有 prompt 归类 |
| **修改** | `frontend/src/main.ts` | 注册 vue-i18n 插件 |
| **修改** | `frontend/src/components/layout/TopBar.vue` | 添加语言切换按钮 |
| **修改** | `frontend/src/index.html` | `<html lang>` 动态化 |
| **修改** | 37 个 .vue/.ts 文件 | 硬编码中文 → `$t()` |
| **修改** | `app.py` | ~80 处响应 → `i18n_helper.get()` |
| **修改** | `prompts/manager.py` | 增加 lang 参数 |
| **修改** | 所有调用 `create_prompt_manager()` 的地方 | 传入 lang |

## 7. 实施策略

### 7.1 分阶段

| 阶段 | 内容 | 依赖 |
|------|------|------|
| **Phase 1** | 基础设施：vue-i18n 配置、`utils/i18n.py`、locales JSON 骨架、store、工具链脚本 | 无 |
| **Phase 2** | 前端迁移：并行 agent 改 37 个文件 | Phase 1 |
| **Phase 3** | 后端迁移：app.py ~80 处 + prompt 分目录 | Phase 1 |
| **Phase 4** | 工具链运行：sync → translate → check 循环，补全翻译 | Phase 2, 3 |
| **Phase 5** | 测试验证 + TopBar 语言切换按钮 | Phase 2, 3 |

### 7.2 执行策略

- Phase 2 前端迁移使用**多个并行 agent**（按组件目录拆分）提高效率
- 每个 agent 负责一个目录的文件改造
- 设一个**审核 agent** 检查每个 agent 的输出，不合格打回重做
- Phase 3 后端迁移同理

## 8. 不做的事

- 不做 LLM 回复语言切换（LLM 用哪种语言由 prompt 决定，已在范围外）
- 不做 V1 旧前端（实际已废弃，指向同一 SPA）
- 不做从中文到英文的逐句人工校对（首版用 LLM 翻译 + `[待翻译]` 标记，后续人工校对独立进行）
- 不做超过中/英两种语言（架构支持扩展，但本次只做双语）
