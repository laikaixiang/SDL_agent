# i18n Language Switching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Chinese/English language switching covering V2 frontend UI, backend API responses, and LLM prompts.

**Architecture:** vue-i18n for Vue 3 frontend (Composition API mode), Python I18nHelper for Flask backend, shared JSON translation files (`frontend/src/locales/{en,zh}.json`). Language detection: localStorage + cookie > navigator.language > default zh. Prompts split by `prompts/{zh,en}/` directory structure.

**Tech Stack:** vue-i18n v10+, Flask, Python 3.x, LLMClient (existing)

---

## File Structure Map

```
Create:
  frontend/src/locales/en.json           — English source translations
  frontend/src/locales/zh.json           — Chinese translations
  frontend/src/i18n/index.ts            — vue-i18n init + getLanguage/setLanguage
  frontend/src/stores/settings.ts       — Language state store
  utils/i18n.py                         — Flask I18nHelper class
  scripts/i18n_sync.py                  — Key alignment tool
  scripts/i18n_translate.py             — LLM auto-translate tool
  scripts/i18n_check.py                 — Consistency checker
  prompts/zh/                           — Move existing 16 yaml files here
  prompts/en/                           — New English translations

Modify:
  frontend/src/main.ts                  — Register vue-i18n plugin
  frontend/src/components/layout/TopBar.vue — Add language toggle
  prompts/manager.py                    — Add lang parameter
  prompts/__init__.py                   — Update create_prompt_manager() signature
  app.py                                — ~80 Chinese response strings → i18n_helper
  frontend/package.json                 — Add vue-i18n dependency

Dependent modifications (need i18n keys defined first):
  7 frontend/src/components/experiment/*.vue (6 files)
  6 frontend/src/components/layout/*.vue (6 files)
  4 frontend/src/components/chat/*.vue (4 files)
  5 frontend/src/components/search/*.vue (5 files)
  4 frontend/src/components/modals/*.vue (4 files)
  5 frontend/src/pages/*.vue (5 files)
  4 frontend/src/stores/*.ts (4 files)
  4 frontend/src/api/*.ts (4 files)
  frontend/src/components/layout/TopBar.vue
```

---

### Task 1: Install vue-i18n and create en.json source file

**Files:**
- Create: `frontend/src/locales/en.json`
- Modify: `frontend/package.json`

- [ ] **Step 1: Install vue-i18n dependency**

Add to frontend/package.json dependencies:
```json
"vue-i18n": "^10.0.5"
```

Run:
```bash
cd D:/PycharmProjects/SDL_agent/frontend && npm install
```

- [ ] **Step 2: Create en.json as source language**

Write `frontend/src/locales/en.json`:

```json
{
  "common": {
    "save": "Save",
    "cancel": "Cancel",
    "delete": "Delete",
    "confirm": "Confirm",
    "loading": "Loading...",
    "close": "Close",
    "back": "Back",
    "add": "Add"
  },
  "topbar": {
    "oldVersion": "Old Version",
    "lightMode": "Light Mode",
    "darkMode": "Dark Mode"
  },
  "sidebar": {
    "expandPanel": "Expand Panel",
    "collapsePanel": "Collapse Panel",
    "expandNav": "Expand Nav",
    "collapseNav": "Collapse Nav"
  },
  "modes": {
    "chat": "Chat",
    "literatureExtraction": "Literature Extraction",
    "hardwareControl": "Hardware Control",
    "experimentDesign": "Experiment Design",
    "dataAnalysis": "Data Analysis"
  },
  "chat": {
    "inputPlaceholder": "Type a message... (Enter to send)",
    "thinking": "AI thinking...",
    "sending": "Sending...",
    "send": "Send",
    "clearHistory": "Clear History"
  }
}
```

- [ ] **Step 3: Create zh.json (placeholder — will be filled by translate script later)**

Write `frontend/src/locales/zh.json` with the same structure, values = Chinese equivalents:

```json
{
  "common": {
    "save": "保存",
    "cancel": "取消",
    "delete": "删除",
    "confirm": "确认",
    "loading": "加载中...",
    "close": "关闭",
    "back": "返回",
    "add": "添加"
  },
  "topbar": {
    "oldVersion": "旧版",
    "lightMode": "亮色模式",
    "darkMode": "暗色模式"
  },
  "sidebar": {
    "expandPanel": "展开历史面板",
    "collapsePanel": "收起历史面板",
    "expandNav": "展开导航面板",
    "collapseNav": "收起导航面板"
  },
  "modes": {
    "chat": "对话",
    "literatureExtraction": "文献提取",
    "hardwareControl": "硬件控制",
    "experimentDesign": "实验设计",
    "dataAnalysis": "数据分析"
  },
  "chat": {
    "inputPlaceholder": "输入消息... (Enter 发送)",
    "thinking": "AI 回复中...",
    "sending": "发送中...",
    "send": "发送",
    "clearHistory": "清除历史"
  }
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/locales/en.json frontend/src/locales/zh.json frontend/package.json frontend/package-lock.json
git commit -m "feat: add vue-i18n dependency and initial locale files"
```

---

### Task 2: Create vue-i18n initialization and settings store

**Files:**
- Create: `frontend/src/i18n/index.ts`
- Create: `frontend/src/stores/settings.ts`
- Modify: `frontend/src/main.ts`

- [ ] **Step 1: Create vue-i18n init module**

Write `frontend/src/i18n/index.ts`:

```typescript
import { createI18n } from 'vue-i18n'
import en from '@/locales/en.json'
import zh from '@/locales/zh.json'

export function getLanguage(): string {
  const stored = localStorage.getItem('language')
  if (stored === 'en' || stored === 'zh') return stored
  if (navigator.language?.toLowerCase().startsWith('en')) return 'en'
  return 'zh'
}

export function setLanguage(lang: string): void {
  localStorage.setItem('language', lang)
  document.cookie = `lang=${lang}; path=/; max-age=${365 * 86400}; SameSite=Lax`
  if (i18n.global.locale.value !== lang) {
    i18n.global.locale.value = lang
  }
}

const i18n = createI18n({
  legacy: false,
  locale: getLanguage(),
  fallbackLocale: 'zh',
  messages: { en, zh }
})

export default i18n
```

- [ ] **Step 2: Create settings store**

Write `frontend/src/stores/settings.ts`:

```typescript
import { defineStore } from 'pinia'
import { ref } from 'vue'
import { getLanguage, setLanguage } from '@/i18n'

export const useSettingsStore = defineStore('settings', () => {
  const language = ref<string>(getLanguage())

  function switchLanguage(lang: string) {
    language.value = lang
    setLanguage(lang)
  }

  return { language, switchLanguage }
})
```

- [ ] **Step 3: Register vue-i18n in main.ts**

Read `frontend/src/main.ts` and add i18n registration:

```typescript
// After: import { createPinia } from 'pinia'
import i18n from '@/i18n'

// After: const app = createApp(App)
// Before: app.use(createPinia())
app.use(i18n)
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/i18n/index.ts frontend/src/stores/settings.ts frontend/src/main.ts
git commit -m "feat: add vue-i18n init module and settings store"
```

---

### Task 3: Create backend I18nHelper

**Files:**
- Create: `utils/i18n.py`

- [ ] **Step 1: Write I18nHelper class**

Write `utils/i18n.py`:

```python
"""I18nHelper — reads shared locales JSON for Flask responses"""
import json
import os
from typing import Any


class I18nHelper:
    def __init__(self, locales_dir: str = None):
        if locales_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            locales_dir = os.path.join(project_root, 'frontend', 'src', 'locales')
        self._translations: dict[str, dict[str, Any]] = {}
        for lang in ('en', 'zh'):
            path = os.path.join(locales_dir, f'{lang}.json')
            with open(path, 'r', encoding='utf-8') as f:
                self._translations[lang] = json.load(f)

    def get(self, key: str, lang: str = 'zh') -> str:
        """Resolve dot-separated path like 'chat.placeholder'"""
        translations = self._translations.get(lang, self._translations['zh'])
        value: Any = translations
        for part in key.split('.'):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return key
        return value if isinstance(value, str) else key

    def get_lang(self, request) -> str:
        lang = request.cookies.get('lang', 'zh')
        return lang if lang in ('en', 'zh') else 'zh'


i18n = I18nHelper()
```

- [ ] **Step 2: Commit**

```bash
git add utils/i18n.py
git commit -m "feat: add backend I18nHelper class"
```

---

### Task 4: Create Python tool chain (3 scripts)

**Files:**
- Create: `scripts/i18n_sync.py`
- Create: `scripts/i18n_translate.py`
- Create: `scripts/i18n_check.py`

- [ ] **Step 1: Write i18n_sync.py**

Write `scripts/i18n_sync.py`:

```python
"""Align zh.json key structure with en.json (source of truth).

Missing keys in zh.json are filled with "[待翻译] <en value>".
Extra keys in zh.json (not in en.json) are removed.

Usage: python scripts/i18n_sync.py [--locales-dir PATH]
"""
import json
import sys
import os

LOCALES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'frontend', 'src', 'locales')


def extract_keys(obj, prefix=''):
    """Recursively extract all dot-separated key paths."""
    keys = set()
    for k, v in obj.items():
        path = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            keys.update(extract_keys(v, path))
        else:
            keys.add(path)
    return keys


def get_value(obj, path):
    parts = path.split('.')
    cur = obj
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur if isinstance(cur, str) else None


def set_value(obj, path, value):
    parts = path.split('.')
    cur = obj
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def delete_extra_keys(obj, valid_paths, prefix=''):
    """Remove keys from obj that are not in valid_paths."""
    keys_to_delete = []
    for k, v in list(obj.items()):
        path = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            keys_to_delete += delete_extra_keys(v, valid_paths, path)
            if not obj[k]:
                keys_to_delete.append(path)
        elif path not in valid_paths:
            keys_to_delete.append(path)
    return keys_to_delete


def sync(zh_path, en_path):
    with open(en_path, 'r', encoding='utf-8') as f:
        en = json.load(f)
    with open(zh_path, 'r', encoding='utf-8') as f:
        zh = json.load(f)

    en_keys = extract_keys(en)
    zh_keys = extract_keys(zh)

    added, removed = 0, 0

    # Add missing keys
    for key in sorted(en_keys - zh_keys):
        en_val = get_value(en, key)
        set_value(zh, key, f'[待翻译] {en_val}')
        print(f'  + {key}')
        added += 1

    # Remove extra keys
    for key in sorted(zh_keys - en_keys):
        parts = key.split('.')
        parent = zh
        for p in parts[:-1]:
            parent = parent[p]
        if parts[-1] in parent:
            del parent[parts[-1]]
        print(f'  - {key}')
        removed += 1

    with open(zh_path, 'w', encoding='utf-8') as f:
        json.dump(zh, f, ensure_ascii=False, indent=2)

    print(f'\nDone: +{added} added, -{removed} removed')
    return added, removed


if __name__ == '__main__':
    en = os.path.join(LOCALES_DIR, 'en.json')
    zh = os.path.join(LOCALES_DIR, 'zh.json')
    if not os.path.exists(en):
        print(f'Error: {en} not found')
        sys.exit(1)
    if not os.path.exists(zh):
        print(f'Error: {zh} not found')
        sys.exit(1)
    sync(zh, en)
```

- [ ] **Step 2: Write i18n_translate.py**

Write `scripts/i18n_translate.py`:

```python
"""Auto-translate [待翻译] entries in zh.json using LLM.

Reads en.json as English source, scans zh.json for entries with '[待翻译]'
prefix, sends batches to LLM for translation, updates zh.json in place.

Usage: python scripts/i18n_translate.py [--batch-size 15]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import config_obj
from core.llm_client import LLMClient

LOCALES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'frontend', 'src', 'locales')


def find_pending(zh) -> list:
    """Find all entries with [待翻译] prefix, return [(key_path, en_value), ...]"""
    pending = []

    def walk(obj, prefix=''):
        for k, v in obj.items():
            path = f'{prefix}.{k}' if prefix else k
            if isinstance(v, dict):
                walk(v, path)
            elif isinstance(v, str) and v.startswith('[待翻译]'):
                en_val = v[len('[待翻译] '):]
                pending.append((path, en_val))
    walk(zh)
    return pending


def apply_translations(zh, translations: dict):
    """Apply translated values back to zh dict."""
    for path, value in translations.items():
        parts = path.split('.')
        cur = zh
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = value


def translate_batch(client: LLMClient, items: list) -> dict:
    """Call LLM to translate a batch. Returns {path: chinese_value, ...}"""
    lines = '\n'.join(f'{i+1}. {en_val}' for i, (_, en_val) in enumerate(items))
    prompt = (
        f'Translate the following English UI strings to Simplified Chinese (简体中文). '
        f'Return ONLY a JSON object mapping the number to the Chinese translation. '
        f'Keep it concise and natural for a lab automation software UI.\n\n{lines}'
    )

    resp = client.call_api([
        {'role': 'user', 'content': prompt}
    ], stream=False, temperature=0.1, max_tokens=2000)

    raw = resp.get('content', '') if isinstance(resp, dict) else str(resp)
    try:
        # Parse JSON from response (handle markdown code blocks)
        raw = raw.replace('```json', '').replace('```', '').strip()
        idx_to_val = json.loads(raw)
        result = {}
        for idx_str, val in idx_to_val.items():
            i = int(idx_str) - 1
            result[items[i][0]] = val
        return result
    except (json.JSONDecodeError, KeyError, IndexError, ValueError):
        print(f'  Parse error, raw response: {raw[:200]}')
        return {}


def main(batch_size=15):
    en_path = os.path.join(LOCALES_DIR, 'en.json')
    zh_path = os.path.join(LOCALES_DIR, 'zh.json')

    with open(en_path, 'r', encoding='utf-8') as f:
        en = json.load(f)
    with open(zh_path, 'r', encoding='utf-8') as f:
        zh = json.load(f)

    pending = find_pending(zh)
    if not pending:
        print('No pending translations found.')
        return

    print(f'Found {len(pending)} entries to translate')

    client = LLMClient(
        api_key=config_obj.TALK_API_KEY or config_obj.API_KEY,
        api_url=config_obj.TALK_API_URL or config_obj.API_URL,
        model=config_obj.TALK_MODEL_NAME or config_obj.MODEL_NAME_TALK,
        extra_body=config_obj.get_extra_body('TALK'),
    )

    for i in range(0, len(pending), batch_size):
        batch = pending[i:i + batch_size]
        print(f'  Batch {i//batch_size + 1}: {len(batch)} items...')
        translations = translate_batch(client, batch)
        apply_translations(zh, translations)
        for path in batch:
            if path[0] in translations:
                print(f'    OK {path[0]}')
            else:
                print(f'    FAIL {path[0]}')

    with open(zh_path, 'w', encoding='utf-8') as f:
        json.dump(zh, f, ensure_ascii=False, indent=2)

    remaining = len(find_pending(zh))
    print(f'\nDone. {remaining} entries still pending.')


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Write i18n_check.py**

Write `scripts/i18n_check.py`:

```python
"""Check that en.json and zh.json have identical key structures.

Exit 0 if consistent, exit 1 if mismatched (for pre-commit / CI).

Usage: python scripts/i18n_check.py
"""
import json
import os
import sys


def extract_paths(obj, prefix=''):
    """Recursively extract all dot-separated key paths."""
    paths = set()
    for k, v in obj.items():
        path = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            paths.update(extract_paths(v, path))
        else:
            paths.add(path)
    return paths


def check(locales_dir=None):
    if locales_dir is None:
        locales_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'frontend', 'src', 'locales'
        )

    en_path = os.path.join(locales_dir, 'en.json')
    zh_path = os.path.join(locales_dir, 'zh.json')

    with open(en_path, 'r', encoding='utf-8') as f:
        en = json.load(f)
    with open(zh_path, 'r', encoding='utf-8') as f:
        zh = json.load(f)

    en_keys = extract_paths(en)
    zh_keys = extract_paths(zh)

    missing_in_zh = en_keys - zh_keys
    missing_in_en = zh_keys - en_keys

    if missing_in_zh or missing_in_en:
        if missing_in_zh:
            print(f'Keys in en.json missing from zh.json ({len(missing_in_zh)}):')
            for k in sorted(missing_in_zh):
                print(f'  - {k}')
        if missing_in_en:
            print(f'Keys in zh.json missing from en.json ({len(missing_in_en)}):')
            for k in sorted(missing_in_en):
                print(f'  - {k}')
        sys.exit(1)

    print(f'OK: {len(en_keys)} keys match between en.json and zh.json')
    sys.exit(0)


if __name__ == '__main__':
    check()
```

- [ ] **Step 4: Verify scripts are syntactically correct**

```bash
cd D:/PycharmProjects/SDL_agent && python -c "import py_compile; py_compile.compile('scripts/i18n_sync.py', doraise=True)"
cd D:/PycharmProjects/SDL_agent && python -c "import py_compile; py_compile.compile('scripts/i18n_check.py', doraise=True)"
```

- [ ] **Step 5: Commit**

```bash
git add scripts/i18n_sync.py scripts/i18n_translate.py scripts/i18n_check.py
git commit -m "feat: add i18n tool chain (sync, translate, check)"
```

---

### Task 5: Restructure prompts directory (zh/ + en/) and update PromptManager

**Files:**
- Create: `prompts/zh/`, `prompts/en/` directories with yaml files
- Modify: `prompts/manager.py`
- Modify: `prompts/__init__.py`

- [ ] **Step 1: Move existing prompts to zh/ directory**
```bash
cd D:/PycharmProjects/SDL_agent
mkdir -p prompts/zh prompts/en
# Move all yaml subdirectories into zh/
for dir in extraction field_inference experiment_design hardware algorithm_gen data_analysis misc optimizer; do
  if [ -d "prompts/$dir" ]; then
    mv "prompts/$dir" "prompts/zh/$dir"
  fi
done
```

- [ ] **Step 2: Update registry.yaml paths**

Edit `prompts/registry.yaml` — prepend `zh/` to every `file:` path:
```yaml
version: 1

prompts:
  extraction_system_vision:
    file: zh/extraction/_system_vision.yaml
    ...
```

(All 16 entries get `zh/` prefix on their `file:` field)

- [ ] **Step 3: Update PromptManager for lang support**

Modify `prompts/manager.py` `__init__` to accept `lang` parameter:

```python
def __init__(self, registry_path: str, overrides_dir: str, lang: str = 'zh'):
    self._registry_path = registry_path
    self._overrides_dir = overrides_dir
    self._lang = lang
    self._registry: Dict[str, dict] = {}
    self._prompts: Dict[str, dict] = {}
    self._templates: Dict[str, Template] = {}
    self._load_all()
```

Modify `_load_prompt_file` to resolve language path. The registry `file` field now contains `zh/extraction/_system_vision.yaml`. When `lang='en'`, replace `zh/` with `en/`:

```python
def _resolve_path(self, file_path: str) -> str:
    """Replace language prefix based on current lang setting."""
    parts = file_path.split('/', 1)
    if parts[0] in ('zh', 'en') and len(parts) == 2:
        return os.path.join(
            os.path.dirname(self._registry_path),
            self._lang, parts[1]
        )
    return os.path.join(os.path.dirname(self._registry_path), file_path)
```

Update `_load_prompt_file` to use `_resolve_path` instead of direct `os.path.join`.

- [ ] **Step 4: Update prompts/__init__.py factory function**

Read `prompts/__init__.py`, update `create_prompt_manager()` to accept `lang`:
```python
def create_prompt_manager(lang: str = 'zh') -> PromptManager:
    registry = os.path.join(os.path.dirname(__file__), 'registry.yaml')
    overrides = os.path.join(os.path.dirname(__file__), 'overrides')
    return PromptManager(registry, overrides, lang=lang)
```

- [ ] **Step 5: Copy zh/ prompt files to en/ as placeholders**

Copy all 16 yaml files to `prompts/en/` with English text. For now, add `[待翻译]` markers to English versions of templates that need translation. The yaml metadata (name, description, variables) should be in English.

- [ ] **Step 6: Commit**

```bash
git add prompts/
git commit -m "feat: restructure prompts for bilingual support"
```

---

### Task 6: Add TopBar language toggle button

**Files:**
- Modify: `frontend/src/components/layout/TopBar.vue`

- [ ] **Step 1: Add language toggle to TopBar**

Add a `zh | en` toggle button in TopBar between the brand and the right buttons:

```vue
<script setup lang="ts">
import { useSettingsStore } from '@/stores/settings'
const settings = useSettingsStore()
</script>

<template>
  <header class="topbar">
    <div class="topbar-left">
      <!-- existing sidebar toggle -->
      <span class="topbar-brand">SDL Agent</span>
      <!-- existing old-link -->
      <div class="lang-toggle">
        <button
          :class="{ active: settings.language === 'zh' }"
          @click="settings.switchLanguage('zh')"
        >中</button>
        <button
          :class="{ active: settings.language === 'en' }"
          @click="settings.switchLanguage('en')"
        >EN</button>
      </div>
    </div>
    <!-- existing right section -->
  </header>
</template>

<style scoped>
.lang-toggle {
  display: flex;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  overflow: hidden;
  margin-left: var(--space-md);
}
.lang-toggle button {
  border: none;
  background: transparent;
  color: var(--color-text-tertiary);
  font-size: 12px;
  padding: 2px 8px;
  cursor: pointer;
  transition: all var(--transition-fast);
}
.lang-toggle button.active {
  background: var(--color-primary);
  color: white;
}
</style>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/layout/TopBar.vue
git commit -m "feat: add language toggle to TopBar"
```

---

### Task 7A: Migrate components/layout/ to $t() (6 files)

**Files:**
- Modify: `frontend/src/components/layout/Sidebar.vue`
- Modify: `frontend/src/components/layout/NavPanel.vue`
- Modify: `frontend/src/components/layout/HistoryPanel.vue`
- Modify: `frontend/src/components/layout/TaskPanel.vue`
- Modify: `frontend/src/components/layout/PdfPanel.vue`
- Modify: `frontend/src/components/layout/TopBar.vue` (additional strings beyond Task 6)
- Create: `frontend/src/locales/_partial_layout.en.json` — new keys found by this agent
- Create: `frontend/src/locales/_partial_layout.zh.json` — Chinese values for new keys

**IMPORTANT — partial file strategy:**
Each parallel agent writes its own `_partial_<dir>.{en,zh}.json` file. Do NOT modify the main `en.json`/`zh.json` directly. A merge step (Task 7H) combines all partials.

**Template changes pattern:**
```vue
<!-- Before -->
<span>暂无会话</span>
<input placeholder="搜索历史..." />

<!-- After -->
<span>{{ $t('history.noSessions') }}</span>
<input :placeholder="$t('history.searchPlaceholder')" />
```

For `<script>` strings (inside Vue SFC `<script setup>`):
```typescript
import { useI18n } from 'vue-i18n'
const { t } = useI18n()
const msg = t('common.success')
```

For standalone `.ts` files (stores, api wrappers — outside Vue components):
```typescript
import i18n from '@/i18n'
const msg = i18n.global.t('experiment.unnamed')
```

**Agent must produce:**
1. Modified `.vue`/`.ts` files (Chinese strings replaced with `$t()`/`t()` calls)
2. `frontend/src/locales/_partial_layout.en.json` with all new keys and English values
3. `frontend/src/locales/_partial_layout.zh.json` with all new keys and Chinese values
4. A checklist of: file modified, old string, new key path used

---

### Task 7B: Migrate components/experiment/ to $t() (6 files)

**Files:**
- Modify: `frontend/src/components/experiment/CodeArea.vue`
- Modify: `frontend/src/components/experiment/ElementPanel.vue`
- Modify: `frontend/src/components/experiment/StepCanvas.vue`
- Modify: `frontend/src/components/experiment/StepCard.vue`
- Modify: `frontend/src/components/experiment/StepEditor.vue`
- Modify: `frontend/src/components/experiment/VariableBar.vue`
- Create: `frontend/src/locales/_partial_experiment.en.json`
- Create: `frontend/src/locales/_partial_experiment.zh.json`

Same pattern as Task 7A. Output `_partial_experiment.{en,zh}.json` — do NOT touch the main locale files.


### Task 7C: Migrate components/chat/ to $t() (4 files)

**Files:**
- Modify: `frontend/src/components/chat/ChatContainer.vue`
- Modify: `frontend/src/components/chat/InputBar.vue`
- Modify: `frontend/src/components/chat/MessageBubble.vue`
- Modify: `frontend/src/components/chat/ThinkingBlock.vue`
- Create: `frontend/src/locales/_partial_chat.en.json`
- Create: `frontend/src/locales/_partial_chat.zh.json`

Same pattern as Task 7A. Output `_partial_chat.{en,zh}.json`.


### Task 7D: Migrate components/search/ to $t() (5 files)

**Files:**
- Modify: `frontend/src/components/search/SearchBar.vue`
- Modify: `frontend/src/components/search/SearchResultCard.vue`
- Modify: `frontend/src/components/search/SearchResultList.vue`
- Modify: `frontend/src/components/search/LiteratureCard.vue`
- Modify: `frontend/src/components/search/AbstractPreview.vue`
- Create: `frontend/src/locales/_partial_search.en.json`
- Create: `frontend/src/locales/_partial_search.zh.json`

Same pattern as Task 7A. Output `_partial_search.{en,zh}.json`.


### Task 7E: Migrate components/modals/ to $t() (4 files)

**Files:**
- Modify: `frontend/src/components/modals/ConfirmDialog.vue`
- Modify: `frontend/src/components/modals/FileSelectorModal.vue`
- Modify: `frontend/src/components/modals/ModalContainer.vue`
- Modify: `frontend/src/components/modals/SummaryModal.vue`
- Create: `frontend/src/locales/_partial_modals.en.json`
- Create: `frontend/src/locales/_partial_modals.zh.json`

Same pattern as Task 7A. Output `_partial_modals.{en,zh}.json`.


### Task 7F: Migrate pages/ to $t() (5 files)

**Files:**
- Modify: `frontend/src/pages/ChatPage.vue`
- Modify: `frontend/src/pages/ExtractionPage.vue`
- Modify: `frontend/src/pages/HardwarePage.vue`
- Modify: `frontend/src/pages/ExperimentPage.vue`
- Modify: `frontend/src/pages/AnalysisPage.vue`
- Create: `frontend/src/locales/_partial_pages.en.json`
- Create: `frontend/src/locales/_partial_pages.zh.json`

Same pattern as Task 7A. Output `_partial_pages.{en,zh}.json`.


### Task 7G: Migrate stores/ and api/ to $t() (8 files)

**Files:**
- Modify: `frontend/src/stores/chat.ts`
- Modify: `frontend/src/stores/experiment.ts`
- Modify: `frontend/src/stores/layout.ts`
- Modify: `frontend/src/stores/analysis.ts`
- Modify: `frontend/src/api/chat.ts`
- Modify: `frontend/src/api/experiment.ts`
- Modify: `frontend/src/api/history.ts`
- Modify: `frontend/src/api/analysis.ts`
- Create: `frontend/src/locales/_partial_stores_api.en.json`
- Create: `frontend/src/locales/_partial_stores_api.zh.json`

Same pattern as Task 7A, but note: `.ts` files outside `.vue` SFCs cannot use `useI18n()` composable. Use direct i18n instance instead:
```typescript
import i18n from '@/i18n'
const msg = i18n.global.t('experiment.unnamed')
```
Output `_partial_stores_api.{en,zh}.json`.

---

### Task 7H: Merge all partial locale files into en.json / zh.json

**Files:**
- Modify: `frontend/src/locales/en.json`
- Modify: `frontend/src/locales/zh.json`
- Delete: `frontend/src/locales/_partial_*.{en,zh}.json` (cleanup)

- [ ] **Step 1: Write merge script inline**

```bash
cd D:/PycharmProjects/SDL_agent
python -c "
import json, os, glob

locales = 'frontend/src/locales'

# Load existing base
with open(f'{locales}/en.json', 'r', encoding='utf-8') as f:
    en = json.load(f)
with open(f'{locales}/zh.json', 'r', encoding='utf-8') as f:
    zh = json.load(f)

# Deep merge each partial into base
def deep_merge(base, patch):
    for k, v in patch.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v

for file in sorted(glob.glob(f'{locales}/_partial_*.en.json')):
    with open(file, 'r', encoding='utf-8') as f:
        patch = json.load(f)
    deep_merge(en, patch)
    print(f'  Merged {os.path.basename(file)} ({len(patch)} top-level keys)')

for file in sorted(glob.glob(f'{locales}/_partial_*.zh.json')):
    with open(file, 'r', encoding='utf-8') as f:
        patch = json.load(f)
    deep_merge(zh, patch)
    print(f'  Merged {os.path.basename(file)} ({len(patch)} top-level keys)')

with open(f'{locales}/en.json', 'w', encoding='utf-8') as f:
    json.dump(en, f, ensure_ascii=False, indent=2)
with open(f'{locales}/zh.json', 'w', encoding='utf-8') as f:
    json.dump(zh, f, ensure_ascii=False, indent=2)

# Delete partials
for file in glob.glob(f'{locales}/_partial_*.json'):
    os.remove(file)
    print(f'  Removed {os.path.basename(file)}')

print('Merge complete.')
"
```

- [ ] **Step 2: Run i18n_check.py to verify consistency**

```bash
cd D:/PycharmProjects/SDL_agent && python scripts/i18n_check.py
```
Expected: key count matches between en.json and zh.json.

- [ ] **Step 3: Run i18n_sync.py to catch any stragglers**

```bash
cd D:/PycharmProjects/SDL_agent && python scripts/i18n_sync.py
```
Expected: no additions or removals needed if all agents were thorough.

---

### Task 8: Review agent — audit all frontend changes

This task runs AFTER all Tasks 7A-7G complete. A review agent checks:

1. **No missed Chinese strings** — grep for `[一-鿿]` in all modified files; any remaining Chinese characters should only be in `zh.json` or intentional LLM output handling
2. **All keys exist in en.json AND zh.json** — extract all `$t('...')` and `t('...')` calls from frontend source, verify each key exists in both JSON files
3. **No unescaped `$t()` in attribute bindings** — in `<template>`, bare attribute `placeholder="..."` should be `:placeholder="$t('...')"`
4. **Vue SFC syntax valid** — no broken template/script blocks
5. **Type consistency** — each key used must resolve to a string value (not an object) in the locales JSON

If issues found → report each issue with file + line → fix → re-check.

---

### Task 9: Migrate app.py backend responses

**Files:**
- Modify: `app.py`

**Pattern:** Replace hardcoded Chinese strings in `jsonify()` responses with `i18n_helper.get()`.

```python
# Before
return jsonify({'type': 'error', 'reply': '提取任务失败，请重试'})
return jsonify({'success': True, 'message': '会话已删除'})

# After
from utils.i18n import i18n
lang = i18n.get_lang(request)
return jsonify({'type': 'error', 'reply': i18n.get('errors.extractionFailed', lang)})
return jsonify({'success': True, 'message': i18n.get('success.sessionDeleted', lang)})
```

Keys used in backend can live in the same locales JSON files. Add a backend-specific section:

en.json additions:
```json
{
  "errors": {
    "extractionFailed": "Extraction failed, please retry",
    "uploadFailed": "Upload failed",
    "noSuchSession": "Session not found",
    "taskBusy": "Task is busy",
    "invalidParams": "Invalid parameters"
  },
  "success": {
    "sessionDeleted": "Session deleted",
    "saved": "Saved",
    "cacheCleared": "Cache cleared",
    "algorithmGenerated": "Algorithm generated"
  },
  "status": {
    "extractionStarted": "Extraction started",
    "extractionComplete": "Extraction complete",
    "taskCancelled": "Task cancelled"
  }
}
```

Apply to all ~80 response strings in app.py. Each takes ~1 minute — systematic scan.

---

### Task 10: Update backend PromptManager callers

**Files:**
- All files that call `create_prompt_manager()` (grep for this call site)

- [ ] **Step 1: Find all callers**
```bash
cd D:/PycharmProjects/SDL_agent && grep -rn "create_prompt_manager" --include="*.py"
```

- [ ] **Step 2: Pass lang to each caller**

```python
# Before
pm = create_prompt_manager()

# After
lang = getattr(request, 'cookies', {}).get('lang', 'zh') if has_request() else 'zh'
pm = create_prompt_manager(lang=lang)
```

---

### Task 11: Build and verify

- [ ] **Step 1: Type check frontend**
```bash
cd D:/PycharmProjects/SDL_agent/frontend && npx vue-tsc -b --noEmit 2>&1 | head -50
```
Expected: no new errors from i18n changes.

- [ ] **Step 2: Build frontend**
```bash
cd D:/PycharmProjects/SDL_agent/frontend && npm run build:flask
```
Expected: successful Vite build, dist files generated.

- [ ] **Step 3: Verify dist/index.html references correct JS hash**
```bash
cd D:/PycharmProjects/SDL_agent && grep -oP 'src="[^"]+\.js"' frontend/dist/index.html
```
Expected: matches files in `frontend/dist/assets/`.

- [ ] **Step 4: Run Python syntax checks**
```bash
cd D:/PycharmProjects/SDL_agent && python -c "from utils.i18n import i18n; print(i18n.get('common.save', 'en'), i18n.get('common.save', 'zh'))"
```
Expected: `Save 保存`

- [ ] **Step 5: Run i18n consistency check**
```bash
cd D:/PycharmProjects/SDL_agent && python scripts/i18n_check.py
```
Expected: `OK: N keys match between en.json and zh.json`

- [ ] **Step 6: Start Flask and smoke test**
```bash
cd D:/PycharmProjects/SDL_agent && python digital_twin.py &
# Wait for startup
# Test: curl http://127.0.0.1:5000/v2 (should load SPA)
# Test: click language toggle in browser, verify UI switches
```

- [ ] **Step 7: Commit**
```bash
git add -A
git commit -m "feat: complete i18n language switching implementation"
```

---

## Execution Order

```
Task 1 (vue-i18n + en.json)
  └─> Task 2 (i18n init + store + main.ts)
        └─> Task 3 (backend I18nHelper)     [can run in PARALLEL]
        └─> Task 4 (Python tool scripts)    [can run in PARALLEL]
        └─> Task 5 (prompts restructure)    [can run in PARALLEL]
        └─> Task 6 (TopBar toggle)          [can run in PARALLEL]
              └─> Tasks 7A-7G (frontend migration)  [ALL 7 in PARALLEL]
                    └─> Task 7H (merge partials → en.json/zh.json)
                          └─> Task 8 (review audit)
                                └─> Task 9 (app.py migration)   [can run in PARALLEL]
                                └─> Task 10 (PromptManager callers) [can run in PARALLEL]
                                      └─> Task 11 (build + verify)
```

**Parallelization strategy:**

| Wave | Tasks | Agent count |
|------|-------|-------------|
| Wave 1 | Tasks 3, 4, 5, 6 (run after Task 2) | 4 agents |
| Wave 2 | Tasks 7A, 7B, 7C, 7D, 7E, 7F, 7G (run after Wave 1) | 7 agents |
| Merge | Task 7H (run after Wave 2) | 1 agent |
| Wave 3 | Task 8 (review, after Merge) | 1 agent |
| Wave 4 | Tasks 9, 10 (run after review passes) | 2 agents |
| Final | Task 11 (build + verify) | 1 agent |

**Review loop:** If Task 8 finds issues → send affected files back to the relevant 7A-7G agent for fix → re-merge → re-review. Max 2 retries.
