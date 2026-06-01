# 历史会话管理 — 删除、改标题、文件夹 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add soft-delete, inline title editing, and folder-based session grouping to the left sidebar history panel.

**Architecture:** All data persists to JSON files under `dialogue data/history/` — `sessions_index.json` (gets `folder_id` and `deleted` fields), `folders.json` (new file for folder definitions). Backend adds 7 REST routes to `app.py`. Frontend modifies `HistoryPanel.vue` with new inline edit, delete confirmation, and folder list UI, plus new API wrappers in `history.ts`.

**Tech Stack:** Flask + Vue 3 + TypeScript, no new dependencies.

---

## File Map

| File | Role |
|------|------|
| `app.py` | Add 7 new routes: delete session, update title, folders CRUD, move session |
| `frontend/src/api/history.ts` | Add API wrappers + new types (Folder, update SessionEntry) |
| `frontend/src/stores/chat.ts` | No changes needed (existing `loadMessages` suffices) |
| `frontend/src/components/layout/HistoryPanel.vue` | All UI changes: edit/delete on items, folder sidebar, create/rename/delete folder |

---

### Task 1: Backend — Soft Delete Session

**Files:**
- Modify: `app.py` — add route after line 1885

- [ ] **Step 1: Add DELETE route for soft delete**

Add after `history_load_session` route (after line 1885):

```python
@app.route('/api/history/session/<timestamp>', methods=['DELETE'])
def history_delete_session(timestamp: str):
    """
    软删除会话：从 sessions_index.json 移除条目，保留文件夹和 chat_history.json。

    返回: { success: true } 或 { success: false, error }
    """
    if not re.match(r'^\d{8}_\d{6}$', timestamp):
        return jsonify({"success": False, "error": "无效的时间戳格式"}), 400

    index_path = os.path.join(config.DIALOGUE_DATA_DIR, "sessions_index.json")
    if not os.path.exists(index_path):
        return jsonify({"success": False, "error": "索引文件不存在"}), 404

    with open(index_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_len = len(data.get("sessions", []))
    data["sessions"] = [s for s in data.get("sessions", []) if s.get("timestamp") != timestamp]

    if len(data["sessions"]) == original_len:
        return jsonify({"success": False, "error": "会话不存在"}), 404

    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return jsonify({"success": True})
```

- [ ] **Step 2: Test with curl**

```bash
# 先确认会话存在
curl -s http://127.0.0.1:5000/api/history/session/20260507_190432 | python -c "import sys,json; print(json.load(sys.stdin)['success'])"
# → True

# 软删除
curl -s -X DELETE http://127.0.0.1:5000/api/history/session/20260507_190432
# → {"success": true}

# 再次查询 — 应返回 404
curl -s -X DELETE http://127.0.0.1:5000/api/history/session/20260507_190432
# → {"error": "会话不存在"}

# 验证文件还在
ls "dialogue data/history/20260507_190432/chat_history.json"
# → 文件仍存在

# sessions_index 中已移除
curl -s http://127.0.0.1:5000/api/history/sessions | python -c "import sys,json; d=json.load(sys.stdin); print(len([s for s in d['sessions'] if s['timestamp']=='20260507_190432']))"
# → 0
```

- [ ] **Step 3: Commit**

```bash
git add digital_twin.py
git commit -m "feat: add soft-delete session API (DELETE /api/history/session/<ts>)"
```

---

### Task 2: Backend — Update Session Title

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add PUT route for title update**

Add after the DELETE route:

```python
@app.route('/api/history/session/<timestamp>/title', methods=['PUT'])
def history_update_title(timestamp: str):
    """
    更新会话标题，同步写入 sessions_index.json 和 chat_history.json。

    Body: {"title": "新标题"}
    返回: { success: true, title }
    """
    if not re.match(r'^\d{8}_\d{6}$', timestamp):
        return jsonify({"success": False, "error": "无效的时间戳格式"}), 400

    data = request.get_json(force=True, silent=True)
    if not data or "title" not in data:
        return jsonify({"success": False, "error": "缺少 title 字段"}), 400

    new_title = data["title"].strip()
    if not new_title or len(new_title) > 100:
        return jsonify({"success": False, "error": "标题长度需在 1-100 字符之间"}), 400

    # 1. 更新 sessions_index.json
    index_path = os.path.join(config.DIALOGUE_DATA_DIR, "sessions_index.json")
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        for s in index_data.get("sessions", []):
            if s.get("timestamp") == timestamp:
                s["title"] = new_title
                break
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

    # 2. 更新 chat_history.json
    history_path = os.path.join(config.DIALOGUE_DATA_DIR, timestamp, "chat_history.json")
    if os.path.exists(history_path):
        with open(history_path, 'r', encoding='utf-8') as f:
            hist_data = json.load(f)
        hist_data["title"] = new_title
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(hist_data, f, ensure_ascii=False, indent=2)

    return jsonify({"success": True, "title": new_title})
```

- [ ] **Step 2: Test with curl**

```bash
curl -s -X PUT http://127.0.0.1:5000/api/history/session/20260507_190432/title \
  -H "Content-Type: application/json" \
  -d '{"title": "FAPbI3 钝化剂提取实验"}'
# → {"success": true, "title": "FAPbI3 钝化剂提取实验"}

# 验证 sessions_index
curl -s http://127.0.0.1:5000/api/history/sessions | python -c "import sys,json; d=json.load(sys.stdin); [print(s['title']) for s in d['sessions'] if s['timestamp']=='20260507_190432']"
# → FAPbI3 钝化剂提取实验

# 验证 chat_history.json
cat "dialogue data/history/20260507_190432/chat_history.json" | python -c "import sys,json; print(json.load(sys.stdin)['title'])"
# → FAPbI3 钝化剂提取实验
```

- [ ] **Step 3: Commit**

```bash
git add digital_twin.py
git commit -m "feat: add update session title API (PUT /api/history/session/<ts>/title)"
```

---

### Task 3: Backend — Folders CRUD + Move Session

**Files:**
- Modify: `app.py`
- Create: `dialogue data/history/folders.json` (auto-created on first folder creation)

- [ ] **Step 1: Add helper function for folders.json**

Add near the top of app.py, next to `_update_session_index` (after line 152):

```python
FOLDERS_PATH = os.path.join(config.DIALOGUE_DATA_DIR, "folders.json")

def _read_folders():
    """读取 folders.json，不存在则返回空列表。"""
    if os.path.exists(FOLDERS_PATH):
        with open(FOLDERS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f).get("folders", [])
    return []

def _write_folders(folders: list):
    """写入 folders.json。"""
    with open(FOLDERS_PATH, 'w', encoding='utf-8') as f:
        json.dump({"folders": folders}, f, ensure_ascii=False, indent=2)
```

- [ ] **Step 2: Add folders CRUD routes**

Add 4 routes in the history section:

```python
@app.route('/api/history/folders', methods=['GET'])
def history_list_folders():
    """返回所有文件夹列表。"""
    return jsonify({"success": True, "folders": _read_folders()})


@app.route('/api/history/folders', methods=['POST'])
def history_create_folder():
    """
    创建文件夹。

    Body: {"name": "钙钛矿实验"}
    返回: { success: true, folder: { id, name, created_at } }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"success": False, "error": "请求体为空"}), 400

    name = (data.get("name") or "").strip()
    if not name or len(name) > 50:
        return jsonify({"success": False, "error": "文件夹名需在 1-50 字符之间"}), 400

    folders = _read_folders()
    folder = {
        "id": str(uuid.uuid4())[:8],
        "name": name,
        "created_at": datetime.now().isoformat()
    }
    folders.append(folder)
    _write_folders(folders)

    return jsonify({"success": True, "folder": folder})


@app.route('/api/history/folders/<folder_id>', methods=['PUT'])
def history_rename_folder(folder_id: str):
    """
    重命名文件夹。

    Body: {"name": "新名称"}
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"success": False, "error": "请求体为空"}), 400

    name = (data.get("name") or "").strip()
    if not name or len(name) > 50:
        return jsonify({"success": False, "error": "文件夹名需在 1-50 字符之间"}), 400

    folders = _read_folders()
    for f in folders:
        if f["id"] == folder_id:
            f["name"] = name
            _write_folders(folders)
            return jsonify({"success": True, "folder": f})

    return jsonify({"success": False, "error": "文件夹不存在"}), 404


@app.route('/api/history/folders/<folder_id>', methods=['DELETE'])
def history_delete_folder(folder_id: str):
    """
    删除文件夹，该文件夹下的所有会话变为未分类（移除 folder_id）。
    """
    folders = _read_folders()
    folders = [f for f in folders if f["id"] != folder_id]
    _write_folders(folders)

    # 清除 sessions_index 中该文件夹的关联
    index_path = os.path.join(config.DIALOGUE_DATA_DIR, "sessions_index.json")
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        for s in index_data.get("sessions", []):
            if s.get("folder_id") == folder_id:
                s.pop("folder_id", None)
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

    return jsonify({"success": True})
```

- [ ] **Step 3: Add move session to folder route**

```python
@app.route('/api/history/session/<timestamp>/move', methods=['PUT'])
def history_move_session(timestamp: str):
    """
    移动会话到指定文件夹（或移除文件夹关联）。

    Body: {"folder_id": "a1b2c3d4"}  或  {"folder_id": null}  移除关联
    返回: { success: true }
    """
    if not re.match(r'^\d{8}_\d{6}$', timestamp):
        return jsonify({"success": False, "error": "无效的时间戳格式"}), 400

    data = request.get_json(force=True, silent=True)
    if not data or "folder_id" not in data:
        return jsonify({"success": False, "error": "缺少 folder_id 字段"}), 400

    folder_id = data["folder_id"]  # None / null means remove from folder

    # 如果指定了 folder_id，验证文件夹存在
    if folder_id is not None:
        folders = _read_folders()
        if not any(f["id"] == folder_id for f in folders):
            return jsonify({"success": False, "error": "文件夹不存在"}), 404

    # 更新 sessions_index.json
    index_path = os.path.join(config.DIALOGUE_DATA_DIR, "sessions_index.json")
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        found = False
        for s in index_data.get("sessions", []):
            if s.get("timestamp") == timestamp:
                found = True
                if folder_id is None:
                    s.pop("folder_id", None)
                else:
                    s["folder_id"] = folder_id
                break
        if not found:
            return jsonify({"success": False, "error": "会话不存在"}), 404
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

    return jsonify({"success": True})
```

- [ ] **Step 4: Test folders CRUD**

```bash
# 创建文件夹
curl -s -X POST http://127.0.0.1:5000/api/history/folders \
  -H "Content-Type: application/json" \
  -d '{"name": "钙钛矿实验"}'
# → {"success": true, "folder": {"id": "...", "name": "钙钛矿实验", ...}}

# 列出文件夹
curl -s http://127.0.0.1:5000/api/history/folders
# → {"success": true, "folders": [...]}

# 移动会话到文件夹（用实际的 folder_id 和 timestamp 替换）
curl -s -X PUT http://127.0.0.1:5000/api/history/session/20260507_190432/move \
  -H "Content-Type: application/json" \
  -d '{"folder_id": "abc12345"}'

# 验证
curl -s http://127.0.0.1:5000/api/history/sessions | python -c "import sys,json; [print(s.get('folder_id')) for s in json.load(sys.stdin)['sessions'] if s['timestamp']=='20260507_190432']"

# 重命名文件夹
curl -s -X PUT http://127.0.0.1:5000/api/history/folders/abc12345 \
  -H "Content-Type: application/json" \
  -d '{"name": "材料合成实验"}'

# 删除文件夹
curl -s -X DELETE http://127.0.0.1:5000/api/history/folders/abc12345
```

- [ ] **Step 5: Commit**

```bash
git add digital_twin.py
git commit -m "feat: add folders CRUD + move session API"
```

---

### Task 4: Frontend API — New TypeScript Wrappers

**Files:**
- Modify: `frontend/src/api/history.ts`

- [ ] **Step 1: Add new types and API functions**

Replace the entire file content:

```typescript
import { request } from './client'

export interface SessionEntry {
  timestamp: string
  started_at: string
  saved_at: string
  message_count: number
  title: string | null
  path: string
  folder_id?: string
}

export interface SessionsIndex {
  sessions: SessionEntry[]
}

export interface SessionData {
  title: string
  messages: { role: string; content: string; timestamp?: string; mode?: string }[]
  outputs: Record<string, string[]>
}

export interface Folder {
  id: string
  name: string
  created_at: string
}

// ── Sessions ──

export async function fetchSessions(): Promise<SessionsIndex> {
  return request<SessionsIndex>('/api/history/sessions')
}

export async function fetchSession(timestamp: string): Promise<{ success: boolean; data: SessionData }> {
  return request(`/api/history/session/${timestamp}`)
}

export async function saveHistoryBatch(messages: unknown[]): Promise<{ success: boolean; saved_count: number }> {
  return request('/api/history/save_batch', {
    method: 'POST',
    body: { messages },
    timeout: 10000,
  })
}

export async function deleteSession(timestamp: string): Promise<{ success: boolean }> {
  return request(`/api/history/session/${timestamp}`, { method: 'DELETE' })
}

export async function updateSessionTitle(timestamp: string, title: string): Promise<{ success: boolean; title: string }> {
  return request(`/api/history/session/${timestamp}/title`, {
    method: 'PUT',
    body: { title },
  })
}

export async function moveSession(timestamp: string, folder_id: string | null): Promise<{ success: boolean }> {
  return request(`/api/history/session/${timestamp}/move`, {
    method: 'PUT',
    body: { folder_id },
  })
}

// ── Folders ──

export async function fetchFolders(): Promise<{ success: boolean; folders: Folder[] }> {
  return request('/api/history/folders')
}

export async function createFolder(name: string): Promise<{ success: boolean; folder: Folder }> {
  return request('/api/history/folders', {
    method: 'POST',
    body: { name },
  })
}

export async function renameFolder(id: string, name: string): Promise<{ success: boolean; folder: Folder }> {
  return request(`/api/history/folders/${id}`, {
    method: 'PUT',
    body: { name },
  })
}

export async function deleteFolder(id: string): Promise<{ success: boolean }> {
  return request(`/api/history/folders/${id}`, { method: 'DELETE' })
}
```

- [ ] **Step 2: Type-check**

```bash
cd frontend && npx vue-tsc -b
# Expected: no output (no errors)
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/api/history.ts
git commit -m "feat: add history management API wrappers (delete, rename, folders)"
```

---

### Task 5: Frontend UI — Inline Edit Title & Delete

**Files:**
- Modify: `frontend/src/components/layout/HistoryPanel.vue`

- [ ] **Step 1: Update imports and add state variables**

Replace the `<script setup>` import line and state declarations:

```typescript
import { ref, onMounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { useChatStore } from '@/stores/chat'
import { useLayoutStore } from '@/stores/layout'
import {
  fetchSessions, fetchSession, deleteSession, updateSessionTitle,
  type SessionEntry,
} from '@/api/history'

const store = useChatStore()
const layout = useLayoutStore()
const router = useRouter()
const sessions = ref<SessionEntry[]>([])
const loading = ref(true)
const restoring = ref<string | null>(null)

// 编辑状态
const editingId = ref<string | null>(null)
const editTitle = ref('')
const editInput = ref<HTMLInputElement | null>(null)

// 删除确认
const deletingId = ref<string | null>(null)
```

- [ ] **Step 2: Add edit and delete functions**

Add after `onSessionClick`:

```typescript
function startEdit(s: SessionEntry) {
  editingId.value = s.timestamp
  editTitle.value = s.title && s.title !== '未命名会话' ? s.title : ''
  nextTick(() => editInput.value?.focus())
}

async function confirmEdit(s: SessionEntry) {
  const newTitle = editTitle.value.trim() || displayTitle(s)
  try {
    await updateSessionTitle(s.timestamp, newTitle)
    s.title = newTitle
  } catch {
    // silently fail
  } finally {
    editingId.value = null
  }
}

function cancelEdit() {
  editingId.value = null
}

async function confirmDelete(s: SessionEntry) {
  if (deletingId.value) return
  deletingId.value = s.timestamp
  try {
    await deleteSession(s.timestamp)
    sessions.value = sessions.value.filter(x => x.timestamp !== s.timestamp)
  } catch {
    // silently fail
  } finally {
    deletingId.value = null
  }
}
```

- [ ] **Step 3: Update template — replace the history-item div**

Replace the existing `.history-item` div block:

```html
<div
  v-for="s in sessions"
  :key="s.timestamp"
  class="history-item"
  :class="{ restoring: restoring === s.timestamp }"
>
  <!-- 点击恢复对话 -->
  <div class="history-item-main" @click="onSessionClick(s)">
    <!-- 编辑标题 -->
    <div v-if="editingId === s.timestamp" class="history-item-title edit-row" @click.stop>
      <input
        ref="editInput"
        v-model="editTitle"
        class="title-input"
        maxlength="100"
        @keydown.enter="confirmEdit(s)"
        @keydown.escape="cancelEdit()"
      />
      <button class="icon-btn" @click="confirmEdit(s)">✓</button>
      <button class="icon-btn" @click="cancelEdit()">✕</button>
    </div>
    <!-- 显示标题 -->
    <div v-else class="history-item-title" :title="displayTitle(s)">
      {{ displayTitle(s) }}
    </div>
    <div class="history-item-meta">
      <span>{{ formatDate(s.started_at || s.timestamp) }}</span>
      <span>{{ s.message_count }} 条消息</span>
      <span v-if="restoring === s.timestamp" class="restoring-hint">加载中...</span>
    </div>
  </div>

  <!-- 操作按钮 -->
  <div v-if="editingId !== s.timestamp" class="history-item-actions">
    <button class="icon-btn" title="重命名" @click.stop="startEdit(s)">
      <span class="action-icon">✎</span>
    </button>
    <button
      v-if="deletingId === s.timestamp"
      class="icon-btn danger"
      title="确认删除"
      @click.stop="confirmDelete(s)"
    >
      <span class="action-icon">⚠</span>
    </button>
    <button
      v-else
      class="icon-btn"
      title="删除"
      @click.stop="deletingId = s.timestamp"
    >
      <span class="action-icon">✕</span>
    </button>
  </div>
</div>
```

- [ ] **Step 4: Add CSS for new elements**

Add to the `<style scoped>` block:

```css
.history-item {
  display: flex;
  align-items: flex-start;
  gap: 4px;
  padding: 10px 12px;
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: background var(--transition-fast);
}

.history-item:hover {
  background: var(--color-bg-soft);
}

.history-item-main {
  flex: 1;
  min-width: 0;
}

.history-item-actions {
  display: none;
  gap: 2px;
  flex-shrink: 0;
  padding-top: 2px;
}

.history-item:hover .history-item-actions {
  display: flex;
}

.icon-btn {
  width: 24px;
  height: 24px;
  border: none;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--color-text-tertiary);
  font-size: 13px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background var(--transition-fast), color var(--transition-fast);
}

.icon-btn:hover {
  background: var(--color-bg-mute);
  color: var(--color-text);
}

.icon-btn.danger:hover {
  background: var(--color-danger-soft, #fdd);
  color: var(--color-danger, #c00);
}

.edit-row {
  display: flex;
  gap: 4px;
  align-items: center;
}

.title-input {
  flex: 1;
  padding: 2px 6px;
  border: 1px solid var(--color-primary);
  border-radius: var(--radius-sm);
  font-size: 13px;
  background: var(--color-surface);
  color: var(--color-text);
  outline: none;
}

.action-icon { line-height: 1; }
```

- [ ] **Step 5: Type-check and build**

```bash
cd frontend && npx vue-tsc -b && npm run build:flask
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/layout/HistoryPanel.vue
git commit -m "feat: add inline title edit and soft-delete to history panel"
```

---

### Task 6: Frontend UI — Folder Sidebar

**Files:**
- Modify: `frontend/src/components/layout/HistoryPanel.vue`

- [ ] **Step 1: Add folder state and functions to script**

Add imports:

```typescript
import {
  fetchSessions, fetchSession, deleteSession, updateSessionTitle,
  fetchFolders, createFolder, renameFolder, deleteFolder, moveSession,
  type SessionEntry, type Folder,
} from '@/api/history'
```

Add state after existing `deletingId`:

```typescript
// 文件夹
const folders = ref<Folder[]>([])
const activeFolderId = ref<string | null>(null)  // null = "全部"
const creatingFolder = ref(false)
const newFolderName = ref('')
const renamingFolderId = ref<string | null>(null)
const renameFolderName = ref('')

// 过滤后的会话
const filteredSessions = computed(() => {
  if (activeFolderId.value === null) return sessions.value
  return sessions.value.filter(s => s.folder_id === activeFolderId.value)
})
```

Add `computed` to the Vue import:

```typescript
import { ref, computed, onMounted, nextTick } from 'vue'
```

Add folder functions after the edit/delete functions:

```typescript
async function loadFolders() {
  try {
    const resp = await fetchFolders()
    folders.value = resp.folders
  } catch { /* silently fail */ }
}

async function onCreateFolder() {
  const name = newFolderName.value.trim()
  if (!name) { creatingFolder.value = false; return }
  try {
    const resp = await createFolder(name)
    folders.value.push(resp.folder)
  } catch { /* silently fail */ }
  newFolderName.value = ''
  creatingFolder.value = false
}

async function onRenameFolder(f: Folder) {
  const name = renameFolderName.value.trim()
  if (!name || !renamingFolderId.value) { renamingFolderId.value = null; return }
  try {
    await renameFolder(f.id, name)
    f.name = name
  } catch { /* silently fail */ }
  renamingFolderId.value = null
}

async function onDeleteFolder(f: Folder) {
  try {
    await deleteFolder(f.id)
    folders.value = folders.value.filter(x => x.id !== f.id)
    sessions.value.forEach(s => { if (s.folder_id === f.id) s.folder_id = undefined })
    if (activeFolderId.value === f.id) activeFolderId.value = null
  } catch { /* silently fail */ }
}
```

Add `loadFolders()` to `onMounted`:

```typescript
onMounted(() => {
  loadSessions()
  loadFolders()
})
```

- [ ] **Step 2: Add folder UI to template**

Insert between the mode-switchers div and the section-divider:

```html
    <!-- 文件夹 -->
    <div class="folder-section">
      <div class="folder-header">
        <span class="folder-title">文件夹</span>
        <button class="icon-btn" title="新建文件夹" @click="creatingFolder = true">+</button>
      </div>

      <!-- 新建文件夹输入框 -->
      <div v-if="creatingFolder" class="folder-input-row">
        <input
          v-model="newFolderName"
          class="title-input"
          placeholder="文件夹名称"
          maxlength="50"
          @keydown.enter="onCreateFolder()"
          @keydown.escape="creatingFolder = false"
        />
        <button class="icon-btn" @click="onCreateFolder()">✓</button>
        <button class="icon-btn" @click="creatingFolder = false">✕</button>
      </div>

      <!-- 文件夹列表 -->
      <div class="folder-list">
        <div
          class="folder-item"
          :class="{ active: activeFolderId === null }"
          @click="activeFolderId = null"
        >
          <span class="folder-icon">📁</span>
          <span class="folder-name">全部</span>
        </div>
        <div
          v-for="f in folders"
          :key="f.id"
          class="folder-item"
          :class="{ active: activeFolderId === f.id }"
          @click="activeFolderId = f.id"
        >
          <span class="folder-icon">📁</span>
          <!-- 重命名模式 -->
          <input
            v-if="renamingFolderId === f.id"
            v-model="renameFolderName"
            class="title-input folder-rename-input"
            maxlength="50"
            @keydown.enter="onRenameFolder(f)"
            @keydown.escape="renamingFolderId = null"
            @click.stop
          />
          <span v-else class="folder-name">{{ f.name }}</span>
          <!-- 文件夹操作 -->
          <div class="folder-actions">
            <button
              class="icon-btn"
              title="重命名"
              @click.stop="renamingFolderId = f.id; renameFolderName = f.name"
            >✎</button>
            <button class="icon-btn danger" title="删除" @click.stop="onDeleteFolder(f)">✕</button>
          </div>
        </div>
      </div>
    </div>
```

- [ ] **Step 3: Add folder CSS**

```css
.folder-section {
  padding: var(--space-sm) var(--space-lg);
  flex-shrink: 0;
}

.folder-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: var(--space-sm);
}

.folder-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text-secondary);
}

.folder-input-row {
  display: flex;
  gap: 4px;
  margin-bottom: var(--space-sm);
}

.folder-list {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.folder-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 8px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: 13px;
  color: var(--color-text-secondary);
  transition: background var(--transition-fast);
}

.folder-item:hover {
  background: var(--color-bg-soft);
}

.folder-item.active {
  background: var(--color-primary-soft);
  color: var(--color-primary);
}

.folder-icon { flex-shrink: 0; }
.folder-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.folder-actions {
  display: none;
  gap: 2px;
}

.folder-item:hover .folder-actions {
  display: flex;
}

.folder-rename-input {
  flex: 1;
  min-width: 0;
}
```

- [ ] **Step 4: Change the history list to use filteredSessions**

Replace `sessions` with `filteredSessions` in the template:

```html
<div class="history-list" v-if="!loading">
  <div
    v-for="s in filteredSessions"
    ...
```

And update the empty state:

```html
<div v-if="filteredSessions.length === 0" class="history-empty">
  {{ activeFolderId ? '此文件夹为空' : '暂无历史会话' }}
</div>
```

- [ ] **Step 5: Type-check and build**

```bash
cd frontend && npx vue-tsc -b && npm run build:flask
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/layout/HistoryPanel.vue
git commit -m "feat: add folder sidebar with create/rename/delete and session filtering"
```

---

## Verification Checklist

After all tasks complete, verify end-to-end:

1. Restart Flask, hard-refresh browser
2. Send a few messages → verify `chat_history.json` created
3. **Edit title**: Click ✎ on a history item → type new title → Enter → title updates
4. **Delete**: Click ✕ → clicks again to confirm ⚠ → item disappears from list; verify file still exists
5. **Create folder**: Click + → type name → Enter → folder appears
6. **Rename folder**: Click ✎ on folder → type new name → Enter
7. **Delete folder**: Click ✕ on folder → folder and its sessions' folder_id are cleared
8. **Filter**: Click a folder → only sessions in that folder shown; click "全部" → all shown

---

## Design Decisions

**Why no `deleted` field in sessions_index?** Simple approach: remove from index entirely. Re-adding is automatic if the session folder still exists and gets `save_batch` called again (e.g., via sendBeacon or future message). No need for a complex undelete flow.

**Why UUID[:8] for folder IDs?** Short enough for URLs/debugging (8 hex chars), collision probability negligible for this scale (~10^3 folders max).

**Why no drag-and-drop?** User asked for simple. Drag-to-folder adds significant complexity (drag event handling, drop zones, touch support). Can be added later.

**Why `folder_id` on SessionEntry instead of a separate mapping?** Keeps data where it's used — one read of `sessions_index.json` gives the complete picture. Avoids cross-file joins in frontend code.
