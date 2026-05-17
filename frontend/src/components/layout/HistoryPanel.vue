<script setup lang="ts">
import { ref, computed, onMounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useChatStore } from '@/stores/chat'
import { useLayoutStore } from '@/stores/layout'
import {
  fetchSessions, fetchSession, deleteSession, updateSessionTitle,
  fetchFolders, createFolder, renameFolder, deleteFolder, moveSession,
  type SessionEntry, type Folder,
} from '@/api/history'

const { t } = useI18n()

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

// 删除弹窗
const deleteModalVisible = ref(false)
const pendingDeleteSession = ref<SessionEntry | null>(null)
const deletingTs = ref<string | null>(null)  // 动画中的会话 timestamp
const activeSessionTs = ref<string | null>(null)  // 当前选中的会话

// 拖拽状态（undefined=未拖拽, null=拖到"全部", string=拖到具体文件夹）
const dragOverFolderId = ref<string | null | undefined>(undefined)

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

async function loadSessions() {
  try {
    const data = await fetchSessions()
    sessions.value = data.sessions
      .filter(s => s.message_count > 0)
      .sort((a, b) => b.saved_at.localeCompare(a.saved_at))
  } catch {
    // silently fail
  } finally {
    loading.value = false
  }
}

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

async function startNewSession() {
  store.clear()
  try {
    const resp = await fetch('/api/get_session_path')
    const data = await resp.json()
    if (data.success && data.timestamp) {
      activeSessionTs.value = data.timestamp
      // 如果当前会话不在列表中，添加一个临时条目方便改名
      if (!sessions.value.some(s => s.timestamp === data.timestamp)) {
        sessions.value.unshift({
          timestamp: data.timestamp,
          started_at: new Date().toISOString(),
          saved_at: new Date().toISOString(),
          message_count: 0,
          title: null,
          path: data.timestamp,
        })
      }
    }
  } catch { /* silently fail */ }
}

onMounted(async () => {
  await loadSessions()
  await loadFolders()
  // Auto-create a new session on entry if no messages exist yet
  if (store.messages.length === 0) {
    await startNewSession()
  }
})

async function onSessionClick(s: SessionEntry) {
  if (restoring.value) return
  restoring.value = s.timestamp
  activeSessionTs.value = s.timestamp
  try {
    const resp = await fetchSession(s.timestamp)
    if (resp.success && resp.data.messages.length > 0) {
      store.loadMessages(resp.data.messages)
      layout.closeTaskPanel()
      router.push('/')
    }
  } catch {
    // silently fail
  } finally {
    restoring.value = null
  }
}

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

function showDeleteModal(s: SessionEntry) {
  pendingDeleteSession.value = s
  deleteModalVisible.value = true
}

function cancelDeleteModal() {
  deleteModalVisible.value = false
  pendingDeleteSession.value = null
}

async function confirmDeleteModal() {
  const s = pendingDeleteSession.value
  if (!s) return
  deleteModalVisible.value = false
  try {
    await deleteSession(s.timestamp)
    // 触发退出动画，动画结束后移除
    deletingTs.value = s.timestamp
    setTimeout(() => {
      sessions.value = sessions.value.filter(x => x.timestamp !== s.timestamp)
      deletingTs.value = null
      pendingDeleteSession.value = null
    }, 500)
  } catch {
    deleteModalVisible.value = false
    pendingDeleteSession.value = null
  }
}

// ── 拖拽：会话拖入文件夹 ──

function onSessionDragStart(e: DragEvent, s: SessionEntry) {
  e.dataTransfer!.effectAllowed = 'move'
  e.dataTransfer!.setData('text/plain', s.timestamp)
}

function onFolderDragOver(e: DragEvent, folderId: string | null) {
  e.preventDefault()
  e.dataTransfer!.dropEffect = 'move'
  dragOverFolderId.value = folderId
}

function onFolderDragLeave() {
  dragOverFolderId.value = undefined
}

async function onFolderDrop(e: DragEvent, folderId: string | null) {
  e.preventDefault()
  dragOverFolderId.value = undefined
  const ts = e.dataTransfer!.getData('text/plain')
  if (!ts || !/^\d{8}_\d{6}$/.test(ts)) return
  try {
    await moveSession(ts, folderId)
    const session = sessions.value.find(s => s.timestamp === ts)
    if (session) {
      if (folderId === null) {
        session.folder_id = undefined
      } else {
        session.folder_id = folderId
      }
    }
  } catch {
    // silently fail
  }
}

interface ModeItem {
  icon: string
  label: string
  panelType?: string
  chatMode?: string
}

const modes: ModeItem[] = [
  { icon: '💬', label: 'modes.chat',                               chatMode: 'normal' },
  { icon: '📄', label: 'modes.literatureExtraction',  panelType: 'extraction', chatMode: 'extraction' },
  { icon: '⚙️', label: 'modes.hardwareControl',   panelType: 'hardware',   chatMode: 'hardware' },
  { icon: '🧪', label: 'modes.experimentDesign',  panelType: 'experiment', chatMode: 'experiment' },
  { icon: '📈', label: 'modes.dataAnalysis',       panelType: 'analysis',   chatMode: 'analysis' },
]

function isActive(mode: ModeItem): boolean {
  if (mode.panelType) return layout.activeTaskPanel === mode.panelType
  return store.currentMode === 'normal' && !layout.activeTaskPanel
}

function onModeClick(mode: ModeItem) {
  if (mode.chatMode) store.setMode(mode.chatMode as any)

  if (mode.panelType) {
    layout.openTaskPanel(mode.panelType as any)
  } else {
    layout.closeTaskPanel()
    router.push('/')
  }
}

function formatDate(ts: string): string {
  if (ts.includes('T')) {
    const d = new Date(ts)
    return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}`
  }
  const m = ts.slice(4, 6)
  const d = ts.slice(6, 8)
  const h = ts.slice(9, 11)
  const min = ts.slice(11, 13)
  return `${parseInt(m)}/${parseInt(d)} ${h}:${min}`
}

function displayTitle(s: SessionEntry): string {
  if (s.title && s.title !== '未命名会话') return s.title
  return `${formatDate(s.started_at || s.timestamp)} ${t('history.conversationSuffix')}`
}
</script>

<template>
  <aside class="history-panel">
    <div class="mode-switchers">
      <button
        v-for="m in modes"
        :key="m.label"
        class="mode-btn"
        :class="{ active: isActive(m) }"
        :title="$t(m.label)"
        @click="onModeClick(m)"
      >
        <span class="mode-emoji">{{ m.icon }}</span>
        <span class="mode-label">{{ $t(m.label) }}</span>
      </button>
    </div>

    <!-- 文件夹 -->
    <div class="folder-section">
      <div class="folder-header">
        <span class="folder-title">{{ $t('history.folders') }}</span>
        <button class="icon-btn" :title="$t('history.createFolder')" @click="creatingFolder = true">+</button>
      </div>

      <!-- 新建文件夹输入框 -->
      <div v-if="creatingFolder" class="folder-input-row">
        <input
          v-model="newFolderName"
          class="title-input"
          :placeholder="$t('history.folderNamePlaceholder')"
          maxlength="50"
          @keydown.enter="onCreateFolder()"
          @keydown.escape="creatingFolder = false"
        />
        <button class="icon-btn" @click="onCreateFolder()">&#10003;</button>
        <button class="icon-btn" @click="creatingFolder = false">&#10005;</button>
      </div>

      <!-- 文件夹列表 -->
      <div class="folder-list">
        <div
          class="folder-item"
          :class="{ active: activeFolderId === null, 'drag-over': dragOverFolderId === null }"
          @click="activeFolderId = null"
          @dragover="onFolderDragOver($event, null)"
          @dragleave="onFolderDragLeave()"
          @drop="onFolderDrop($event, null)"
        >
          <span class="folder-icon">&#128193;</span>
          <span class="folder-name">{{ $t('history.uncategorized') }}</span>
        </div>
        <div
          v-for="f in folders"
          :key="f.id"
          class="folder-item"
          :class="{ active: activeFolderId === f.id, 'drag-over': dragOverFolderId === f.id }"
          @click="activeFolderId = f.id"
          @dragover="onFolderDragOver($event, f.id)"
          @dragleave="onFolderDragLeave()"
          @drop="onFolderDrop($event, f.id)"
        >
          <span class="folder-icon">&#128193;</span>
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
              :title="$t('history.rename')"
              @click.stop="renamingFolderId = f.id; renameFolderName = f.name"
            >&#9998;</button>
            <button class="icon-btn danger" :title="$t('common.delete')" @click.stop="onDeleteFolder(f)">&#10005;</button>
          </div>
        </div>
      </div>
    </div>

    <div class="section-divider" />

    <div class="history-header">
      <span class="history-title">{{ $t('history.sessions') }}</span>
      <span class="history-count" v-if="sessions.length">{{ sessions.length }}</span>
      <button class="icon-btn new-session-btn" :title="$t('history.newSession')" @click="startNewSession()">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="1.5" y="2.5" width="11" height="8" rx="2"/>
          <polygon points="4,10.5 5.5,13 7.5,10.5"/>
          <line x1="11" y1="5" x2="14" y2="5"/>
          <line x1="12.5" y1="3.5" x2="12.5" y2="6.5"/>
        </svg>
      </button>
    </div>
    <div class="history-list" v-if="!loading">
      <div
        v-for="s in filteredSessions"
        :key="s.timestamp"
        class="history-item"
        :class="{ restoring: restoring === s.timestamp, deleting: deletingTs === s.timestamp, active: activeSessionTs === s.timestamp }"
        draggable="true"
        @dragstart="onSessionDragStart($event, s)"
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
            <button class="icon-btn" @click="confirmEdit(s)">&#10003;</button>
            <button class="icon-btn" @click="cancelEdit()">&#10005;</button>
          </div>
          <!-- 显示标题 -->
          <div v-else class="history-item-title" :title="displayTitle(s)">
            {{ displayTitle(s) }}
          </div>
          <div class="history-item-meta">
            <span>{{ formatDate(s.started_at || s.timestamp) }}</span>
            <span>{{ s.message_count }}{{ $t('history.msgCountSuffix') }}</span>
            <span v-if="restoring === s.timestamp" class="restoring-hint">{{ $t('common.loading') }}</span>
          </div>
        </div>

        <!-- 操作按钮 -->
        <div v-if="editingId !== s.timestamp" class="history-item-actions">
          <button class="icon-btn" :title="$t('history.rename')" @click.stop="startEdit(s)">
            <span class="action-icon">&#9998;</span>
          </button>
          <button class="icon-btn" :title="$t('common.delete')" @click.stop="showDeleteModal(s)">
            <span class="action-icon">&#10005;</span>
          </button>
        </div>
      </div>
      <div v-if="filteredSessions.length === 0" class="history-empty">
        {{ activeFolderId ? $t('history.emptyFolder') : $t('history.noSessions') }}
      </div>
    </div>
    <div class="history-list" v-else>
      <div class="history-item skeleton" v-for="i in 5" :key="i">
        <div class="skeleton-line w-80"></div>
        <div class="skeleton-line w-40"></div>
      </div>
    </div>

    <!-- 删除确认弹窗 -->
    <Teleport to="body">
      <div v-if="deleteModalVisible" class="modal-overlay" @click.self="cancelDeleteModal()">
        <div class="modal-dialog">
          <div class="modal-header">{{ $t('history.confirmDelete') }}</div>
          <div class="modal-body">
            {{ $t('history.deleteConfirmBody', { title: pendingDeleteSession ? displayTitle(pendingDeleteSession) : '' }) }}
          </div>
          <div class="modal-footer">
            <button class="modal-btn cancel" @click="cancelDeleteModal()">{{ $t('common.cancel') }}</button>
            <button class="modal-btn confirm" @click="confirmDeleteModal()">{{ $t('history.confirmDelete') }}</button>
          </div>
        </div>
      </div>
    </Teleport>
  </aside>
</template>

<style scoped>
.history-panel {
  width: var(--right-panel-width);
  background: var(--color-surface);
  border-right: 1px solid var(--color-border);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  overflow: hidden;
}

.mode-switchers {
  padding: var(--space-md);
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.mode-btn {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: 10px 12px;
  border: none;
  border-radius: var(--radius-md);
  background: transparent;
  color: var(--color-text-secondary);
  font-size: 14px;
  cursor: pointer;
  transition: background var(--transition-fast), color var(--transition-fast);
  text-align: left;
  width: 100%;
}

.mode-btn:hover {
  background: var(--color-bg-soft);
  color: var(--color-text);
}

.mode-btn.active {
  background: var(--color-primary-soft);
  color: var(--color-primary);
}

.mode-emoji {
  font-size: 18px;
  width: 24px;
  text-align: center;
  flex-shrink: 0;
}

.mode-label { white-space: nowrap; }

.section-divider {
  height: 1px;
  background: var(--color-border);
  margin: var(--space-sm) var(--space-lg);
}

.history-header {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: var(--space-lg);
  flex-shrink: 0;
}

.history-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text);
}

.history-count {
  font-size: 11px;
  color: var(--color-text-tertiary);
  background: var(--color-bg-mute);
  padding: 1px 6px;
  border-radius: var(--radius-full);
}

.new-session-btn {
  margin-left: auto;
  flex-shrink: 0;
}

.history-list {
  flex: 1;
  overflow-y: auto;
  padding: var(--space-sm);
}

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

.history-item-title {
  font-size: 13px;
  color: var(--color-text);
  line-height: 1.4;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.history-item-meta {
  display: flex;
  gap: var(--space-md);
  margin-top: 4px;
  font-size: 11px;
  color: var(--color-text-tertiary);
}

.history-empty {
  padding: var(--space-xl);
  text-align: center;
  font-size: 13px;
  color: var(--color-text-tertiary);
}

.history-item.restoring {
  opacity: 0.6;
  pointer-events: none;
}

.history-item.active {
  background: var(--color-primary-soft);
  border-left: 3px solid var(--color-primary);
  padding-left: 9px;
}

.restoring-hint {
  color: var(--color-primary);
  font-size: 11px;
}

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

.skeleton { cursor: default; }
.skeleton-line {
  height: 10px;
  border-radius: var(--radius-sm);
  background: var(--color-bg-mute);
  margin-bottom: 6px;
}
.w-80 { width: 80%; }
.w-40 { width: 40%; }

/* ── 删除动画（仿 Mac 收起反向）── */
@keyframes deleteOut {
  0% {
    opacity: 1;
    transform: scale(1, 1) translate(0, 0);
    max-height: 60px;
    background: transparent;
  }
  30% {
    background: #fdd;
    max-height: 30px;
  }
  60% {
    background: #e88;
    max-height: 10px;
    padding-top: 0;
    padding-bottom: 0;
  }
  100% {
    opacity: 0;
    transform: scale(0.3, 0.1) translate(-24px, -16px);
    max-height: 0;
    padding: 0;
    margin: 0;
    border-radius: var(--radius-sm);
    background: #c00;
  }
}

.history-item.deleting {
  animation: deleteOut 0.45s cubic-bezier(0.4, 0, 0.6, 1) forwards;
  overflow: hidden;
  pointer-events: none;
}

/* ── 拖拽高亮 ── */
.folder-item.drag-over {
  background: var(--color-primary-soft);
  outline: 2px dashed var(--color-primary);
  outline-offset: -2px;
}

/* ── 删除弹窗 ── */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}

.modal-dialog {
  background: var(--color-surface);
  border-radius: var(--radius-lg);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
  width: 380px;
  max-width: 90vw;
  overflow: hidden;
}

.modal-header {
  padding: 16px 20px;
  font-size: 15px;
  font-weight: 600;
  color: var(--color-text);
  border-bottom: 1px solid var(--color-border);
}

.modal-body {
  padding: 16px 20px;
  font-size: 13px;
  color: var(--color-text-secondary);
  line-height: 1.6;
}

.modal-footer {
  padding: 12px 20px;
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  border-top: 1px solid var(--color-border);
}

.modal-btn {
  padding: 6px 16px;
  border: none;
  border-radius: var(--radius-sm);
  font-size: 13px;
  cursor: pointer;
  transition: background var(--transition-fast);
}

.modal-btn.cancel {
  background: var(--color-bg-mute);
  color: var(--color-text-secondary);
}

.modal-btn.cancel:hover {
  background: var(--color-bg-soft);
}

.modal-btn.confirm {
  background: var(--color-danger, #c00);
  color: #fff;
}

.modal-btn.confirm:hover {
  background: #a00;
}
</style>
