<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useChatStore } from '@/stores/chat'
import { useLayoutStore } from '@/stores/layout'
import { fetchSessions, type SessionEntry } from '@/api/history'

const store = useChatStore()
const layout = useLayoutStore()
const router = useRouter()
const sessions = ref<SessionEntry[]>([])
const loading = ref(true)

onMounted(async () => {
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
})

interface ModeItem {
  icon: string
  label: string
  panelType?: string
  chatMode?: string
}

const modes: ModeItem[] = [
  { icon: '💬', label: '对话',                               chatMode: 'normal' },
  { icon: '📄', label: '文献提取',  panelType: 'extraction', chatMode: 'extraction' },
  { icon: '⚙️', label: '硬件控制',  panelType: 'hardware',   chatMode: 'hardware' },
  { icon: '🧪', label: '实验设计',  panelType: 'experiment', chatMode: 'experiment' },
  { icon: '📈', label: '数据分析',  panelType: 'analysis',   chatMode: 'analysis' },
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
  return `${formatDate(s.started_at || s.timestamp)} 的对话`
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
        :title="m.label"
        @click="onModeClick(m)"
      >
        <span class="mode-emoji">{{ m.icon }}</span>
        <span class="mode-label">{{ m.label }}</span>
      </button>
    </div>

    <div class="section-divider" />

    <div class="history-header">
      <span class="history-title">历史会话</span>
      <span class="history-count" v-if="sessions.length">{{ sessions.length }}</span>
    </div>
    <div class="history-list" v-if="!loading">
      <div
        v-for="s in sessions"
        :key="s.timestamp"
        class="history-item"
        :title="displayTitle(s)"
      >
        <div class="history-item-title">{{ displayTitle(s) }}</div>
        <div class="history-item-meta">
          <span>{{ formatDate(s.started_at || s.timestamp) }}</span>
          <span>{{ s.message_count }} 条消息</span>
        </div>
      </div>
      <div v-if="sessions.length === 0" class="history-empty">
        暂无历史会话
      </div>
    </div>
    <div class="history-list" v-else>
      <div class="history-item skeleton" v-for="i in 5" :key="i">
        <div class="skeleton-line w-80"></div>
        <div class="skeleton-line w-40"></div>
      </div>
    </div>
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

.history-list {
  flex: 1;
  overflow-y: auto;
  padding: var(--space-sm);
}

.history-item {
  padding: 10px 12px;
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: background var(--transition-fast);
}

.history-item:hover {
  background: var(--color-bg-soft);
}

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

.skeleton { cursor: default; }
.skeleton-line {
  height: 10px;
  border-radius: var(--radius-sm);
  background: var(--color-bg-mute);
  margin-bottom: 6px;
}
.w-80 { width: 80%; }
.w-40 { width: 40%; }
</style>
