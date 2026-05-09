<script setup lang="ts">
import { ref } from 'vue'
import { useLayoutStore, type TaskEntry } from '@/stores/layout'
import { X } from 'lucide-vue-next'

const layout = useLayoutStore()

const hoveredTask = ref<string | null>(null)
const hoverTimer = ref<ReturnType<typeof setTimeout> | null>(null)

const panelIcons: Record<string, string> = {
  search: '🔍',
  extraction: '📄',
  hardware: '⚙️',
  experiment: '🧪',
  analysis: '📈',
}

const panelLabels: Record<string, string> = {
  search: '语义搜索',
  extraction: '文献提取',
  hardware: '硬件控制',
  experiment: '实验设计',
  analysis: '数据分析',
}

function onTaskClick(task: TaskEntry) {
  layout.acknowledgeTask(task.type)
  layout.openTaskPanel(task.type)
}

function onCloseClick(task: TaskEntry) {
  if (hoverTimer.value) {
    clearTimeout(hoverTimer.value)
    hoverTimer.value = null
  }
  hoveredTask.value = null
  layout.closeTask(task.type)
}

function onMouseEnter(type: string) {
  hoverTimer.value = setTimeout(() => {
    hoveredTask.value = type
  }, 500)
}

function onMouseLeave() {
  if (hoverTimer.value) {
    clearTimeout(hoverTimer.value)
    hoverTimer.value = null
  }
  hoveredTask.value = null
}
</script>

<template>
  <nav class="nav-panel">
    <div class="nav-list">
      <div
        v-for="task in layout.taskList"
        :key="task.type"
        class="task-item"
      >
        <button
          class="nav-item"
          :class="{ active: layout.activeTaskPanel === task.type }"
          :title="panelLabels[task.type]"
          @click="onTaskClick(task)"
          @mouseenter="onMouseEnter(task.type)"
          @mouseleave="onMouseLeave"
        >
          <span class="nav-icon">{{ panelIcons[task.type] }}</span>

          <span v-if="task.status === 'completed'" class="status-dot" />
          <span v-if="task.status === 'running'" class="status-spinner" />

          <span
            v-if="hoveredTask === task.type"
            class="status-close"
            @click.stop="onCloseClick(task)"
          >
            <X :size="10" />
          </span>
        </button>

        <div v-if="task.status === 'running'" class="task-progress">
          <div
            class="task-progress-fill"
            :style="{ width: task.progress + '%' }"
          />
        </div>
      </div>
    </div>
  </nav>
</template>

<style scoped>
.nav-panel {
  width: 52px;
  background: var(--color-surface);
  border-left: 1px solid var(--color-border);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  overflow: hidden;
}

.nav-list {
  flex: 1;
  overflow-y: auto;
  padding: var(--space-sm) 4px;
  display: flex;
  flex-direction: column;
  gap: 0;
  align-items: center;
}

.task-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: 2px;
}

.nav-item {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  border: none;
  border-radius: var(--radius-md);
  background: transparent;
  color: var(--color-text-secondary);
  cursor: pointer;
  transition: background var(--transition-fast), color var(--transition-fast);
  flex-shrink: 0;
}

.nav-item:hover {
  background: var(--color-bg-soft);
  color: var(--color-text);
}

.nav-item.active {
  background: var(--color-primary-soft);
  color: var(--color-primary);
}

.nav-icon {
  font-size: 18px;
  text-align: center;
}

/* Green dot — completed */
.status-dot {
  position: absolute;
  top: 2px;
  left: 2px;
  width: 7px;
  height: 7px;
  background: #10b981;
  border-radius: 50%;
  border: 1.5px solid var(--color-surface);
}

/* Spinner — running */
.status-spinner {
  position: absolute;
  bottom: 2px;
  right: 2px;
  width: 12px;
  height: 12px;
  border: 2px solid var(--color-border);
  border-top-color: var(--color-primary);
  border-radius: 50%;
  animation: task-spin 0.7s linear infinite;
}

/* Red X — close on hover */
.status-close {
  position: absolute;
  top: -2px;
  right: -2px;
  width: 18px;
  height: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 50%;
  color: #ef4444;
  cursor: pointer;
  transition: transform var(--transition-fast);
}

.status-close:hover {
  transform: scale(1.15);
}

/* Progress bar */
.task-progress {
  width: 32px;
  height: 2px;
  background: var(--color-bg-mute);
  border-radius: 1px;
  overflow: hidden;
  margin-top: 0;
}

.task-progress-fill {
  height: 100%;
  background: var(--color-primary);
  border-radius: 1px;
  transition: width 0.3s ease;
}

@keyframes task-spin {
  to { transform: rotate(360deg); }
}
</style>
