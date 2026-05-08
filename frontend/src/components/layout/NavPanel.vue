<script setup lang="ts">
import { useRoute, useRouter } from 'vue-router'
import { useLayoutStore, type TaskPanelType } from '@/stores/layout'

const route = useRoute()
const router = useRouter()
const layout = useLayoutStore()

interface NavItem {
  icon: string
  label: string
  action: 'navigate' | 'panel'
  target: string
  panelType?: TaskPanelType
}

const navItems: NavItem[] = [
  { icon: '💬', label: '对话',       action: 'navigate', target: '/' },
  { icon: '🔍', label: '语义搜索',   action: 'panel',    target: '', panelType: 'search' },
  { icon: '📄', label: '文献提取',   action: 'panel',    target: '', panelType: 'extraction' },
  { icon: '⚙️', label: '硬件控制',   action: 'panel',    target: '', panelType: 'hardware' },
  { icon: '🧪', label: '实验设计',   action: 'panel',    target: '', panelType: 'experiment' },
  { icon: '📈', label: '数据分析',   action: 'panel',    target: '', panelType: 'analysis' },
]

function isActive(item: NavItem): boolean {
  if (item.action === 'navigate') return route.path === '/'
  return layout.activeTaskPanel === item.panelType
}

function onClick(item: NavItem) {
  if (item.action === 'navigate') {
    router.push('/')
  } else if (item.panelType) {
    layout.openTaskPanel(item.panelType)
  }
}
</script>

<template>
  <nav class="nav-panel">
    <div class="nav-list">
      <button
        v-for="item in navItems"
        :key="item.label"
        class="nav-item"
        :class="{ active: isActive(item) }"
        :title="item.label"
        @click="onClick(item)"
      >
        <span class="nav-icon">{{ item.icon }}</span>
      </button>
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
  gap: 2px;
  align-items: center;
}

.nav-item {
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
</style>
