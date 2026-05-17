<script setup lang="ts">
import { useRoute } from 'vue-router'
import { useLayoutStore } from '@/stores/layout'

const route = useRoute()
const layout = useLayoutStore()

const navItems = [
  { path: '/',           name: 'chat',        label: 'modes.chat',               icon: '💬' },
  { path: '/extraction', name: 'extraction',  label: 'modes.literatureExtraction', icon: '📄' },
  { path: '/hardware',   name: 'hardware',    label: 'modes.hardwareControl',    icon: '⚙️' },
  { path: '/experiment', name: 'experiment',  label: 'modes.experimentDesign',   icon: '🧪' },
  { path: '/analysis',   name: 'analysis',    label: 'modes.dataAnalysis',       icon: '📈' },
]

function isActive(path: string): boolean {
  if (path === '/') return route.path === '/'
  return route.path.startsWith(path)
}
</script>

<template>
  <nav class="sidebar" :class="{ collapsed: layout.sidebarCollapsed }">
    <div class="sidebar-nav">
      <router-link
        v-for="item in navItems"
        :key="item.name"
        :to="item.path"
        class="nav-item"
        :class="{ active: isActive(item.path) }"
        :title="layout.sidebarCollapsed ? $t(item.label) : ''"
      >
        <span class="nav-icon">{{ item.icon }}</span>
        <span class="nav-label">{{ $t(item.label) }}</span>
      </router-link>
    </div>
  </nav>
</template>

<style scoped>
.sidebar {
  width: var(--sidebar-width);
  background: var(--color-surface);
  border-right: 1px solid var(--color-border);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  transition: width var(--transition-slow), opacity var(--transition-slow);
  overflow: hidden;
}

.sidebar.collapsed {
  width: 0;
  border-right: none;
}

.sidebar-nav {
  padding: var(--space-md);
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: var(--sidebar-width);
}

.nav-item {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: 10px 12px;
  border-radius: var(--radius-md);
  color: var(--color-text-secondary);
  text-decoration: none;
  font-size: 14px;
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
  width: 24px;
  text-align: center;
  flex-shrink: 0;
}

.nav-label {
  white-space: nowrap;
}
</style>
