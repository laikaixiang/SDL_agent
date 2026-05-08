<script setup lang="ts">
import { useThemeStore } from '@/stores/theme'
import { useLayoutStore } from '@/stores/layout'
import { Sun, Moon, ExternalLink, PanelLeftClose, PanelLeft, PanelRightClose, PanelRight } from 'lucide-vue-next'
import IconButton from '@/components/common/IconButton.vue'

const theme = useThemeStore()
const layout = useLayoutStore()
</script>

<template>
  <header class="topbar">
    <div class="topbar-left">
      <IconButton
        :title="layout.sidebarCollapsed ? '展开历史面板' : '收起历史面板'"
        @click="layout.toggleSidebar()"
      >
        <PanelLeftClose v-if="layout.sidebarCollapsed" :size="18" />
        <PanelLeft v-else :size="18" />
      </IconButton>
      <span class="topbar-brand">SDL Agent</span>
      <a href="/" class="old-link" title="返回旧版界面">
        <ExternalLink :size="12" />
        <span>旧版</span>
      </a>
    </div>
    <div class="topbar-right">
      <IconButton :title="theme.theme === 'dark' ? '亮色模式' : '暗色模式'" @click="theme.toggle()">
        <Sun v-if="theme.theme === 'dark'" :size="18" />
        <Moon v-else :size="18" />
      </IconButton>
      <IconButton
        :title="layout.rightPanelCollapsed ? '展开导航面板' : '收起导航面板'"
        @click="layout.toggleRightPanel()"
      >
        <PanelRightClose v-if="layout.rightPanelCollapsed" :size="18" />
        <PanelRight v-else :size="18" />
      </IconButton>
    </div>
  </header>
</template>

<style scoped>
.topbar {
  height: var(--navbar-height);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 var(--space-xl);
  background: var(--color-surface);
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}
.topbar-left {
  display: flex;
  align-items: center;
  gap: var(--space-md);
}
.topbar-brand {
  font-size: 16px;
  font-weight: 600;
  color: var(--color-text);
}
.old-link {
  display: flex; align-items: center; gap: 3px;
  font-size: 12px; color: var(--color-text-tertiary); text-decoration: none;
  padding: 3px 8px; border-radius: var(--radius-sm); border: 1px solid var(--color-border);
  transition: color var(--transition-fast), border-color var(--transition-fast);
}
.old-link:hover { color: var(--color-text-secondary); border-color: var(--color-border-strong); }
.topbar-right {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
}
</style>
