<script setup lang="ts">
import { useThemeStore } from '@/stores/theme'
import { useLayoutStore } from '@/stores/layout'
import { useSettingsStore } from '@/stores/settings'
import { Sun, Moon, PanelLeftClose, PanelLeft, PanelRightClose, PanelRight } from 'lucide-vue-next'
import IconButton from '@/components/common/IconButton.vue'

const theme = useThemeStore()
const layout = useLayoutStore()
const settings = useSettingsStore()
</script>

<template>
  <header class="topbar">
    <div class="topbar-left">
      <IconButton
        :title="layout.sidebarCollapsed ? $t('sidebar.expandPanel') : $t('sidebar.collapsePanel')"
        @click="layout.toggleSidebar()"
      >
        <PanelLeftClose v-if="layout.sidebarCollapsed" :size="18" />
        <PanelLeft v-else :size="18" />
      </IconButton>
      <span class="topbar-brand">SDL Agent</span>
    </div>
    <div class="topbar-right">
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
      <IconButton :title="theme.theme === 'dark' ? $t('topbar.lightMode') : $t('topbar.darkMode')" @click="theme.toggle()">
        <Sun v-if="theme.theme === 'dark'" :size="18" />
        <Moon v-else :size="18" />
      </IconButton>
      <IconButton
        :title="layout.rightPanelCollapsed ? $t('sidebar.expandNav') : $t('sidebar.collapseNav')"
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
.lang-toggle {
  display: flex;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  overflow: hidden;
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
.topbar-right {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
}
</style>
