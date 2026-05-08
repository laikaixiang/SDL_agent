<script setup lang="ts">
import TopBar from '@/components/layout/TopBar.vue'
import HistoryPanel from '@/components/layout/HistoryPanel.vue'
import NavPanel from '@/components/layout/NavPanel.vue'
import TaskPanel from '@/components/layout/TaskPanel.vue'
import { useLayoutStore } from '@/stores/layout'

const layout = useLayoutStore()
</script>

<template>
  <div class="app-shell">
    <HistoryPanel v-show="!layout.sidebarCollapsed" />
    <div class="app-main">
      <TopBar />
      <main class="app-content">
        <RouterView />
      </main>
    </div>
    <TaskPanel v-if="layout.activeTaskPanel" />
    <NavPanel v-show="!layout.rightPanelCollapsed" />
  </div>
</template>

<style scoped>
.app-shell {
  display: flex;
  height: 100vh;
  overflow: hidden;
}

.app-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.app-content {
  flex: 1;
  overflow-y: auto;
  background: var(--color-bg);
}
</style>
