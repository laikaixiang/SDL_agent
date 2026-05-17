<script setup lang="ts">
import { computed, defineAsyncComponent } from 'vue'
import { useLayoutStore } from '@/stores/layout'
import { X } from 'lucide-vue-next'

const layout = useLayoutStore()

const panelComponents: Record<string, ReturnType<typeof defineAsyncComponent>> = {
  extraction: defineAsyncComponent(() => import('@/pages/ExtractionPage.vue')),
  hardware: defineAsyncComponent(() => import('@/pages/HardwarePage.vue')),
  analysis: defineAsyncComponent(() => import('@/pages/AnalysisPage.vue')),
  experiment: defineAsyncComponent(() => import('@/pages/ExperimentPage.vue')),
}

const panelLabels: Record<string, string> = {
  extraction: 'modes.literatureExtraction',
  hardware: 'modes.hardwareControl',
  analysis: 'modes.dataAnalysis',
  experiment: 'modes.experimentDesign',
}

const isWide = computed(() => layout.activeTaskPanel === 'experiment')
</script>

<template>
  <aside
    class="task-panel"
    :class="{ wide: isWide }"
    v-if="layout.activeTaskPanel"
  >
    <div class="task-header">
      <span class="task-title">{{ $t(panelLabels[layout.activeTaskPanel]) }}</span>
      <button class="task-close" :title="$t('common.close')" @click="layout.closeTaskPanel()">
        <X :size="16" />
      </button>
    </div>
    <div class="task-body">
      <component :is="panelComponents[layout.activeTaskPanel]" />
    </div>
  </aside>
</template>

<style scoped>
.task-panel {
  width: 360px;
  background: var(--color-surface);
  border-left: 1px solid var(--color-border);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  overflow: hidden;
}

.task-panel.wide {
  width: 700px;
}

.task-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-md) var(--space-lg);
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}

.task-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text);
}

.task-close {
  width: 28px; height: 28px; display: flex; align-items: center; justify-content: center;
  border: none; border-radius: var(--radius-sm); background: transparent;
  color: var(--color-text-tertiary); cursor: pointer;
  transition: background var(--transition-fast), color var(--transition-fast);
}

.task-close:hover { background: var(--color-bg-soft); color: var(--color-text); }

.task-body {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
}
</style>
