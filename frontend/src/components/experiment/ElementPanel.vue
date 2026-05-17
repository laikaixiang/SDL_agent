<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useExperimentStore } from '@/stores/experiment'
import type { HelperType } from '@/types/experiment'
import { Wrench, BarChart3, GitBranch, ChevronDown, ChevronRight } from 'lucide-vue-next'

const { t } = useI18n()
const store = useExperimentStore()

const toolsOpen = ref(true)
const algosOpen = ref(true)
const helpersOpen = ref(true)

const selectedTool = ref<string | null>(null)
const selectedAlgo = ref<string | null>(null)

const helperTypes: { type: HelperType; label: string }[] = [
  { type: 'LOOP', label: t('experiment.loop') },
  { type: 'GROUP', label: t('experiment.group') },
  { type: 'WAIT', label: t('experiment.wait') },
  { type: 'CONDITION', label: t('experiment.condition') },
  { type: 'END', label: t('experiment.end') },
  { type: 'USER_INPUT', label: t('experiment.input') },
]

onMounted(() => {
  store.loadHardwareTools()
  store.loadAlgorithms()
})

function onToolClick(toolName: string) {
  selectedTool.value = selectedTool.value === toolName ? null : toolName
}

function onToolDblClick(toolName: string) {
  const tool = store.hardwareTools.find(t => t.name === toolName)
  if (tool) store.addToolStep(tool)
}

function onAlgoClick(algoName: string) {
  selectedAlgo.value = selectedAlgo.value === algoName ? null : algoName
}

function onAlgoDblClick(algoName: string) {
  const algo = store.algorithms.find(a => a.name === algoName)
  if (algo) store.addAlgorithmStep(algo)
}

function onHelperClick(type: HelperType) {
  store.addHelperFunction(type)
}
</script>

<template>
  <aside class="element-panel">
    <!-- Tools -->
    <div class="elem-section">
      <button class="elem-section-header" @click="toolsOpen = !toolsOpen">
        <component :is="toolsOpen ? ChevronDown : ChevronRight" :size="12" />
        <Wrench :size="13" />
        <span>{{ $t('experiment.tools') }}</span>
        <span class="elem-count">{{ store.hardwareTools.length }}</span>
      </button>
      <div v-if="toolsOpen" class="elem-list">
        <button
          v-for="t in store.hardwareTools"
          :key="t.name"
          class="elem-item"
          :class="{ selected: selectedTool === t.name }"
          @click="onToolClick(t.name)"
          @dblclick="onToolDblClick(t.name)"
          :title="t.description"
        >
          <span class="elem-name">{{ t.name }}</span>
          <span v-if="selectedTool === t.name" class="elem-params">
            <span v-for="(v, k) in t.params" :key="k" class="param-hint">{{ k }}: {{ v.type }}</span>
          </span>
        </button>
        <div v-if="!store.hardwareTools.length" class="elem-empty">{{ $t('experiment.noTools') }}</div>
      </div>
    </div>

    <!-- Algorithms -->
    <div class="elem-section">
      <button class="elem-section-header" @click="algosOpen = !algosOpen">
        <component :is="algosOpen ? ChevronDown : ChevronRight" :size="12" />
        <BarChart3 :size="13" />
        <span>{{ $t('experiment.algorithms') }}</span>
        <span class="elem-count">{{ store.algorithms.length }}</span>
      </button>
      <div v-if="algosOpen" class="elem-list">
        <button
          v-for="a in store.algorithms"
          :key="a.name"
          class="elem-item"
          :class="{ selected: selectedAlgo === a.name }"
          @click="onAlgoClick(a.name)"
          @dblclick="onAlgoDblClick(a.name)"
          :title="a.description"
        >
          <span class="elem-name">{{ a.chinese_name || a.name }}</span>
        </button>
        <div v-if="!store.algorithms.length" class="elem-empty">{{ $t('experiment.noAlgorithms') }}</div>
      </div>
    </div>

    <!-- Helpers -->
    <div class="elem-section">
      <button class="elem-section-header" @click="helpersOpen = !helpersOpen">
        <component :is="helpersOpen ? ChevronDown : ChevronRight" :size="12" />
        <GitBranch :size="13" />
        <span>{{ $t('experiment.helpers') }}</span>
      </button>
      <div v-if="helpersOpen" class="elem-list helper-list">
        <button
          v-for="h in helperTypes"
          :key="h.type"
          class="helper-btn"
          @click="onHelperClick(h.type)"
        >
          {{ h.label }}
        </button>
      </div>
    </div>
  </aside>
</template>

<style scoped>
.element-panel {
  width: 180px;
  min-width: 180px;
  border-right: 1px solid var(--color-border);
  overflow-y: auto;
  background: var(--color-bg-soft);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}

.elem-section {
  border-bottom: 1px solid var(--color-border);
}

.elem-section-header {
  display: flex;
  align-items: center;
  gap: 4px;
  width: 100%;
  padding: 8px 10px;
  border: none;
  background: transparent;
  color: var(--color-text-secondary);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  text-align: left;
}

.elem-section-header:hover {
  background: var(--color-bg-mute);
}

.elem-count {
  margin-left: auto;
  font-size: 10px;
  color: var(--color-text-tertiary);
  background: var(--color-bg-mute);
  padding: 0 5px;
  border-radius: var(--radius-full);
}

.elem-list {
  padding: 2px 0;
}

.elem-item {
  display: block;
  width: 100%;
  padding: 6px 10px 6px 24px;
  border: none;
  background: transparent;
  color: var(--color-text);
  font-size: 12px;
  text-align: left;
  cursor: pointer;
  transition: background var(--transition-fast);
}

.elem-item:hover {
  background: var(--color-primary-mute);
}

.elem-item.selected {
  background: var(--color-primary-soft);
}

.elem-name {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.elem-params {
  display: flex;
  flex-wrap: wrap;
  gap: 2px;
  margin-top: 3px;
}

.param-hint {
  font-size: 10px;
  color: var(--color-text-tertiary);
  background: var(--color-bg-mute);
  padding: 0 4px;
  border-radius: 3px;
}

.elem-empty {
  padding: 8px 24px;
  font-size: 12px;
  color: var(--color-text-tertiary);
}

.helper-list {
  display: flex;
  flex-wrap: wrap;
  gap: 2px;
  padding: 4px 8px;
}

.helper-btn {
  padding: 3px 7px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-surface);
  color: var(--color-text-secondary);
  font-size: 11px;
  cursor: pointer;
  transition: background var(--transition-fast), border-color var(--transition-fast);
}

.helper-btn:hover {
  background: #fef3c7;
  border-color: #f59e0b;
  color: #92400e;
}
</style>
