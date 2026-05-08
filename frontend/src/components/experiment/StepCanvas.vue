<script setup lang="ts">
import { computed } from 'vue'
import { useExperimentStore } from '@/stores/experiment'
import { useLayoutStore } from '@/stores/layout'
import type { ExperimentStep } from '@/types/experiment'
import StepCard from './StepCard.vue'

const store = useExperimentStore()

const emit = defineEmits<{
  (e: 'edit', index: number): void
}>()

const draggedIndex = computed({
  get: () => store.draggedStepIndex,
  set: (v) => { store.draggedStepIndex = v },
})

function onDragStart(index: number, event: DragEvent) {
  draggedIndex.value = index
  if (event.dataTransfer) {
    event.dataTransfer.effectAllowed = 'move'
  }
}

function onDragOver(index: number, event: DragEvent) {
  event.preventDefault()
  if (event.dataTransfer) {
    event.dataTransfer.dropEffect = 'move'
  }
}

function onDrop(index: number) {
  if (draggedIndex.value !== null && draggedIndex.value !== index) {
    store.moveStep(draggedIndex.value, index)
  }
  draggedIndex.value = null
}

function onDragEnd() {
  draggedIndex.value = null
}
</script>

<template>
  <div class="step-canvas">
    <!-- Empty state -->
    <div v-if="!store.steps.length" class="canvas-empty">
      <div class="empty-icon">🧪</div>
      <p>从左侧双击工具/算法添加步骤</p>
      <p class="empty-hint">或点击工具栏 [AI 生成] 自动设计</p>
    </div>

    <!-- Step cards -->
    <div v-else class="canvas-list">
      <StepCard
        v-for="(step, i) in store.steps"
        :key="i"
        :step="step"
        :index="i"
        :total="store.steps.length"
        :editing="store.editingStepIndex === i"
        @edit="store.toggleEdit(i)"
        @remove="store.removeStep(i)"
        @move-up="store.moveStepUp(i)"
        @move-down="store.moveStepDown(i)"
        @dragstart="onDragStart(i, $event)"
        @dragover="onDragOver(i, $event)"
        @drop="onDrop(i)"
        @dragend="onDragEnd"
      />
    </div>
  </div>
</template>

<style scoped>
.step-canvas {
  flex: 1;
  overflow-y: auto;
  padding: var(--space-md);
  min-width: 0;
}

.canvas-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--color-text-tertiary);
  font-size: 14px;
  text-align: center;
  gap: var(--space-sm);
}

.empty-icon { font-size: 36px; opacity: 0.5; }
.empty-hint { font-size: 12px; }

.canvas-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-sm);
}
</style>
