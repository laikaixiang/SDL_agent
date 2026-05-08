<script setup lang="ts">
import type { ExperimentStep } from '@/types/experiment'
import Badge from '@/components/common/Badge.vue'
import StepEditor from './StepEditor.vue'
import { ArrowUp, ArrowDown, Pencil, Trash2, GripVertical } from 'lucide-vue-next'

defineProps<{
  step: ExperimentStep
  index: number
  total: number
  editing: boolean
}>()

const emit = defineEmits<{
  (e: 'edit'): void
  (e: 'remove'): void
  (e: 'moveUp'): void
  (e: 'moveDown'): void
  (e: 'dragstart', payload: DragEvent): void
  (e: 'dragover', payload: DragEvent): void
  (e: 'drop'): void
  (e: 'dragend'): void
}>()

function typeVariant(t: string) {
  return t === 'tool' ? 'success' : t === 'software' ? 'default' : 'warning'
}
</script>

<template>
  <div
    class="step-card"
    :class="{ editing }"
    draggable="true"
    @dragstart="emit('dragstart', $event)"
    @dragover="emit('dragover', $event)"
    @drop="emit('drop')"
    @dragend="emit('dragend')"
  >
    <div class="step-main">
      <span class="step-grip" title="拖拽排序"><GripVertical :size="14" /></span>
      <span class="step-num">{{ index + 1 }}</span>
      <div class="step-body">
        <div class="step-head">
          <Badge :variant="typeVariant(step.type)">{{ step.type }}</Badge>
          <span class="step-name">{{ step.name }}</span>
        </div>
        <div class="step-desc" v-if="step.description">{{ step.description }}</div>
      </div>
      <div class="step-actions">
        <button class="step-btn" title="上移" :disabled="index === 0" @click="emit('moveUp')"><ArrowUp :size="12" /></button>
        <button class="step-btn" title="下移" :disabled="index === total - 1" @click="emit('moveDown')"><ArrowDown :size="12" /></button>
        <button class="step-btn" title="编辑" @click="emit('edit')"><Pencil :size="12" /></button>
        <button class="step-btn step-btn-danger" title="删除" @click="emit('remove')"><Trash2 :size="12" /></button>
      </div>
    </div>

    <StepEditor
      v-if="editing"
      :step="step"
      :index="index"
      @close="emit('edit')"
    />
  </div>
</template>

<style scoped>
.step-card {
  background: var(--color-surface);
  border: 2px solid var(--color-border);
  border-radius: var(--radius-md);
  cursor: default;
  transition: border-color var(--transition-fast);
  overflow: hidden;
}

.step-card:hover {
  border-color: var(--color-primary-soft);
}

.step-card.editing {
  border-color: var(--color-primary);
}

.step-main {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: var(--space-md);
}

.step-grip {
  color: var(--color-text-tertiary);
  cursor: grab;
  flex-shrink: 0;
}

.step-num {
  width: 26px; height: 26px;
  border-radius: 50%;
  background: var(--color-primary-soft);
  color: var(--color-primary);
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 600;
  flex-shrink: 0;
}

.step-body { flex: 1; min-width: 0; }

.step-head {
  display: flex; align-items: center; gap: var(--space-sm);
}

.step-name { font-weight: 600; font-size: 13px; }

.step-desc {
  font-size: 12px; color: var(--color-text-secondary);
  margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

.step-actions {
  display: flex; gap: 2px; flex-shrink: 0;
}

.step-btn {
  width: 26px; height: 26px;
  display: flex; align-items: center; justify-content: center;
  border: none; border-radius: var(--radius-sm);
  background: transparent;
  color: var(--color-text-tertiary);
  cursor: pointer;
  transition: background var(--transition-fast), color var(--transition-fast);
}

.step-btn:hover:not(:disabled) {
  background: var(--color-bg-soft);
  color: var(--color-text);
}

.step-btn:disabled { opacity: 0.3; cursor: default; }
.step-btn-danger:hover:not(:disabled) { color: var(--color-error); }
</style>
