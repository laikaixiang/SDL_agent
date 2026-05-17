<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import type { ExperimentStep } from '@/types/experiment'
import Badge from '@/components/common/Badge.vue'
import StepEditor from './StepEditor.vue'
import { ArrowUp, ArrowDown, Pencil, Trash2 } from 'lucide-vue-next'

const { t } = useI18n()

defineProps<{
  step: ExperimentStep
  index: number
  total: number
  level: number
  isBlockStart: boolean
  isBlockEnd: boolean
  editing: boolean
}>()

const emit = defineEmits<{
  (e: 'edit'): void
  (e: 'remove'): void
  (e: 'moveUp'): void
  (e: 'moveDown'): void
}>()

function typeVariant(t: string) {
  return t === 'tool' ? 'success' : t === 'software' ? 'default' : 'warning'
}

function getHelperLabel(name: string): string {
  const labels: Record<string, string> = {
    LOOP: t('experiment.loop'), GROUP: t('experiment.group'), CONDITION: t('experiment.condition'),
    WAIT: t('experiment.wait'), END: t('experiment.end'), USER_INPUT: t('experiment.input'),
  }
  return labels[name] || name
}
</script>

<template>
  <div
    class="step-card"
    :class="{
      editing,
      'is-block-start': isBlockStart,
      'is-block-end': isBlockEnd,
    }"
  >
    <div class="step-main">
      <div class="step-body">
        <div class="step-head">
          <Badge :variant="typeVariant(step.type)">{{ step.type }}</Badge>
          <span class="step-name">
            <template v-if="step.type === 'helper'">{{ getHelperLabel(step.name) }}</template>
            <template v-else>{{ step.name }}</template>
          </span>
          <span v-if="isBlockStart && step.params" class="block-hint">
            <template v-if="step.name === 'LOOP' && step.params.iterations">({{ step.params.iterations }}{{ $t('experiment.times') }})</template>
            <template v-else-if="step.name === 'CONDITION' && step.params.condition">({{ step.params.condition }})</template>
            <template v-else-if="step.name === 'GROUP' && step.params.name">({{ step.params.name }})</template>
          </span>
        </div>
        <div v-if="step.description && !['LOOP','GROUP','CONDITION','END'].includes(step.name)" class="step-desc">
          {{ step.description }}
        </div>
      </div>
      <div class="step-actions">
        <button class="step-btn" :title="$t('experiment.moveUp')" :disabled="index === 0" @click="emit('moveUp')"><ArrowUp :size="12" /></button>
        <button class="step-btn" :title="$t('experiment.moveDown')" :disabled="index === total - 1" @click="emit('moveDown')"><ArrowDown :size="12" /></button>
        <button class="step-btn" :title="$t('experiment.edit')" @click="emit('edit')"><Pencil :size="12" /></button>
        <button class="step-btn step-btn-danger" :title="$t('common.delete')" @click="emit('remove')"><Trash2 :size="12" /></button>
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
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  transition: border-color var(--transition-fast);
  overflow: hidden;
}

.step-card:hover {
  border-color: var(--color-primary-soft);
}

.step-card.editing {
  border-color: var(--color-primary);
}

/* Block-type styling */
.step-card.is-block-start {
  border-left: 3px solid #f59e0b;
  background: #fffbeb;
}

.step-card.is-block-end {
  border-left: 3px solid var(--color-border);
  background: var(--color-bg-soft);
  opacity: 0.85;
}

.step-main {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: var(--space-sm) var(--space-md);
}

.step-body {
  flex: 1;
  min-width: 0;
}

.step-head {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
}

.step-name {
  font-weight: 600;
  font-size: 13px;
}

.block-hint {
  font-size: 11px;
  color: var(--color-text-tertiary);
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
}

.step-desc {
  font-size: 12px;
  color: var(--color-text-secondary);
  margin-top: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.step-actions {
  display: flex;
  gap: 2px;
  flex-shrink: 0;
}

.step-btn {
  width: 26px;
  height: 26px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: none;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--color-text-tertiary);
  cursor: pointer;
  transition: background var(--transition-fast), color var(--transition-fast);
}

.step-btn:hover:not(:disabled) {
  background: var(--color-bg-soft);
  color: var(--color-text);
}

.step-btn:disabled {
  opacity: 0.3;
  cursor: default;
}

.step-btn-danger:hover:not(:disabled) {
  color: var(--color-error);
}
</style>
