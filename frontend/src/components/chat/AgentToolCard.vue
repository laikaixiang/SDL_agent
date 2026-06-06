<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  index: number
  name: string
  args?: Record<string, unknown>
  result?: string
  status: 'running' | 'done' | 'error'
}>()

const iconMap: Record<string, string> = {
  ask_user: '❓',
  spawn_agent: '🤖',
  search_literature: '🔍',
  control_hardware: '⚙️',
  design_experiment: '🧪',
  analyze_data: '📊',
  generate_algorithm: '💻',
}

const icon = computed(() => iconMap[props.name] || '🔧')

const statusLabel = computed(() => {
  switch (props.status) {
    case 'running': return '⏳ 执行中...'
    case 'done': return '✓'
    case 'error': return '✗'
  }
})

const argsSummary = computed(() => {
  if (!props.args) return ''
  const entries = Object.entries(props.args).slice(0, 2)
  return entries.map(([k, v]) => `${k}=${v}`).join(', ')
})

const displayResult = computed(() => {
  if (!props.result) return ''
  return props.result.length > 300 ? props.result.slice(0, 300) + '...' : props.result
})
</script>

<template>
  <div class="tool-card" :class="`tool-card--${status}`">
    <div class="tool-card__header">
      <span class="tool-card__icon">{{ icon }}</span>
      <span class="tool-card__name">{{ name }}</span>
      <span class="tool-card__status">{{ statusLabel }}</span>
    </div>
    <div v-if="argsSummary" class="tool-card__args">{{ argsSummary }}</div>
    <div
      v-if="displayResult"
      class="tool-card__result"
      :class="{ 'tool-card__result--error': status === 'error' }"
    >{{ displayResult }}</div>
  </div>
</template>

<style scoped>
.tool-card {
  margin: var(--space-sm) 0;
  padding: var(--space-md);
  border-radius: var(--radius-md);
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  font-size: 13px;
  line-height: 1.5;
}

.tool-card--running {
  border-color: var(--color-primary);
}

.tool-card--done {
  border-color: var(--color-success);
}

.tool-card--error {
  border-color: var(--color-error);
  background: rgba(239, 68, 68, 0.04);
}

.tool-card__header {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
}

.tool-card__icon {
  font-size: 16px;
  flex-shrink: 0;
}

.tool-card__name {
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-weight: 600;
  color: var(--color-text);
  flex: 1;
}

.tool-card__status {
  font-size: 13px;
  color: var(--color-text-secondary);
  flex-shrink: 0;
}

.tool-card__args {
  margin-top: var(--space-xs);
  font-size: 12px;
  color: var(--color-text-tertiary);
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  padding: var(--space-xs) var(--space-sm);
  background: var(--color-bg-soft);
  border-radius: var(--radius-sm);
}

.tool-card__result {
  margin-top: var(--space-xs);
  padding: var(--space-xs) var(--space-sm);
  background: var(--color-bg-soft);
  border-radius: var(--radius-sm);
  font-size: 12px;
  color: var(--color-text-secondary);
  white-space: pre-wrap;
  word-break: break-word;
}

.tool-card__result--error {
  color: var(--color-error);
}
</style>
