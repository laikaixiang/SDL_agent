<script setup lang="ts">
import { computed } from 'vue'

export interface TeamAgent {
  id: string
  template: string
  task: string
  status: 'spawning' | 'running' | 'done' | 'error'
  summary?: string
}

const props = defineProps<{
  mode: 'parallel' | 'pipeline' | 'single'
  agents: TeamAgent[]
  collapsed?: boolean
}>()

const emit = defineEmits<{
  (e: 'toggle'): void
}>()

const templateIcons: Record<string, string> = {
  literature_searcher: '🔍',
  literature_extractor: '📄',
  summarizer: '📋',
  experiment_designer: '🧪',
  data_analyst: '📊',
  extraction_pipeline: '⚙️',
}

const modeLabel = computed(() => {
  switch (props.mode) {
    case 'parallel': return '并行执行'
    case 'pipeline': return '流水线'
    case 'single': return '子任务'
  }
})

const modeIcon = computed(() => {
  switch (props.mode) {
    case 'parallel': return '⇉'
    case 'pipeline': return '→'
    case 'single': return '↳'
  }
})

const doneCount = computed(() => props.agents.filter(a => a.status === 'done').length)
const errorCount = computed(() => props.agents.filter(a => a.status === 'error').length)
const totalCount = computed(() => props.agents.length)

const progressPercent = computed(() =>
  totalCount.value > 0 ? Math.round((doneCount.value / totalCount.value) * 100) : 0
)

function agentIcon(template: string) {
  return templateIcons[template] || '🤖'
}

function agentStatusIcon(status: string) {
  switch (status) {
    case 'spawning': return '🟡'
    case 'running': return '⏳'
    case 'done': return '✅'
    case 'error': return '❌'
  }
}

function shortTask(task: string) {
  return task.length > 60 ? task.slice(0, 60) + '...' : task
}

function templateName(t: string) {
  const names: Record<string, string> = {
    literature_searcher: '文献检索',
    literature_extractor: '文献提取',
    summarizer: '总结清洗',
    experiment_designer: '实验设计',
    data_analyst: '数据分析',
    extraction_pipeline: '提取流水线',
  }
  return names[t] || t
}
</script>

<template>
  <div class="team-card" :class="`team-card--${mode}`">
    <!-- Header -->
    <div class="team-card__header" @click="emit('toggle')">
      <span class="team-card__mode-icon">{{ modeIcon }}</span>
      <span class="team-card__mode-label">{{ modeLabel }}</span>
      <span class="team-card__progress">
        {{ doneCount }}/{{ totalCount }}
        <span v-if="errorCount > 0" class="team-card__error-count"> ({{ errorCount }} 错误)</span>
      </span>
      <span class="team-card__chevron">{{ collapsed ? '▸' : '▾' }}</span>
    </div>

    <!-- Progress bar -->
    <div class="team-card__bar-track">
      <div
        class="team-card__bar-fill"
        :class="{ 'team-card__bar-fill--done': progressPercent === 100 }"
        :style="{ width: progressPercent + '%' }"
      />
    </div>

    <!-- Agent list (collapsible) -->
    <div v-if="!collapsed" class="team-card__agents">
      <div
        v-for="agent in agents"
        :key="agent.id"
        class="team-card__agent"
        :class="`team-card__agent--${agent.status}`"
      >
        <span class="team-card__agent-icon">{{ agentIcon(agent.template) }}</span>
        <div class="team-card__agent-info">
          <span class="team-card__agent-template">{{ templateName(agent.template) }}</span>
          <span class="team-card__agent-task">{{ shortTask(agent.task) }}</span>
        </div>
        <span class="team-card__agent-status">{{ agentStatusIcon(agent.status) }}</span>
        <div v-if="agent.summary" class="team-card__agent-summary">{{ agent.summary }}</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.team-card {
  margin: 8px 0;
  border-radius: 10px;
  border: 1px solid var(--color-border, #e5e7eb);
  background: var(--color-surface, #fff);
  overflow: hidden;
  font-size: 13px;
  transition: border-color 0.2s;
}

.team-card--parallel {
  border-left: 3px solid #6366f1;
}

.team-card--pipeline {
  border-left: 3px solid #0ea5e9;
}

.team-card--single {
  border-left: 3px solid #10b981;
}

.team-card__header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  cursor: pointer;
  user-select: none;
  transition: background 0.15s;
}
.team-card__header:hover {
  background: var(--color-bg-soft, #f9fafb);
}

.team-card__mode-icon {
  font-size: 16px;
  font-weight: 700;
  color: var(--color-primary, #6366f1);
  width: 20px;
  text-align: center;
}

.team-card__mode-label {
  font-weight: 600;
  color: var(--color-text, #111827);
}

.team-card__progress {
  margin-left: auto;
  font-size: 12px;
  color: var(--color-text-secondary, #6b7280);
  font-variant-numeric: tabular-nums;
}

.team-card__error-count {
  color: var(--color-error, #ef4444);
}

.team-card__chevron {
  font-size: 11px;
  color: var(--color-text-tertiary, #9ca3af);
  width: 16px;
  text-align: center;
}

.team-card__bar-track {
  height: 3px;
  background: var(--color-bg-soft, #f3f4f6);
  margin: 0 14px;
  border-radius: 2px;
  overflow: hidden;
}

.team-card__bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #6366f1, #818cf8);
  border-radius: 2px;
  transition: width 0.4s ease;
}

.team-card__bar-fill--done {
  background: linear-gradient(90deg, #10b981, #34d399);
}

.team-card__agents {
  padding: 4px 14px 10px;
}

.team-card__agent {
  display: grid;
  grid-template-columns: 20px 1fr 24px;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-radius: 6px;
  margin-top: 4px;
  transition: background 0.15s;
}
.team-card__agent:hover {
  background: var(--color-bg-soft, #f9fafb);
}

.team-card__agent--running {
  background: rgba(99, 102, 241, 0.04);
}

.team-card__agent--error {
  background: rgba(239, 68, 68, 0.04);
}

.team-card__agent-icon {
  font-size: 14px;
  text-align: center;
}

.team-card__agent-info {
  display: flex;
  flex-direction: column;
  gap: 1px;
  min-width: 0;
}

.team-card__agent-template {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text, #111827);
}

.team-card__agent-task {
  font-size: 11px;
  color: var(--color-text-tertiary, #9ca3af);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.team-card__agent-status {
  font-size: 13px;
  text-align: center;
}

.team-card__agent-summary {
  grid-column: 2 / 4;
  font-size: 11px;
  color: var(--color-text-secondary, #6b7280);
  padding: 4px 0;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
