<script setup lang="ts">
import { FileText } from 'lucide-vue-next'
import Badge from '@/components/common/Badge.vue'

defineProps<{
  title: string
  subtitle?: string
  tags?: { label: string; type?: 'default' | 'success' | 'warning' | 'error' }[]
}>()
</script>

<template>
  <div class="result-card">
    <div class="card-header">
      <FileText :size="20" class="card-icon" />
      <div class="card-info">
        <div class="card-title">{{ title }}</div>
        <div class="card-sub" v-if="subtitle">{{ subtitle }}</div>
      </div>
    </div>
    <div class="card-footer" v-if="tags?.length">
      <Badge v-for="t in tags" :key="t.label" :variant="t.type || 'default'">{{ t.label }}</Badge>
    </div>
    <slot />
  </div>
</template>

<style scoped>
.result-card {
  background: var(--color-surface); border: 1px solid var(--color-border);
  border-radius: var(--radius-md); padding: var(--space-lg);
  transition: box-shadow var(--transition-fast);
}
.result-card:hover { box-shadow: var(--shadow-md); }
.card-header { display: flex; gap: var(--space-md); align-items: flex-start; }
.card-icon { color: var(--color-primary); margin-top: 2px; flex-shrink: 0; }
.card-info { flex: 1; min-width: 0; }
.card-title { font-size: 14px; font-weight: 600; color: var(--color-text); }
.card-sub { font-size: 13px; color: var(--color-text-secondary); margin-top: 2px; }
.card-footer { display: flex; gap: var(--space-xs); margin-top: var(--space-md); flex-wrap: wrap; }
</style>
