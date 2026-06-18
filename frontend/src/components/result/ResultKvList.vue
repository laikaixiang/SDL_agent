<script setup lang="ts">
/**
 * ResultKvList.vue — 键值对列表组件
 *
 * 用于 spectrum_analysis 这类"小数据, 大数字"结果展示
 * 每项: label(中文名) + value(大数字) + unit(可选)
 */
import { computed } from 'vue'
import type { KvItem } from '@/types/analysis'

interface Props {
  items: KvItem[]
  /** 数据源, 用 item.key 取值 */
  data: Record<string, unknown>
  /** 标题, 可选 */
  title?: string
}

const props = defineProps<Props>()

function formatValue(value: unknown, format?: string): string {
  if (value === null || value === undefined) return '—'
  if (!format) return String(value)
  if (format === 'integer') {
    const n = Number(value)
    return Number.isFinite(n) ? n.toFixed(0) : String(value)
  }
  if (format.startsWith('decimal:')) {
    const digits = parseInt(format.split(':')[1] || '3', 10)
    const n = Number(value)
    return Number.isFinite(n) ? n.toFixed(digits) : String(value)
  }
  if (format === 'scientific') {
    const n = Number(value)
    return Number.isFinite(n) ? n.toExponential(3) : String(value)
  }
  return String(value)
}

const resolved = computed(() =>
  props.items.map(item => ({
    ...item,
    raw: props.data[item.key],
    display: formatValue(props.data[item.key], item.format),
  }))
)
</script>

<template>
  <div class="kv-wrap">
    <h4 v-if="title" class="kv-title">{{ title }}</h4>
    <div class="kv-grid">
      <div v-for="item in resolved" :key="item.key" class="kv-item">
        <div class="kv-label">{{ item.label }}</div>
        <div class="kv-value-row">
          <span class="kv-value">{{ item.display }}</span>
          <span v-if="item.unit" class="kv-unit">{{ item.unit }}</span>
        </div>
      </div>
    </div>
    <div v-if="!resolved.length" class="kv-empty">无数据</div>
  </div>
</template>

<style scoped>
.kv-wrap { margin-bottom: var(--space-md); }
.kv-title { font-size: 12px; font-weight: 600; color: var(--color-text-secondary); margin: 0 0 var(--space-sm) 0; }
.kv-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: var(--space-sm);
}
.kv-item {
  background: var(--color-bg-soft);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  padding: var(--space-sm) var(--space-md);
  display: flex; flex-direction: column; gap: 4px;
}
.kv-label { font-size: 11px; color: var(--color-text-tertiary); }
.kv-value-row { display: flex; align-items: baseline; gap: 4px; }
.kv-value { font-size: 18px; font-weight: 600; color: var(--color-primary); font-family: monospace; }
.kv-unit { font-size: 12px; color: var(--color-text-tertiary); }
.kv-empty { padding: var(--space-md); text-align: center; color: var(--color-text-tertiary); font-size: 12px; }
</style>
