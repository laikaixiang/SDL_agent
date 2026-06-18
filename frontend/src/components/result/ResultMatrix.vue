<script setup lang="ts">
/**
 * ResultMatrix.vue — 相关性矩阵转置显示
 *
 * 接收 {col: {row: value}} 嵌套 dict, 转置为表格
 * 例: {"A": {"A":1.0, "B":0.5}, "B": {"A":0.5, "B":1.0}}
 *   → 行: A B / 列: A B / value: 1.0 0.5 / 0.5 1.0
 */
import { computed } from 'vue'

interface Props {
  data: Record<string, Record<string, number>>
  /** 值格式 */
  valueFormat?: string
  title?: string
}

const props = defineProps<Props>()

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return '—'
  const n = Number(v)
  if (!Number.isFinite(n)) return String(v)
  if (props.valueFormat?.startsWith('decimal:')) {
    const digits = parseInt(props.valueFormat.split(':')[1] || '3', 10)
    return n.toFixed(digits)
  }
  return n.toFixed(3)
}

const rows = computed(() => {
  const keys = Object.keys(props.data || {})
  if (!keys.length) return []
  return keys.map(rowKey => {
    const row = props.data[rowKey] || {}
    return {
      label: rowKey,
      cells: keys.map(colKey => row[colKey]),
    }
  })
})

const colKeys = computed(() => Object.keys(props.data || {}))
</script>

<template>
  <div class="matrix-wrap">
    <h4 v-if="title" class="matrix-title">{{ title }}</h4>
    <div class="matrix-scroll">
      <table class="matrix-table">
        <thead>
          <tr>
            <th></th>
            <th v-for="ck in colKeys" :key="ck">{{ ck }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in rows" :key="row.label">
            <th class="row-head">{{ row.label }}</th>
            <td
              v-for="(cell, i) in row.cells"
              :key="i"
              :style="{
                background: cellColor(cell),
              }"
            >
              {{ formatValue(cell) }}
            </td>
          </tr>
        </tbody>
      </table>
      <div v-if="!rows.length" class="matrix-empty">无矩阵数据</div>
    </div>
  </div>
</template>

<script lang="ts">
function cellColor(v: unknown): string {
  const n = Number(v)
  if (!Number.isFinite(n)) return 'transparent'
  // 相关系数 [-1, 1] → 红色负相关 / 绿色正相关
  const abs = Math.abs(n)
  const alpha = abs * 0.4
  if (n > 0) return `rgba(16,185,129,${alpha})`
  if (n < 0) return `rgba(239,68,68,${alpha})`
  return 'transparent'
}
export default { name: 'ResultMatrix' }
</script>

<style scoped>
.matrix-wrap { display: flex; flex-direction: column; gap: 6px; margin-bottom: var(--space-md); }
.matrix-title { font-size: 12px; font-weight: 600; color: var(--color-text-secondary); margin: 0; }
.matrix-scroll { overflow-x: auto; border: 1px solid var(--color-border); border-radius: var(--radius-sm); }
.matrix-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.matrix-table th, .matrix-table td {
  padding: 6px 10px; text-align: center;
  border: 1px solid var(--color-border);
  font-family: monospace;
}
.matrix-table thead th { background: var(--color-bg-soft); font-weight: 600; }
.row-head { background: var(--color-bg-soft); font-weight: 600; text-align: left; }
.matrix-table td { transition: background 0.2s; }
.matrix-empty { padding: var(--space-md); text-align: center; color: var(--color-text-tertiary); font-size: 12px; }
</style>
