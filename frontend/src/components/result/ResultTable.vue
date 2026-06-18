<script setup lang="ts">
/**
 * ResultTable.vue — 通用表格组件
 *
 * 接收列定义 (columns) + 行数据 (rows), 支持:
 *  - 数值格式化: decimal:N / integer / percent:N / scientific / json
 *  - 空值占位 (—)
 *  - 横向滚动
 *  - 单行 hover 高亮
 */
import { computed } from 'vue'
import type { TableColumn } from '@/types/analysis'

interface Props {
  columns: TableColumn[]
  /** 每行是一个 record, key 对应 TableColumn.key */
  rows: Record<string, unknown>[]
  /** 标题, 可选 */
  title?: string
  /** 行 key 字段 (用于 v-for), 默认用 index */
  rowKey?: string
}

const props = defineProps<Props>()

function formatValue(value: unknown, format?: string): string {
  if (value === null || value === undefined || value === '') return '—'
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
  if (format.startsWith('percent:')) {
    const digits = parseInt(format.split(':')[1] || '1', 10)
    const n = Number(value)
    return Number.isFinite(n) ? `${(n * 100).toFixed(digits)}%` : String(value)
  }
  if (format === 'scientific') {
    const n = Number(value)
    return Number.isFinite(n) ? n.toExponential(3) : String(value)
  }
  if (format === 'json') {
    try {
      return JSON.stringify(value)
    } catch {
      return String(value)
    }
  }
  return String(value)
}

const rowKeys = computed(() => props.rows.map((_, i) => (props.rowKey ? String(props.rows[i][props.rowKey]) : String(i))))
</script>

<template>
  <div class="result-table-wrap">
    <h4 v-if="title" class="result-table-title">{{ title }}</h4>
    <div class="result-table-scroll">
      <table class="result-table">
        <thead>
          <tr>
            <th v-for="col in columns" :key="col.key">{{ col.label }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, i) in rows" :key="rowKeys[i]">
            <td v-for="col in columns" :key="col.key">
              {{ formatValue(row[col.key], col.format) }}
            </td>
          </tr>
        </tbody>
      </table>
      <div v-if="!rows.length" class="result-table-empty">无数据</div>
    </div>
  </div>
</template>

<style scoped>
.result-table-wrap { display: flex; flex-direction: column; gap: 6px; margin-bottom: var(--space-md); }
.result-table-title { font-size: 12px; font-weight: 600; color: var(--color-text-secondary); margin: 0; }
.result-table-scroll { overflow-x: auto; border: 1px solid var(--color-border); border-radius: var(--radius-sm); }
.result-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.result-table th, .result-table td {
  text-align: left; padding: 6px 10px;
  border-bottom: 1px solid var(--color-border);
  white-space: nowrap;
}
.result-table th { background: var(--color-bg-soft); color: var(--color-text-secondary); font-weight: 600; }
.result-table tbody tr:hover { background: var(--color-bg-soft); }
.result-table td { color: var(--color-text); font-family: monospace; }
.result-table-empty { padding: var(--space-md); text-align: center; color: var(--color-text-tertiary); font-size: 12px; }
</style>
