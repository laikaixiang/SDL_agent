<script setup lang="ts">
/**
 * CsvPreviewModal.vue — 全屏 CSV 预览 modal
 *
 * 显示前 20 行 + 列类型 + 行/列统计 + 文件大小
 * 用于"我想仔细看看这个 CSV"场景
 */
import { ref, watch } from 'vue'
import { X, FileText, Loader2 } from 'lucide-vue-next'
import { previewCsv } from '@/api/analysis'
import type { PreviewData, ColumnType } from '@/types/analysis'

const props = defineProps<{
  open: boolean
  path: string
}>()

const emit = defineEmits<{
  (e: 'update:open', val: boolean): void
}>()

const loading = ref(false)
const preview = ref<PreviewData | null>(null)
const error = ref('')

watch(() => [props.open, props.path], async ([isOpen, path]) => {
  if (isOpen && path) {
    await load(String(path))
  }
})

async function load(path: string) {
  loading.value = true
  error.value = ''
  preview.value = null
  try {
    const resp = await previewCsv(path, 20)
    if (resp.success && resp.data) {
      preview.value = resp.data
    } else {
      error.value = resp.message || '预览失败'
    }
  } catch (err) {
    error.value = (err as Error).message
  } finally {
    loading.value = false
  }
}

function close() {
  emit('update:open', false)
}

function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / 1024 / 1024).toFixed(2)} MB`
}

function typeVariant(t: ColumnType): string {
  switch (t) {
    case 'int':
    case 'float': return 'success'
    case 'bool': return 'warning'
    case 'date': return 'default'
    default: return 'default'
  }
}
</script>

<template>
  <div v-if="open" class="cpm-overlay" @click.self="close">
    <div class="cpm-modal">
      <div class="cpm-header">
        <div class="cpm-title">
          <FileText :size="18" />
          <h3>{{ path.split(/[/\\]/).pop() }}</h3>
        </div>
        <button class="cpm-close" @click="close"><X :size="16" /></button>
      </div>

      <div v-if="loading" class="cpm-loading">
        <Loader2 :size="20" class="spin" />
        <span>加载中...</span>
      </div>
      <div v-else-if="error" class="cpm-error">{{ error }}</div>
      <div v-else-if="preview" class="cpm-body">
        <div class="cpm-stats">
          <div class="stat-cell">
            <span class="stat-label">列数</span>
            <span class="stat-value">{{ preview.columns.length }}</span>
          </div>
          <div class="stat-cell">
            <span class="stat-label">总行数 (估算)</span>
            <span class="stat-value">{{ preview.total_rows }}</span>
          </div>
          <div class="stat-cell">
            <span class="stat-label">文件大小</span>
            <span class="stat-value">{{ formatBytes(preview.file_size) }}</span>
          </div>
          <div class="stat-cell">
            <span class="stat-label">路径</span>
            <span class="stat-value stat-path">{{ preview.path }}</span>
          </div>
        </div>

        <div class="cpm-table-wrap">
          <table class="cpm-table">
            <thead>
              <tr>
                <th v-for="col in preview.columns" :key="col.name">
                  <div class="cpm-col-head">
                    <span class="cpm-col-name">{{ col.name }}</span>
                    <span :class="['cpm-col-type', `type-${col.type}`]">{{ col.type }}</span>
                  </div>
                </th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="i in preview.row_count" :key="i">
                <td v-for="col in preview.columns" :key="col.name">
                  {{ col.sample[i - 1] || '—' }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.cpm-overlay {
  position: fixed; inset: 0; z-index: 11000;
  background: rgba(0,0,0,0.5);
  display: flex; align-items: center; justify-content: center;
}
.cpm-modal {
  background: var(--color-surface); border-radius: var(--radius-lg);
  width: 90vw; max-width: 1100px; max-height: 85vh;
  display: flex; flex-direction: column;
  box-shadow: var(--shadow-lg);
}
.cpm-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: var(--space-lg) var(--space-xl);
  border-bottom: 1px solid var(--color-border);
}
.cpm-title { display: flex; align-items: center; gap: var(--space-sm); color: var(--color-primary); }
.cpm-title h3 { font-size: 15px; color: var(--color-text); margin: 0; }
.cpm-close {
  width: 28px; height: 28px; display: flex; align-items: center; justify-content: center;
  border: none; border-radius: var(--radius-sm); background: transparent;
  color: var(--color-text-tertiary); cursor: pointer;
}
.cpm-close:hover { background: var(--color-bg-soft); }
.cpm-loading, .cpm-error { padding: var(--space-2xl); text-align: center; color: var(--color-text-tertiary); }
.cpm-error { color: var(--color-error); }
.spin { animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

.cpm-body { display: flex; flex-direction: column; flex: 1; min-height: 0; }
.cpm-stats {
  display: flex; gap: var(--space-lg);
  padding: var(--space-md) var(--space-xl);
  background: var(--color-bg-soft);
  border-bottom: 1px solid var(--color-border);
  flex-wrap: wrap;
}
.stat-cell { display: flex; flex-direction: column; gap: 2px; }
.stat-label { font-size: 11px; color: var(--color-text-tertiary); }
.stat-value { font-size: 14px; font-weight: 600; color: var(--color-text); }
.stat-path { font-size: 11px; font-family: monospace; max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.cpm-table-wrap { flex: 1; overflow: auto; padding: var(--space-md) var(--space-xl); }
.cpm-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.cpm-table th, .cpm-table td {
  text-align: left; padding: 6px 10px;
  border: 1px solid var(--color-border);
  max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.cpm-table th { background: var(--color-bg-soft); position: sticky; top: 0; z-index: 1; }
.cpm-table td { color: var(--color-text-secondary); font-family: monospace; }

.cpm-col-head { display: flex; flex-direction: column; gap: 2px; }
.cpm-col-name { font-weight: 600; color: var(--color-text); font-size: 12px; }
.cpm-col-type {
  font-size: 10px; padding: 1px 6px; border-radius: var(--radius-full);
  display: inline-block; font-family: monospace; width: fit-content;
}
.type-int, .type-float { background: rgba(59,130,246,0.15); color: var(--color-primary); }
.type-bool { background: rgba(245,158,11,0.15); color: var(--color-warning); }
.type-date { background: rgba(16,185,129,0.15); color: var(--color-success); }
.type-str { background: var(--color-bg-mute); color: var(--color-text-tertiary); }
</style>
