<script setup lang="ts">
/**
 * CsvPreviewInline.vue — 折叠式 CSV 预览组件
 *
 * 用在 FileSelectorModal 的每行末尾, 用户点击 👁 按钮展开, 看到列名+类型+前 3 个非空值
 * 不全屏, 不打开新 modal, 适合快速"扫一眼"
 */
import { ref } from 'vue'
import { ChevronDown, ChevronRight, Eye, Loader2 } from 'lucide-vue-next'
import { previewCsv } from '@/api/analysis'
import type { PreviewData, ColumnType } from '@/types/analysis'

const props = defineProps<{
  path: string
}>()

const expanded = ref(false)
const loading = ref(false)
const preview = ref<PreviewData | null>(null)
const error = ref('')

async function toggle() {
  if (!expanded.value && !preview.value) {
    await loadPreview()
  }
  expanded.value = !expanded.value
}

async function loadPreview() {
  loading.value = true
  error.value = ''
  try {
    const resp = await previewCsv(props.path, 5)
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

function typeColor(t: ColumnType): string {
  switch (t) {
    case 'int':
    case 'float':
      return 'var(--color-primary)'
    case 'bool':
      return 'var(--color-warning)'
    case 'date':
      return 'var(--color-success)'
    default:
      return 'var(--color-text-tertiary)'
  }
}
</script>

<template>
  <div class="csv-inline">
    <button class="inline-toggle" @click="toggle">
      <component :is="expanded ? ChevronDown : ChevronRight" :size="12" />
      <Eye :size="12" />
      <span>{{ expanded ? '收起预览' : '预览' }}</span>
    </button>
    <div v-if="expanded" class="inline-body">
      <div v-if="loading" class="inline-loading">
        <Loader2 :size="12" class="spin" /> 加载预览...
      </div>
      <div v-else-if="error" class="inline-error">{{ error }}</div>
      <div v-else-if="preview" class="inline-content">
        <div class="inline-stats">
          <span>{{ preview.columns.length }} 列</span>
          <span>·</span>
          <span>{{ preview.total_rows }} 行 (估算)</span>
        </div>
        <table class="inline-table">
          <thead>
            <tr>
              <th v-for="col in preview.columns" :key="col.name">
                <div class="col-head">
                  <span class="col-name">{{ col.name }}</span>
                  <span class="col-type" :style="{ color: typeColor(col.type) }">{{ col.type }}</span>
                </div>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="i in 3" :key="i">
              <td v-for="col in preview.columns" :key="col.name">
                <span class="cell-val">{{ col.sample[i - 1] || '—' }}</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<style scoped>
.csv-inline {
  border-top: 1px solid var(--color-border-soft, var(--color-border));
  margin-top: 4px;
  padding-top: 4px;
}
.inline-toggle {
  display: flex; align-items: center; gap: 4px;
  padding: 4px 8px; border: none; background: transparent;
  color: var(--color-text-tertiary); font-size: 11px; cursor: pointer;
  border-radius: var(--radius-sm);
}
.inline-toggle:hover { background: var(--color-bg-soft); color: var(--color-primary); }

.inline-body { padding: 6px 8px 8px; }
.inline-loading, .inline-error { font-size: 12px; color: var(--color-text-tertiary); padding: 8px; }
.inline-error { color: var(--color-error); }
.spin { animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

.inline-stats { font-size: 11px; color: var(--color-text-tertiary); margin-bottom: 6px; display: flex; gap: 4px; }
.inline-table { width: 100%; border-collapse: collapse; font-size: 11px; }
.inline-table th, .inline-table td {
  text-align: left; padding: 4px 6px;
  border-bottom: 1px solid var(--color-border-soft, var(--color-border));
  max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.col-head { display: flex; flex-direction: column; gap: 2px; }
.col-name { font-weight: 600; color: var(--color-text); }
.col-type { font-size: 10px; font-family: monospace; }
.cell-val { color: var(--color-text-secondary); }
</style>
