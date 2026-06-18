<script setup lang="ts">
/**
 * ExtractionResultTable.vue
 * Step 6: 渲染抽取结果 + 跳转原文链接
 *
 * 列:
 *  - 主键 (fields[0])
 *  - grounding ("原文原句" hover 显示完整)
 *  - 字段值 (fields[1..n])
 *  - 🔗 跳转 按钮
 *  - 置信度 (low_confidence 标灰, evidence_score 显示)
 *  - 来源 ({_source_doc} p.{_source_page})
 *
 * 低置信度行加 class="row-low-confidence" (CSS 灰底)
 */
import { computed } from 'vue'
import { ExternalLink, AlertTriangle } from 'lucide-vue-next'
import { useSearchStore } from '@/stores/search'

interface ExtractionRow {
  [key: string]: any
}

const props = defineProps<{
  rows: ExtractionRow[]
  fields: string[]
}>()

const search = useSearchStore()

const primaryKey = computed(() => props.fields[0] || '')
const groundingKey = computed(() => (props.fields.includes('原文原句') ? '原文原句' : ''))
const valueFields = computed(() =>
  props.fields.filter(f => f !== primaryKey.value && f !== groundingKey.value)
)

function jumpToSource(row: ExtractionRow) {
  const doc = row._source_doc as string | undefined
  const page = row._source_page as number | undefined
  if (!doc || !page) return
  const offset = row._evidence_offset ?? null
  const length = row._evidence_length ?? null
  search.jumpToSource(doc, page, offset, length)
}

function isLowConfidence(row: ExtractionRow): boolean {
  return row._low_confidence === true || (row._evidence_score ?? 1) < 0.7
}

function confidenceLabel(row: ExtractionRow): string {
  const score = row._evidence_score
  if (score == null) return '—'
  return `${Math.round(Number(score) * 100)}%`
}

function truncate(text: any, max: number = 80): string {
  if (text == null) return ''
  const s = String(text)
  return s.length > max ? s.slice(0, max) + '…' : s
}
</script>

<template>
  <div class="extraction-result-table">
    <table>
      <thead>
        <tr>
          <th class="col-primary">{{ primaryKey }}</th>
          <th v-if="groundingKey" class="col-grounding" :title="groundingKey">
            原文原句
          </th>
          <th v-for="f in valueFields" :key="f" :title="f">
            {{ f }}
          </th>
          <th class="col-confidence">置信度</th>
          <th class="col-source">来源</th>
          <th class="col-action"></th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="(row, idx) in rows"
          :key="idx"
          :class="{ 'row-low-confidence': isLowConfidence(row) }"
        >
          <td class="col-primary">
            <strong>{{ row[primaryKey] || '—' }}</strong>
            <span
              v-if="row._occurrence_count && row._occurrence_count > 1"
              class="occurrence-badge"
              :title="`出现 ${row._occurrence_count} 次`"
            >
              ×{{ row._occurrence_count }}
            </span>
            <span
              v-if="row._review_flag === 'duplicate'"
              class="duplicate-badge"
              title="LLM 审查标记为重复"
            >
              <AlertTriangle :size="12" /> dup
            </span>
            <span
              v-if="row._review_flag === 'low_value'"
              class="lowvalue-badge"
              title="LLM 审查标记为低置信度"
            >
              <AlertTriangle :size="12" /> low
            </span>
          </td>
          <td
            v-if="groundingKey"
            class="col-grounding"
            :title="row[groundingKey] || ''"
          >
            {{ truncate(row[groundingKey], 60) }}
          </td>
          <td v-for="f in valueFields" :key="f" :title="row[f] || ''">
            {{ truncate(row[f], 50) }}
          </td>
          <td class="col-confidence">
            <span :class="['confidence-pill', { 'low': isLowConfidence(row) }]">
              {{ confidenceLabel(row) }}
            </span>
          </td>
          <td class="col-source">
            {{ row._source_doc || '—' }}
            <span v-if="row._source_page">p.{{ row._source_page }}</span>
          </td>
          <td class="col-action">
            <button
              v-if="row._source_doc && row._source_page"
              class="jump-btn"
              title="跳转到原文位置"
              @click="jumpToSource(row)"
            >
              <ExternalLink :size="14" />
            </button>
          </td>
        </tr>
        <tr v-if="rows.length === 0">
          <td :colspan="valueFields.length + 4" class="empty-row">
            暂无提取结果
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.extraction-result-table {
  width: 100%;
  overflow-x: auto;
  font-size: 13px;
  border: 1px solid var(--border-color, #444);
  border-radius: 6px;
  background: var(--bg-secondary, #1e1e1e);
}

table {
  width: 100%;
  border-collapse: collapse;
}

th, td {
  padding: 8px 10px;
  text-align: left;
  border-bottom: 1px solid var(--border-color, #333);
  vertical-align: top;
}

th {
  background: var(--bg-tertiary, #2a2a2a);
  color: var(--text-secondary, #aaa);
  font-weight: 600;
  font-size: 12px;
  white-space: nowrap;
  position: sticky;
  top: 0;
  z-index: 1;
}

tbody tr:hover {
  background: var(--bg-hover, rgba(255, 255, 255, 0.03));
}

.row-low-confidence {
  background: var(--bg-warning, rgba(255, 170, 0, 0.06));
  color: var(--text-muted, #999);
}

.row-low-confidence:hover {
  background: var(--bg-warning-hover, rgba(255, 170, 0, 0.1));
}

.col-primary { min-width: 140px; }
.col-grounding { min-width: 180px; max-width: 280px; }
.col-confidence { width: 80px; text-align: center; }
.col-source { font-size: 11px; color: var(--text-muted, #888); min-width: 140px; }
.col-action { width: 40px; text-align: center; }

.confidence-pill {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 10px;
  background: var(--bg-pill, #2a4a2a);
  color: #8f8;
  font-size: 11px;
  font-weight: 600;
}

.confidence-pill.low {
  background: var(--bg-pill-low, #4a3a1a);
  color: #fa3;
}

.occurrence-badge {
  display: inline-block;
  margin-left: 6px;
  padding: 1px 6px;
  border-radius: 8px;
  background: var(--bg-badge, #3a3a5a);
  color: #aaf;
  font-size: 10px;
  font-weight: 600;
}

.duplicate-badge, .lowvalue-badge {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  margin-left: 6px;
  padding: 1px 6px;
  border-radius: 8px;
  font-size: 10px;
  font-weight: 600;
}

.duplicate-badge {
  background: #5a2a2a;
  color: #f88;
}

.lowvalue-badge {
  background: #5a4a1a;
  color: #fd8;
}

.jump-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: 1px solid var(--border-color, #555);
  border-radius: 4px;
  background: var(--bg-button, #3a3a3a);
  color: var(--color-primary, #fa0);
  cursor: pointer;
  transition: all 0.15s;
}

.jump-btn:hover {
  background: var(--bg-button-hover, #4a4a4a);
  border-color: var(--color-primary, #fa0);
}

.empty-row {
  text-align: center;
  padding: 24px;
  color: var(--text-muted, #888);
  font-style: italic;
}
</style>
