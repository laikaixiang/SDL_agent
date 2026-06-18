<script setup lang="ts">
/**
 * ResultRenderer.vue — 算法结果分发器
 *
 * 根据 algorithm.result_schema.type 派发到对应的子组件:
 *  - "table"  → ResultTable (单 section) / 多 sections
 *  - "kv"     → ResultKvList
 *  - "matrix" → ResultMatrix
 *  - "chart"  → ResultChart (Phase 2 单独 commit)
 *  - 空 / 未声明 → plain text fallback (JSON.stringify)
 */
import { computed } from 'vue'
import ResultTable from './ResultTable.vue'
import ResultKvList from './ResultKvList.vue'
import ResultMatrix from './ResultMatrix.vue'
import type { ResultSchema } from '@/types/analysis'

interface Props {
  /** 算法完整 result (含 success/result/message) */
  algoResult: { success?: boolean; result?: unknown; message?: string }
  /** result_schema 字段, 缺省或空 = fallback */
  schema?: ResultSchema
}

const props = defineProps<Props>()

// 局部非空版本, 供模板使用
const schema = computed(() => props.schema || ({} as ResultSchema))
const hasSchema = computed(() => {
  return props.schema && Object.keys(props.schema).length > 0
})

/**
 * 沿 dotted path 取值
 * 例: resolvePath(obj, "result.statistics") → obj["result"]["statistics"]
 */
function resolvePath(obj: unknown, path: string): unknown {
  if (!obj || !path) return undefined
  const parts = path.split('.')
  let cur: unknown = obj
  for (const p of parts) {
    if (cur === null || cur === undefined) return undefined
    if (typeof cur !== 'object') return undefined
    cur = (cur as Record<string, unknown>)[p]
  }
  return cur
}

const isError = computed(() => {
  return props.algoResult?.success === false
})

const errorMessage = computed(() => {
  return props.algoResult?.message || '执行失败'
})

/** 将 result 中的 dict 转 rows 数组 (key 作为 label) */
function dictToRows(data: Record<string, unknown>): Record<string, unknown>[] {
  return Object.entries(data).map(([k, v]) => ({ key: k, value: v }))
}
</script>

<template>
  <!-- 错误态 -->
  <div v-if="isError" class="result-error">
    <strong>执行失败:</strong> {{ errorMessage }}
  </div>

  <!-- 无 schema 走 fallback -->
  <div v-else-if="!hasSchema" class="result-fallback">
    <pre>{{ JSON.stringify(algoResult, null, 2) }}</pre>
  </div>

  <!-- 有 schema: 按 type 派发 -->
  <div v-else class="result-rendered">
    <!-- kv 类型 -->
    <template v-if="schema.type === 'kv'">
      <ResultKvList
        v-if="schema.items && schema.items.length"
        :items="schema.items"
        :data="(algoResult.result as Record<string, unknown>) || {}"
      />
      <pre v-else class="result-fallback">{{ JSON.stringify(algoResult.result, null, 2) }}</pre>
    </template>

    <!-- matrix 类型 -->
    <template v-else-if="schema.type === 'matrix'">
      <ResultMatrix
        :data="(algoResult.result as Record<string, Record<string, number>>) || {}"
      />
    </template>

    <!-- table 类型: 多 sections -->
    <template v-else-if="schema.type === 'table'">
      <div v-for="(section, i) in schema.sections" :key="i" class="result-section">
        <!-- matrix sub-section -->
        <ResultMatrix
          v-if="section.type === 'matrix'"
          :title="section.title"
          :data="(resolvePath(algoResult.result, section.rows_from) as Record<string, Record<string, number>>) || {}"
          :value-format="section.value_format"
        />
        <!-- table sub-section -->
        <ResultTable
          v-else-if="section.columns && section.columns.length"
          :title="section.title"
          :columns="section.columns"
          :rows="dictToRows((resolvePath(algoResult.result, section.rows_from) as Record<string, unknown>) || {})"
          row-key="key"
        />
        <!-- 无列定义: 走 fallback -->
        <pre v-else class="result-fallback">{{ JSON.stringify(resolvePath(algoResult.result, section.rows_from), null, 2) }}</pre>
      </div>
    </template>

    <!-- chart 类型 (占位, 后续 commit 实现 ResultChart) -->
    <template v-else-if="schema.type === 'chart'">
      <pre class="result-fallback">{{ JSON.stringify(algoResult.result, null, 2) }}</pre>
    </template>

    <!-- list 类型 -->
    <template v-else-if="schema.type === 'list'">
      <pre class="result-fallback">{{ JSON.stringify(algoResult.result, null, 2) }}</pre>
    </template>

    <!-- 未知 type: fallback -->
    <pre v-else class="result-fallback">{{ JSON.stringify(algoResult.result, null, 2) }}</pre>
  </div>
</template>

<style scoped>
.result-error {
  padding: var(--space-md);
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid var(--color-error);
  border-radius: var(--radius-sm);
  color: var(--color-error);
  font-size: 13px;
}
.result-fallback {
  background: var(--color-bg-soft);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  padding: var(--space-md);
  font-size: 12px;
  overflow-x: auto;
  margin: 0;
}
.result-rendered { display: flex; flex-direction: column; gap: var(--space-sm); }
.result-section { display: flex; flex-direction: column; }
</style>
