<script setup lang="ts">
/**
 * ParamForm.vue — 根据 params_schema 动态生成表单
 *
 * 支持的 type:
 *  - bool      → 复选框
 *  - int/float → 数字输入框
 *  - str       → 文本输入框; 若有 options → 下拉框
 *  - list      → JSON 文本框
 *  - columns   → 多选下拉框 (从传入的 columnList 取)
 */
import { computed } from 'vue'
import type { ParamSchema } from '@/types/analysis'

interface Props {
  schema: Record<string, ParamSchema>
  modelValue: Record<string, unknown>
  /** 用于 type=columns 的可选列名列表 */
  columnList?: string[]
}

const props = withDefaults(defineProps<Props>(), {
  columnList: () => [],
})

const emit = defineEmits<{
  (e: 'update:modelValue', v: Record<string, unknown>): void
}>()

// 完整 schema 含默认值, 用于显示"留空使用 default"
const entries = computed(() => {
  return Object.entries(props.schema).map(([key, def]) => ({
    key,
    label: def.description || key,
    type: def.type,
    required: def.required || false,
    default: def.default,
    options: def.options || [],
  }))
})

function getDisplay(key: string): string {
  const v = props.modelValue[key]
  if (v === null || v === undefined) {
    const def = props.schema[key]
    if (def && def.default !== undefined && def.default !== null) {
      return `默认: ${JSON.stringify(def.default)}`
    }
    return ''
  }
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

function setValue(key: string, raw: string | number | boolean | string[]) {
  const def = props.schema[key]
  if (!def) return
  let val: unknown = raw
  if (def.type === 'int') val = raw === '' ? null : parseInt(String(raw), 10)
  else if (def.type === 'float') val = raw === '' ? null : parseFloat(String(raw))
  else if (def.type === 'bool') val = Boolean(raw)
  else if (def.type === 'list') {
    try {
      val = raw === '' ? null : JSON.parse(String(raw))
    } catch {
      val = raw
    }
  }
  emit('update:modelValue', { ...props.modelValue, [key]: val })
}

function toggleColumn(key: string, col: string) {
  const cur = (props.modelValue[key] as string[] | undefined) || []
  const next = cur.includes(col) ? cur.filter(c => c !== col) : [...cur, col]
  setValue(key, next)
}

function isColumnSelected(key: string, col: string): boolean {
  const cur = (props.modelValue[key] as string[] | undefined) || []
  return cur.includes(col)
}
</script>

<template>
  <div v-if="entries.length" class="param-form">
    <div v-for="entry in entries" :key="entry.key" class="param-row">
      <label class="param-label">
        {{ entry.label }}
        <span v-if="entry.required" class="param-required">*</span>
        <span class="param-type">{{ entry.type }}</span>
      </label>

      <!-- bool: 复选框 -->
      <input
        v-if="entry.type === 'bool'"
        type="checkbox"
        :checked="Boolean(modelValue[entry.key] ?? entry.default)"
        @change="setValue(entry.key, ($event.target as HTMLInputElement).checked)"
      />

      <!-- int / float: 数字输入 -->
      <input
        v-else-if="entry.type === 'int' || entry.type === 'float'"
        type="number"
        :step="entry.type === 'float' ? '0.0001' : '1'"
        :value="(modelValue[entry.key] ?? entry.default ?? '') as string | number"
        :placeholder="getDisplay(entry.key)"
        @input="setValue(entry.key, ($event.target as HTMLInputElement).value)"
      />

      <!-- str + options: 下拉框 -->
      <select
        v-else-if="entry.type === 'str' && entry.options.length"
        :value="(modelValue[entry.key] ?? entry.default ?? '') as string"
        @change="setValue(entry.key, ($event.target as HTMLSelectElement).value)"
      >
        <option value="" disabled>{{ entry.default ? `默认: ${entry.default}` : '选择...' }}</option>
        <option v-for="opt in entry.options" :key="opt" :value="opt">{{ opt }}</option>
      </select>

      <!-- str (无 options): 文本输入 -->
      <input
        v-else-if="entry.type === 'str'"
        type="text"
        :value="(modelValue[entry.key] ?? entry.default ?? '') as string"
        :placeholder="getDisplay(entry.key)"
        @input="setValue(entry.key, ($event.target as HTMLInputElement).value)"
      />

      <!-- list: JSON 文本框 -->
      <input
        v-else-if="entry.type === 'list'"
        type="text"
        :value="getDisplay(entry.key)"
        :placeholder="getDisplay(entry.key)"
        @change="setValue(entry.key, ($event.target as HTMLInputElement).value)"
      />

      <!-- columns: 多选 chips -->
      <div v-else-if="entry.type === 'columns'" class="param-columns">
        <div v-if="!columnList.length" class="param-hint">{{ $t('analysis.noFileSelected') }}</div>
        <div v-else class="param-col-chips">
          <button
            v-for="col in columnList"
            :key="col"
            type="button"
            class="param-col-chip"
            :class="{ active: isColumnSelected(entry.key, col) }"
            @click="toggleColumn(entry.key, col)"
          >
            {{ col }}
          </button>
        </div>
      </div>

      <!-- 未知 type: 通用文本输入 -->
      <input
        v-else
        type="text"
        :value="getDisplay(entry.key)"
        :placeholder="getDisplay(entry.key)"
        @input="setValue(entry.key, ($event.target as HTMLInputElement).value)"
      />
    </div>
  </div>
  <div v-else class="param-empty">{{ $t('analysis.noParamsNeeded') }}</div>
</template>

<style scoped>
.param-form { display: flex; flex-direction: column; gap: 8px; }
.param-row { display: flex; align-items: center; gap: 8px; }
.param-label {
  flex-shrink: 0; min-width: 100px; max-width: 160px;
  font-size: 12px; color: var(--color-text-secondary);
  display: flex; align-items: center; gap: 4px;
}
.param-required { color: var(--color-error); }
.param-type { font-size: 10px; color: var(--color-text-tertiary); font-family: monospace; padding: 1px 4px; background: var(--color-bg-mute); border-radius: var(--radius-sm); }

.param-row input[type="text"],
.param-row input[type="number"],
.param-row select {
  flex: 1; min-width: 0;
  padding: 5px 8px; border: 1px solid var(--color-border);
  border-radius: var(--radius-sm); font-size: 12px;
  background: var(--color-surface);
  color: var(--color-text);
}
.param-row input:focus, .param-row select:focus { outline: none; border-color: var(--color-primary); }

.param-columns { flex: 1; }
.param-hint { font-size: 11px; color: var(--color-text-tertiary); }
.param-col-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.param-col-chip {
  padding: 3px 8px; font-size: 11px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-full);
  background: var(--color-surface);
  color: var(--color-text-secondary);
  cursor: pointer;
  transition: all var(--transition-fast);
}
.param-col-chip:hover { border-color: var(--color-primary); }
.param-col-chip.active { background: var(--color-primary); color: #fff; border-color: var(--color-primary); }

.param-empty { font-size: 12px; color: var(--color-text-tertiary); padding: var(--space-sm); }
</style>
