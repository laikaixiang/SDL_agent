<script setup lang="ts">
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useExperimentStore } from '@/stores/experiment'
import { Plus, Upload, Trash2, Download } from 'lucide-vue-next'

const { t } = useI18n()
const store = useExperimentStore()

const showAddForm = ref(false)
const newVarName = ref('')
const newVarType = ref<'int' | 'float' | 'str' | 'bool'>('int')

const varList = computed(() => {
  return Object.values(store.variables)
})

const selectedRefs = computed(() => {
  if (!store.selectedVariable) return []
  return store.getVariableReferences(store.selectedVariable)
})

function onAdd() {
  const name = newVarName.value.trim()
  if (!name) return
  if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name)) {
    store.addLog(t('experiment.invalidVarNameFormat'))
    return
  }
  store.addVariable(name, newVarType.value)
  newVarName.value = ''
  showAddForm.value = false
}

function onImportCSV() {
  const input = document.createElement('input')
  input.type = 'file'
  input.accept = '.csv'
  input.onchange = () => {
    const file = input.files?.[0]
    if (file) store.importCSVFile(file)
  }
  input.click()
}

function onExportCSV() {
  const lines: string[] = ['name,type,default_value,min,max,required']
  for (const v of varList.value) {
    const min = v.constraints?.min ?? ''
    const max = v.constraints?.max ?? ''
    const required = v.constraints?.required ? 'true' : ''
    lines.push(`${v.name},${v.type},${v.default_value},${min},${max},${required}`)
  }
  // Append batch_data if present
  if (store.batchData.length > 0) {
    const keys = Object.keys(store.batchData[0])
    lines.push('')
    lines.push(keys.join(','))
    for (const row of store.batchData) {
      lines.push(keys.map(k => row[k] ?? '').join(','))
    }
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'variables.csv'
  a.click()
  URL.revokeObjectURL(url)
}

function onDefaultValueInput(name: string, raw: string) {
  const v = store.variables[name]
  if (!v) return
  let parsed: number | string | boolean = raw
  if (v.type === 'int') {
    const n = parseInt(raw, 10)
    parsed = isNaN(n) ? 0 : n
  } else if (v.type === 'float') {
    const n = parseFloat(raw)
    parsed = isNaN(n) ? 0.0 : n
  } else if (v.type === 'bool') {
    parsed = raw === 'true' || raw === '1'
  }
  store.updateVariable(name, { default_value: parsed })
}

function onConstraintInput(name: string, raw: string) {
  const trimmed = raw.trim()
  if (!trimmed) {
    store.updateVariable(name, { constraints: undefined })
    return
  }
  if (trimmed === t('experiment.required')) {
    store.updateVariable(name, { constraints: { required: true } })
    return
  }
  // Parse "1000-6000" format
  const rangeMatch = trimmed.match(/^(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)$/)
  if (rangeMatch) {
    store.updateVariable(name, {
      constraints: {
        min: parseFloat(rangeMatch[1]),
        max: parseFloat(rangeMatch[2]),
      },
    })
  }
}

function getConstraintDisplay(v: typeof varList.value[0]): string {
  if (!v?.constraints) return ''
  const c = v.constraints
  if (c.required && c.min === undefined && c.max === undefined) return t('experiment.required')
  if (c.min !== undefined && c.max !== undefined) return `${c.min}-${c.max}`
  if (c.min !== undefined) return `≥${c.min}`
  if (c.max !== undefined) return `≤${c.max}`
  return ''
}

function getDefaultDisplay(v: typeof varList.value[0]): string {
  if (!v) return ''
  if (v.type === 'bool') return v.default_value ? 'true' : 'false'
  return String(v.default_value ?? '')
}
</script>

<template>
  <div class="variable-bar">
    <!-- Header -->
    <div class="vb-header">
      <span class="vb-title">{{ $t('experiment.variables') }}</span>
      <div class="vb-actions">
        <button class="vb-btn" @click="showAddForm = !showAddForm" :title="$t('experiment.addVariable')">
          <Plus :size="13" /> {{ $t('common.add') }}
        </button>
        <button class="vb-btn" @click="onImportCSV" :title="$t('experiment.csvImport')">
          <Upload :size="13" /> {{ $t('experiment.csvImport') }}
        </button>
        <button
          class="vb-btn vb-btn-danger"
          :disabled="!store.selectedVariable"
          @click="store.selectedVariable && store.removeVariable(store.selectedVariable)"
          :title="$t('experiment.deleteSelectedVar')"
        >
          <Trash2 :size="13" /> {{ $t('common.delete') }}
        </button>
        <button class="vb-btn" @click="onExportCSV" :title="$t('experiment.csvExport')">
          <Download :size="13" /> {{ $t('experiment.csvExport') }}
        </button>
        <label class="vb-checkbox">
          <input type="checkbox" v-model="store.batchMode" />
          <span>{{ $t('experiment.batchMode') }}</span>
        </label>
      </div>
    </div>

    <!-- Add form -->
    <div v-if="showAddForm" class="vb-add-form">
      <input
        v-model="newVarName"
        class="vb-input"
        :placeholder="$t('experiment.varNamePlaceholder')"
        @keyup.enter="onAdd"
        autofocus
      />
      <select v-model="newVarType" class="vb-select">
        <option value="int">int</option>
        <option value="float">float</option>
        <option value="str">str</option>
        <option value="bool">bool</option>
      </select>
      <button class="vb-btn vb-btn-primary" @click="onAdd">{{ $t('common.confirm') }}</button>
      <button class="vb-btn" @click="showAddForm = false">{{ $t('common.cancel') }}</button>
    </div>

    <!-- Table -->
    <div v-if="varList.length > 0" class="vb-table-wrap">
      <table class="vb-table">
        <thead>
          <tr>
            <th class="col-name">{{ $t('experiment.varName') }}</th>
            <th class="col-default">{{ $t('experiment.defaultValue') }}</th>
            <th class="col-constraint">{{ $t('experiment.constraint') }}</th>
            <th class="col-refs">{{ $t('experiment.refSteps') }}</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="v in varList"
            :key="v.name"
            :class="{ 'row-selected': store.selectedVariable === v.name }"
            @click="store.selectVariable(v.name)"
          >
            <td class="col-name">
              <span class="var-name">{{ v.name }}</span>
              <span class="var-type-tag">{{ v.type }}</span>
            </td>
            <td class="col-default">
              <input
                class="vb-inline-input"
                :value="getDefaultDisplay(v)"
                @input="(e) => onDefaultValueInput(v.name, (e.target as HTMLInputElement).value)"
              />
            </td>
            <td class="col-constraint">
              <input
                class="vb-inline-input"
                :value="getConstraintDisplay(v)"
                :placeholder="$t('experiment.constraintPlaceholder')"
                @input="(e) => onConstraintInput(v.name, (e.target as HTMLInputElement).value)"
              />
            </td>
            <td class="col-refs">
              <span v-if="selectedRefs.length && store.selectedVariable === v.name" class="ref-list">
                {{ selectedRefs.join(', ') }}
              </span>
              <span v-else class="ref-placeholder">-</span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Empty state -->
    <div v-else class="vb-empty">
      {{ $t('experiment.emptyVariables') }}
    </div>

    <!-- Batch data count -->
    <div v-if="store.batchData.length > 0" class="vb-batch-info">
      {{ $t('experiment.batchData') }}: {{ store.batchData.length }}{{ $t('experiment.rows') }}
    </div>
  </div>
</template>

<style scoped>
.variable-bar {
  border-top: 1px solid var(--color-border);
  border-bottom: 1px solid var(--color-border);
  background: var(--color-surface);
  max-height: 160px;
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}

.vb-header {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: 6px var(--space-md);
  flex-shrink: 0;
}

.vb-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-right: var(--space-sm);
}

.vb-actions {
  display: flex;
  align-items: center;
  gap: 4px;
  flex: 1;
}

.vb-btn {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  padding: 3px 8px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-surface);
  color: var(--color-text-secondary);
  font-size: 11px;
  cursor: pointer;
  transition: background var(--transition-fast);
}

.vb-btn:hover:not(:disabled) {
  background: var(--color-bg-soft);
  color: var(--color-text);
}

.vb-btn:disabled {
  opacity: 0.4;
  cursor: default;
}

.vb-btn-primary {
  background: var(--color-primary);
  color: #fff;
  border-color: var(--color-primary);
}

.vb-btn-primary:hover {
  opacity: 0.9;
}

.vb-btn-danger:not(:disabled):hover {
  border-color: var(--color-error);
  color: var(--color-error);
}

.vb-checkbox {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: var(--color-text-secondary);
  cursor: pointer;
  margin-left: var(--space-sm);
}

.vb-checkbox input {
  cursor: pointer;
}

/* Add form */
.vb-add-form {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px var(--space-md) 8px;
}

.vb-input {
  padding: 4px 8px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-bg-soft);
  color: var(--color-text);
  font-size: 12px;
  width: 120px;
}

.vb-input:focus {
  outline: none;
  border-color: var(--color-primary);
}

.vb-select {
  padding: 4px 6px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-bg-soft);
  color: var(--color-text);
  font-size: 12px;
}

/* Table */
.vb-table-wrap {
  overflow-x: auto;
  overflow-y: auto;
  flex: 1;
}

.vb-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

.vb-table th {
  padding: 4px var(--space-sm);
  text-align: left;
  font-size: 10px;
  font-weight: 600;
  color: var(--color-text-tertiary);
  text-transform: uppercase;
  background: var(--color-bg-soft);
  border-bottom: 1px solid var(--color-border);
  white-space: nowrap;
  position: sticky;
  top: 0;
  z-index: 1;
}

.vb-table td {
  padding: 3px var(--space-sm);
  border-bottom: 1px solid var(--color-border-subtle);
  white-space: nowrap;
}

.vb-table tbody tr {
  cursor: pointer;
  transition: background var(--transition-fast);
}

.vb-table tbody tr:hover {
  background: var(--color-bg-soft);
}

.vb-table tbody tr.row-selected {
  background: rgba(59, 130, 246, 0.08);
}

.col-name { min-width: 100px; }
.col-default { min-width: 80px; }
.col-constraint { min-width: 100px; }
.col-refs { min-width: 120px; }

.var-name {
  font-weight: 500;
  color: var(--color-text);
}

.var-type-tag {
  display: inline-block;
  margin-left: 6px;
  padding: 0 4px;
  border-radius: 3px;
  background: var(--color-bg-mute);
  color: var(--color-text-tertiary);
  font-size: 10px;
  font-family: monospace;
}

.vb-inline-input {
  width: 100%;
  padding: 2px 4px;
  border: 1px solid transparent;
  border-radius: 3px;
  background: transparent;
  color: var(--color-text);
  font-size: 12px;
  font-family: inherit;
}

.vb-inline-input:focus {
  outline: none;
  border-color: var(--color-primary);
  background: var(--color-surface);
}

.ref-list {
  color: var(--color-text-secondary);
  font-size: 11px;
}

.ref-placeholder {
  color: var(--color-text-tertiary);
}

.vb-empty {
  padding: 10px var(--space-md);
  font-size: 12px;
  color: var(--color-text-tertiary);
}

.vb-batch-info {
  padding: 3px var(--space-md);
  font-size: 11px;
  color: var(--color-primary);
  background: rgba(59, 130, 246, 0.05);
  border-top: 1px solid var(--color-border);
}
</style>
