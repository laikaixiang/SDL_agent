<script setup lang="ts">
import { ref, reactive, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useExperimentStore } from '@/stores/experiment'
import type { ExperimentStep } from '@/types/experiment'

const { t } = useI18n()

const props = defineProps<{
  step: ExperimentStep
  index: number
}>()

defineEmits<{ (e: 'close'): void }>()

const store = useExperimentStore()

const desc = ref(props.step.description || '')
const params = ref(JSON.stringify(props.step.params, null, 2))
const inputFile = ref(props.step.input_file || '')
const outputFile = ref(props.step.output_file || '')

const paramState = reactive<Record<string, 'normal' | 'undeclared' | 'linked' | 'invalid'>>({})
const variablesHint = reactive<Record<string, string>>({})
// 缓存用户输入值，避免 blur/re-render 丢失
const draftValues = reactive<Record<string, string>>({})

// 当 step 切换时重新初始化 draftValues
watch(() => props.step, (s) => {
  desc.value = s.description || ''
  params.value = JSON.stringify(s.params, null, 2)
  inputFile.value = s.input_file || ''
  outputFile.value = s.output_file || ''
  // 清空旧 draftValues
  for (const k of Object.keys(draftValues)) { delete draftValues[k] }
  // 初始化 draftValues
  for (const [k, v] of Object.entries(s.params || {})) {
    draftValues[k] = String(v ?? '')
    paramState[k] = 'normal'
    variablesHint[k] = ''
  }
}, { immediate: true })

function syncParamsFromDraft() {
  const p: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(draftValues)) {
    const trimmed = v.trim()
    if (!trimmed) {
      p[k] = v  // 保留空字符串
      continue
    }
    // 优先数字解析
    const num = Number(trimmed)
    if (!isNaN(num) && String(num) === trimmed) {
      p[k] = num
    } else {
      p[k] = v
    }
  }
  params.value = JSON.stringify(p, null, 2)
}

function onParamInput(paramKey: string, raw: string) {
  draftValues[paramKey] = raw
  // 输入变化时清除旧状态，等 blur 再判定
  if (paramState[paramKey] === 'undeclared' || paramState[paramKey] === 'linked' || paramState[paramKey] === 'invalid') {
    paramState[paramKey] = 'normal'
    variablesHint[paramKey] = ''
  }
  syncParamsFromDraft()
}

function onParamBlur(paramKey: string) {
  const value = draftValues[paramKey] ?? ''
  const trimmed = value.trim()
  if (!trimmed) { paramState[paramKey] = 'normal'; return }

  // 纯数字 / 浮点数 → 正常字面量
  const num = Number(trimmed)
  if (!isNaN(num) && String(num) === trimmed) {
    paramState[paramKey] = 'normal'
    syncParamsFromDraft()
    return
  }

  // 纯数字不允许作为变量名
  if (/^\d+$/.test(trimmed)) {
    paramState[paramKey] = 'invalid'
    variablesHint[paramKey] = t('experiment.invalidVarName')
    return
  }

  // 检查是否为合法变量名格式
  const isVarName = /^[a-zA-Z_][a-zA-Z0-9_]*$/.test(trimmed)

  if (isVarName) {
    if (store.isVariableDeclared(trimmed)) {
      paramState[paramKey] = 'linked'
      const v = store.variables[trimmed]
      variablesHint[paramKey] = `→ ${v?.default_value ?? '?'}`
    } else {
      paramState[paramKey] = 'undeclared'
    }
  } else {
    // 含运算符或其他 → 交由后端表达式引擎处理
    paramState[paramKey] = 'normal'
  }
}

function onParamKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter') {
    e.preventDefault()
    // 找到当前输入框在 param-grid 中的所有 input，聚焦下一个
    const inputs = (e.currentTarget as HTMLElement)?.closest('.param-grid')?.querySelectorAll('input.param-input') as NodeListOf<HTMLInputElement> | undefined
    if (!inputs || inputs.length === 0) return
    const current = e.target as HTMLInputElement
    const idx = Array.from(inputs).indexOf(current)
    if (idx >= 0 && idx < inputs.length - 1) {
      inputs[idx + 1].focus()
    } else {
      current.blur()
    }
  }
}

function onDeclareVariable(paramKey: string, varName: string) {
  if (!varName.trim()) return
  let varType: 'int' | 'float' | 'str' | 'bool' = 'int'
  if (toolDef) {
    const paramDef = toolDef.params?.[paramKey]
    if (paramDef?.type === 'str') varType = 'str'
    else if (paramDef?.type === 'float') varType = 'float'
    else if (paramDef?.type === 'bool') varType = 'bool'
  }
  store.addVariable(varName.trim(), varType)
  paramState[paramKey] = 'linked'
  variablesHint[paramKey] = `→ ${t('experiment.fillDefaultValue')}`
}

function onSave() {
  let parsedParams: Record<string, unknown> = {}
  try {
    parsedParams = JSON.parse(params.value)
  } catch {
    // keep as raw string for now
  }

  store.updateStep(props.index, {
    ...props.step,
    description: desc.value,
    params: parsedParams,
    input_file: inputFile.value || undefined,
    output_file: outputFile.value || undefined,
  })
}

const toolDef = props.step.type === 'tool'
  ? store.hardwareTools.find(t => t.name === props.step.name)
  : null
</script>

<template>
  <div class="step-editor">
    <div class="editor-fields">
      <label>{{ $t('experiment.description') }}</label>
      <input v-model="desc" class="editor-input" :placeholder="$t('experiment.stepDesc')" />

      <!-- Tool params: use tool definition to render fields -->
      <template v-if="step.type === 'tool' && toolDef">
        <label>{{ $t('experiment.params') }}</label>
        <div class="param-grid">
          <div v-for="(v, k) in toolDef.params" :key="k" class="param-field">
            <span class="param-label">{{ k }}
              <span v-if="v.required" class="param-req">*</span>
              <span v-else class="param-opt">{{ $t('experiment.optional') }}</span>
            </span>
            <span class="param-type">{{ v.type }}</span>
            <div class="param-input-row">
              <input
                class="editor-input param-input"
                :class="{
                  'param-undeclared': paramState[k] === 'undeclared',
                  'param-linked': paramState[k] === 'linked',
                  'param-invalid': paramState[k] === 'invalid',
                }"
                :placeholder="v.description || k"
                :value="draftValues[k] ?? String(step.params[k] ?? v.default ?? '')"
                @input="(e) => onParamInput(k, (e.target as HTMLInputElement).value)"
                @blur="() => onParamBlur(k)"
                @keydown="onParamKeydown"
              />
              <button
                v-if="paramState[k] === 'undeclared'"
                class="btn-declare"
                @click="onDeclareVariable(k, draftValues[k] ?? '')"
              >{{ $t('experiment.declare') }}</button>
              <span v-if="paramState[k] === 'linked'" class="param-linked-hint">→ {{ variablesHint[k] }}</span>
              <span v-if="paramState[k] === 'invalid'" class="param-invalid-hint">{{ variablesHint[k] }}</span>
            </div>
          </div>
        </div>
      </template>

      <!-- Software params -->
      <template v-if="step.type === 'software'">
        <label>{{ $t('experiment.inputFile') }}</label>
        <input v-model="inputFile" class="editor-input" :placeholder="$t('experiment.csvPath')" />
        <label>{{ $t('experiment.outputDir') }}</label>
        <input v-model="outputFile" class="editor-input" :placeholder="$t('experiment.outputDirOptional')" />
      </template>

      <!-- Helper / generic params: raw JSON editor -->
      <template v-if="step.type === 'helper' || (step.type === 'tool' && !toolDef)">
        <label>{{ $t('experiment.paramsJson') }}</label>
        <textarea v-model="params" class="editor-textarea" rows="4" spellcheck="false" />
      </template>
    </div>

    <div class="editor-actions">
      <button class="btn-save" @click="onSave">{{ $t('common.save') }}</button>
      <button class="btn-cancel" @click="$emit('close')">{{ $t('common.cancel') }}</button>
    </div>
  </div>
</template>

<style scoped>
.step-editor {
  padding: var(--space-md);
  border-top: 1px solid var(--color-border);
  background: var(--color-bg-soft);
}

.editor-fields {
  display: flex; flex-direction: column; gap: var(--space-sm);
}

.editor-fields label {
  font-size: 11px; font-weight: 600; color: var(--color-text-secondary);
  text-transform: uppercase; letter-spacing: 0.5px;
}

.editor-input {
  padding: 6px 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-surface);
  color: var(--color-text);
  font-size: 13px;
  width: 100%;
}

.editor-input:focus { outline: none; border-color: var(--color-primary); }

.editor-textarea {
  padding: 8px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-surface);
  color: var(--color-text);
  font-size: 12px;
  font-family: monospace;
  resize: vertical;
  width: 100%;
}

.editor-textarea:focus { outline: none; border-color: var(--color-primary); }

.param-grid { display: flex; flex-direction: column; gap: 6px; }

.param-field { display: flex; flex-direction: column; gap: 2px; }

.param-label { font-size: 12px; color: var(--color-text); }
.param-req { color: var(--color-error); font-size: 10px; }
.param-opt { color: var(--color-text-tertiary); font-size: 10px; }
.param-type { font-size: 10px; color: var(--color-text-tertiary); }
.param-input { margin-top: 2px; }

.editor-actions {
  display: flex; gap: var(--space-sm); justify-content: flex-end;
  margin-top: var(--space-md);
}

.btn-save, .btn-cancel {
  padding: 6px 16px; border: none; border-radius: var(--radius-sm);
  font-size: 13px; cursor: pointer;
}

.btn-save { background: var(--color-primary); color: #fff; }
.btn-save:hover { opacity: 0.9; }
.btn-cancel { background: var(--color-bg-mute); color: var(--color-text); }

.param-input-row { display: flex; align-items: center; gap: 4px; }
.param-input.param-undeclared { border-color: var(--color-error); background: rgba(220, 38, 38, 0.05); }
.param-input.param-linked { border-color: #10b981; background: rgba(16, 185, 129, 0.05); }
.param-input.param-invalid { border-color: #f59e0b; background: rgba(245, 158, 11, 0.05); }
.param-linked-hint { font-size: 11px; color: #10b981; white-space: nowrap; min-width: 40px; }
.param-invalid-hint { font-size: 11px; color: #f59e0b; white-space: nowrap; }
.btn-declare { padding: 2px 8px; border: 1px solid var(--color-error); border-radius: 4px; background: var(--color-error); color: #fff; font-size: 11px; cursor: pointer; white-space: nowrap; flex-shrink: 0; }
.btn-declare:hover { opacity: 0.85; }
</style>
