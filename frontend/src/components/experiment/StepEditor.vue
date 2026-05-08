<script setup lang="ts">
import { ref } from 'vue'
import { useExperimentStore } from '@/stores/experiment'
import type { ExperimentStep } from '@/types/experiment'

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
      <label>描述</label>
      <input v-model="desc" class="editor-input" placeholder="步骤描述" />

      <!-- Tool params: use tool definition to render fields -->
      <template v-if="step.type === 'tool' && toolDef">
        <label>参数</label>
        <div class="param-grid">
          <div v-for="(v, k) in toolDef.params" :key="k" class="param-field">
            <span class="param-label">{{ k }}
              <span v-if="v.required" class="param-req">*</span>
              <span v-else class="param-opt">可选</span>
            </span>
            <span class="param-type">{{ v.type }}</span>
            <input
              class="editor-input param-input"
              :placeholder="v.description || k"
              :value="String(step.params[k] ?? v.default ?? '')"
              @input="(e) => {
                const val = (e.target as HTMLInputElement).value
                const p = { ...step.params }
                p[k] = v.type === 'number' || v.type === 'int' ? Number(val) : val
                params = JSON.stringify(p, null, 2)
              }"
            />
          </div>
        </div>
      </template>

      <!-- Software params -->
      <template v-if="step.type === 'software'">
        <label>输入文件</label>
        <input v-model="inputFile" class="editor-input" placeholder="CSV 文件路径" />
        <label>输出目录</label>
        <input v-model="outputFile" class="editor-input" placeholder="输出目录（可选）" />
      </template>

      <!-- Helper / generic params: raw JSON editor -->
      <template v-if="step.type === 'helper' || (step.type === 'tool' && !toolDef)">
        <label>参数 (JSON)</label>
        <textarea v-model="params" class="editor-textarea" rows="4" spellcheck="false" />
      </template>
    </div>

    <div class="editor-actions">
      <button class="btn-save" @click="onSave">保存</button>
      <button class="btn-cancel" @click="$emit('close')">取消</button>
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
</style>
