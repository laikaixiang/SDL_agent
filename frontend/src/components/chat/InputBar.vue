<script setup lang="ts">
import { ref } from 'vue'
import { Send, Paperclip, MessageSquare, FileText, Cpu, FlaskConical, BarChart3 } from 'lucide-vue-next'
import { useChatStore, MODE_LABEL, type ChatMode } from '@/stores/chat'

const store = useChatStore()
const modelValue = defineModel<string>('modelValue', { default: '' })
defineProps<{ disabled?: boolean; placeholder?: string }>()
const emit = defineEmits<{ send: [text: string]; fileSelected: [file: File]; cancelExtraction: [] }>()

const textarea = ref<HTMLTextAreaElement>()
const fileInput = ref<HTMLInputElement>()

const modes: { id: ChatMode; label: string; icon: typeof MessageSquare; hint: string }[] = [
  { id: 'normal',      label: '对话',     icon: MessageSquare,   hint: '自由对话' },
  { id: 'extraction',  label: '文献提取', icon: FileText,        hint: '自动添加"帮我搜寻："前缀进行文献提取' },
  { id: 'hardware',    label: '硬件控制', icon: Cpu,             hint: '输入硬件指令操控设备' },
  { id: 'experiment',  label: '实验设计', icon: FlaskConical,    hint: '输入实验描述设计实验' },
  { id: 'analysis',    label: '数据分析', icon: BarChart3,       hint: '输入数据分析需求' },
]

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    submit()
  }
}

function submit() {
  if (store.extractionRunning) {
    emit('cancelExtraction')
    return
  }
  const text = modelValue.value.trim()
  if (!text && store.currentMode === 'normal') return
  emit('send', text)
  modelValue.value = ''
}

function autoResize() {
  const el = textarea.value
  if (el) { el.style.height = 'auto'; el.style.height = el.scrollHeight + 'px' }
}

function triggerFileInput() {
  fileInput.value?.click()
}

function onFileChange(e: Event) {
  const input = e.target as HTMLInputElement
  if (input.files && input.files.length > 0) {
    emit('fileSelected', input.files[0])
    input.value = ''
  }
}
</script>

<template>
  <div class="input-bar">
    <div class="input-row">
      <div class="input-wrapper">
        <div v-if="store.currentMode !== 'normal'" class="extraction-bubble">
          <span>{{ MODE_LABEL[store.currentMode] }}</span>
          <button class="bubble-close" @click="store.setMode('normal')">×</button>
        </div>
        <textarea
          ref="textarea"
          v-model="modelValue"
          class="input-textarea"
          :class="{ 'with-bubble': store.currentMode !== 'normal' }"
          :placeholder="store.extractionRunning ? '提取任务运行中...' : (placeholder || (store.currentMode === 'extraction' ? '按 Enter 直接使用默认提取配置 (FAPbI3 钙钛矿钝化剂)...' : '输入消息... (Enter 发送, Shift+Enter 换行)'))"
          :disabled="disabled || store.extractionRunning"
          rows="1"
          @keydown="onKeydown"
          @input="autoResize"
        />
      </div>
      <button
        class="send-btn"
        :class="{ 'cancel-mode': store.extractionRunning }"
        :disabled="disabled || (!store.extractionRunning && !modelValue.trim() && store.currentMode === 'normal')"
        @click="submit"
      >
        <div v-if="store.extractionRunning" class="btn-spinner" />
        <Send v-else :size="18" />
      </button>
    </div>
    <div class="input-toolbar">
      <input
        ref="fileInput"
        type="file"
        accept=".pdf,.csv,.txt,.json,.xlsx,.xls"
        class="file-input-hidden"
        @change="onFileChange"
      />
      <button class="toolbar-btn" title="上传文件" @click="triggerFileInput">
        <Paperclip :size="15" />
      </button>
      <span class="toolbar-divider" />
      <button
        v-for="m in modes"
        :key="m.id"
        class="toolbar-btn"
        :class="{ active: store.currentMode === m.id }"
        :title="m.hint"
        @click="store.setMode(m.id)"
      >
        <component :is="m.icon" :size="16" />
      </button>
    </div>
  </div>
</template>

<style scoped>
.input-bar {
  padding: var(--space-md) var(--space-xl);
  background: var(--color-surface);
  border-top: 1px solid var(--color-border);
}
.input-row { display: flex; align-items: flex-end; gap: var(--space-sm); }
.input-wrapper { position: relative; flex: 1; min-width: 0; }
.input-textarea {
  display: block;
  width: 100%;
  box-sizing: border-box;
  min-height: var(--input-height); max-height: 200px; resize: none;
  padding: 12px 16px; border: 1px solid var(--color-border); border-radius: var(--radius-md);
  background: var(--color-bg-soft); color: var(--color-text); font-size: 14px; line-height: 1.5;
  outline: none; transition: border var(--transition-fast);
}
.input-textarea.with-bubble {
  padding-left: 130px;
}
.input-textarea:focus { border-color: var(--color-primary); }
.send-btn {
  width: 48px; height: 48px; border: none; border-radius: var(--radius-full);
  background: var(--color-primary); color: #fff; display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; transition: opacity var(--transition-fast);
}
.send-btn:disabled { opacity: 0.4; cursor: default; }
.send-btn:not(:disabled):hover { opacity: 0.85; }

.input-toolbar {
  display: flex;
  align-items: center;
  gap: 2px;
  margin-top: var(--space-xs);
  padding-left: 2px;
}
.file-input-hidden { display: none; }
.toolbar-btn {
  width: 32px; height: 32px; display: flex; align-items: center; justify-content: center;
  border: none; border-radius: var(--radius-sm); background: transparent;
  color: var(--color-text-tertiary); cursor: pointer;
  transition: color var(--transition-fast), background var(--transition-fast);
}
.toolbar-btn:hover { color: var(--color-text); background: var(--color-bg-soft); }
.toolbar-btn.active { color: var(--color-primary); background: var(--color-primary-soft); }
.toolbar-divider {
  width: 1px; height: 18px; background: var(--color-border); margin: 0 4px;
}

.extraction-bubble {
  position: absolute;
  left: 10px;
  top: 12px;
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  background: #fffbeb;
  border: 1px solid #fde68a;
  border-radius: var(--radius-full);
  font-size: 12px;
  color: #92400e;
  white-space: nowrap;
  z-index: 1;
  pointer-events: auto;
  line-height: 1.5;
}

.bubble-close {
  border: none;
  background: transparent;
  color: #92400e;
  font-size: 14px;
  cursor: pointer;
  padding: 0;
  line-height: 1;
  opacity: 0.5;
}

.bubble-close:hover {
  opacity: 1;
}

/* Spinner and cancel mode */
.btn-spinner {
  width: 18px;
  height: 18px;
  border: 2px solid rgba(255,255,255,0.3);
  border-top-color: #fff;
  border-radius: 50%;
  animation: btn-spin 0.6s linear infinite;
}
@keyframes btn-spin {
  to { transform: rotate(360deg); }
}
.send-btn.cancel-mode {
  background: var(--color-error);
}
.send-btn.cancel-mode:hover:not(:disabled) {
  opacity: 0.85;
}
</style>
