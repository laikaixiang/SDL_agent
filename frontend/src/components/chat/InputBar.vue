<script setup lang="ts">
import { ref } from 'vue'
import { Send, Paperclip, MessageSquare, FileText, Cpu, FlaskConical, BarChart3 } from 'lucide-vue-next'
import { useChatStore, type ChatMode } from '@/stores/chat'

const store = useChatStore()
const modelValue = defineModel<string>('modelValue', { default: '' })
defineProps<{ disabled?: boolean; placeholder?: string }>()
const emit = defineEmits<{ send: [text: string]; fileSelected: [file: File] }>()

const textarea = ref<HTMLTextAreaElement>()
const fileInput = ref<HTMLInputElement>()

const modes: { id: ChatMode; label: string; icon: typeof MessageSquare; hint: string }[] = [
  { id: 'normal',      label: '对话',     icon: MessageSquare,   hint: '自由对话' },
  { id: 'extraction',  label: '文献提取', icon: FileText,        hint: '输入"帮我搜寻：<描述>"开始提取' },
  { id: 'hardware',    label: '硬件控制', icon: Cpu,             hint: '输入"硬件控制：<指令>"操控设备' },
  { id: 'experiment',  label: '实验设计', icon: FlaskConical,    hint: '输入"实验设计：<描述>"设计实验' },
  { id: 'analysis',    label: '数据分析', icon: BarChart3,       hint: '输入"数据分析"开始分析' },
]

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    submit()
  }
}

function submit() {
  const text = modelValue.value.trim()
  if (!text) return
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
      <textarea
        ref="textarea"
        v-model="modelValue"
        class="input-textarea"
        :placeholder="placeholder || '输入消息... (Enter 发送, Shift+Enter 换行)'"
        :disabled="disabled"
        rows="1"
        @keydown="onKeydown"
        @input="autoResize"
      />
      <button class="send-btn" :disabled="disabled || !modelValue.trim()" @click="submit">
        <Send :size="18" />
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
.input-textarea {
  flex: 1; min-height: var(--input-height); max-height: 200px; resize: none;
  padding: 12px 16px; border: 1px solid var(--color-border); border-radius: var(--radius-md);
  background: var(--color-bg-soft); color: var(--color-text); font-size: 14px; line-height: 1.5;
  outline: none; transition: border var(--transition-fast);
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
</style>
