<script setup lang="ts">
import { ref, shallowRef, watch, nextTick } from 'vue'
import { useI18n } from 'vue-i18n'
import { storeToRefs } from 'pinia'
import { useChatStore, MODE_PREFIX } from '@/stores/chat'
import { useAnalysisStore } from '@/stores/analysis'
import { uploadPDF, sendAgentMessage, respondToAgent } from '@/api/chat'
import MessageBubble from './MessageBubble.vue'
import InputBar from './InputBar.vue'
import AgentToolCard from './AgentToolCard.vue'
import AgentQuestionCard from './AgentQuestionCard.vue'
import ThinkingBubble from './ThinkingBubble.vue'
import type { ToolCallInfo, AgentQuestion } from '@/types/chat'
import { Plus, X } from 'lucide-vue-next'

const { t } = useI18n()
const store = useChatStore()
const analysisStore = useAnalysisStore()
const { messages, isStreaming, fieldConfirm } = storeToRefs(store)
const { showGuide, guideReply, guideProgress, guideDone, generating } = storeToRefs(analysisStore)
const inputText = ref('')
const chatEl = ref<HTMLDivElement>()
const newFieldName = ref('')
const editingFieldIndex = ref<number | null>(null)
const editFieldValue = ref('')

// Agent state
const sessionId = ref('')
const agentActiveToolCalls = shallowRef<Map<number, ToolCallInfo>>(new Map())
const agentQuestion = ref<AgentQuestion | null>(null)
const agentThinking = ref('')
const agentThinkingDuration = ref(0)
const isAgentResponding = ref(false)

function startEditField(index: number, current: string) {
  editingFieldIndex.value = index
  editFieldValue.value = current
}

function saveEditField() {
  if (editingFieldIndex.value !== null && editFieldValue.value.trim()) {
    store.updateConfirmField(editingFieldIndex.value, editFieldValue.value.trim())
  }
  editingFieldIndex.value = null
  editFieldValue.value = ''
}

function addField() {
  if (newFieldName.value.trim()) {
    store.addConfirmField(newFieldName.value.trim())
    newFieldName.value = ''
  }
}

function scrollToBottom() {
  nextTick(() => {
    if (chatEl.value) chatEl.value.scrollTop = chatEl.value.scrollHeight
  })
}

watch(() => store.messages.length, scrollToBottom)

async function onSend(text: string) {
  inputText.value = ''
  if (showGuide.value && !guideDone.value) {
    store.addMessage('user', text || t('chat.skip'))
    await analysisStore.submitGuideAnswer(text)
    scrollToBottom()
    return
  }
  await store.send(text)
  scrollToBottom()
}

async function onFileSelected(file: File) {
  try {
    const result = await uploadPDF(file)
    if (result.success) {
      inputText.value = `${MODE_PREFIX.extraction}${result.filename}`
    }
  } catch (err) {
    console.error('文件上传失败:', err)
  }
}

async function onCancelExtraction() {
  await store.cancelExtractionTask()
}

function buildAgentCallbacks() {
  let currentToolArgs: Record<string, unknown> = {}
  let thinkingStart = 0

  const cbs: import('@/types/chat').AgentCallbacks = {
    onThinkingChunk(t: string) {
      if (!thinkingStart) thinkingStart = Date.now()
      agentThinking.value = t
    },
    onThinkingComplete(t: string) {
      agentThinking.value = t
      agentThinkingDuration.value = thinkingStart ? Math.round((Date.now() - thinkingStart) / 1000) : 0
    },
    onToolCallStart(i: number, n: string) {
      const calls = new Map(agentActiveToolCalls.value)
      calls.set(i, { index: i, name: n, callId: '', arguments: {}, status: 'running' })
      agentActiveToolCalls.value = calls
      currentToolArgs = {}
    },
    onToolCallArgs(i: number, d: string) {
      try {
        const parsed = JSON.parse(d)
        currentToolArgs = { ...currentToolArgs, ...parsed }
        const calls = new Map(agentActiveToolCalls.value)
        const existing = calls.get(i)
        if (existing) {
          calls.set(i, { ...existing, arguments: { ...currentToolArgs } })
          agentActiveToolCalls.value = calls
        }
      } catch { /* partial JSON, skip */ }
    },
    onToolCallEnd(i: number, n: string, a: Record<string, unknown>) {
      const calls = new Map(agentActiveToolCalls.value)
      calls.set(i, { index: i, name: n, callId: '', arguments: a, status: 'running' })
      agentActiveToolCalls.value = calls
    },
    onToolResult(i: number, n: string, r: string, s: string) {
      const calls = new Map(agentActiveToolCalls.value)
      calls.set(i, { index: i, name: n, callId: '', arguments: calls.get(i)?.arguments ?? {}, result: r, status: s as 'done' | 'error' })
      agentActiveToolCalls.value = calls
    },
    onAgentQuestion(q: string, o?: string) {
      agentQuestion.value = { question: q, options: o }
    },
    onError(m: string) {
      store.addMessage('ai', m)
    },
    onDone() {
      isAgentResponding.value = false
    },
  }
  return cbs
}

async function sendToAgent(message: string) {
  isAgentResponding.value = true
  agentActiveToolCalls.value = new Map()
  agentQuestion.value = null
  agentThinking.value = ''
  agentThinkingDuration.value = 0

  const callbacks = buildAgentCallbacks()
  callbacks.onTextChunk = (t: string) => {
    store.addMessage('ai', t)
  }
  callbacks.onTextComplete = (t: string) => {
    const msgs = store.messages
    const last = msgs[msgs.length - 1]
    if (last && last.role === 'ai') {
      last.content = t
    }
  }

  const history = store.messages.map(m => ({ role: m.role, content: m.content }))
  await sendAgentMessage({ message, session_id: sessionId.value, history }, callbacks)
}

async function handleAgentQuestionAnswer(answer: string) {
  agentQuestion.value = null
  isAgentResponding.value = true

  store.addMessage('user', answer)

  try {
    const result = await respondToAgent(sessionId.value, answer)
    if (result.type === 'agent_continue') {
      // The agent will send more events, continue listening
    } else if (result.type === 'text' || result.type === 'reply') {
      store.addMessage('ai', result.reply)
    }
  } catch (err) {
    store.addMessage('ai', (err as Error).message)
  } finally {
    isAgentResponding.value = false
  }
}
</script>

<template>
  <div class="chat-container">
    <div class="chat-messages" ref="chatEl">
      <MessageBubble
        v-for="(msg, i) in messages"
        :key="i"
        :role="msg.role"
        :content="msg.content"
        :thinking="msg.thinking"
        :thinking-duration="msg.thinking_duration"
        :timestamp="msg.timestamp"
      />

      <!-- Agent thinking bubble -->
      <template v-if="agentThinking">
        <ThinkingBubble :text="agentThinking" :duration="agentThinkingDuration" />
      </template>

      <!-- Agent tool calls -->
      <template v-if="agentActiveToolCalls.size > 0">
        <AgentToolCard
          v-for="[idx, tc] in agentActiveToolCalls"
          :key="idx"
          :index="tc.index"
          :name="tc.name"
          :args="tc.arguments"
          :result="tc.result"
          :status="tc.status"
        />
      </template>

      <!-- Agent question -->
      <template v-if="agentQuestion">
        <AgentQuestionCard
          :question="agentQuestion.question"
          :options="agentQuestion.options"
          @select="handleAgentQuestionAnswer"
        />
      </template>

      <!-- Algorithm guide card -->
      <div v-if="showGuide" class="guide-card">
        <div class="guide-progress-bar">
          <div class="guide-progress-fill"
            :style="{ width: guideProgress === 'complete' ? '100%' : (parseInt(guideProgress) / 4 * 100) + '%' }">
          </div>
        </div>
        <div class="guide-progress-label">{{ guideProgress === 'complete' ? $t('common.complete') : guideProgress }}</div>
        <div class="guide-reply">{{ guideReply }}</div>
        <div v-if="!guideDone" class="guide-actions">
          <button class="btn-guide-cancel" @click="analysisStore.cancelGuide()">{{ $t('common.cancel') }}</button>
          <button class="btn-guide-back" @click="analysisStore.guideGoBack()">{{ $t('common.back') }}</button>
          <button class="btn-guide-submit" :disabled="generating" @click="onSend(inputText)">{{ $t('common.submit') }}</button>
        </div>
      </div>

      <!-- Inline field confirm card -->
      <div v-if="fieldConfirm" class="confirm-card">
        <div class="confirm-label">{{ $t('chat.fieldConfirmHint') }}</div>
        <div class="confirm-fields">
          <div v-for="(f, i) in fieldConfirm.fields" :key="i" class="field-tag-row">
            <template v-if="editingFieldIndex !== i">
              <span class="field-tag" @dblclick="startEditField(i, f)">{{ f }}</span>
              <button class="field-del" :title="$t('common.delete')" @click="store.removeConfirmField(i)"><X :size="12" /></button>
            </template>
            <template v-else>
              <input
                v-model="editFieldValue"
                class="field-edit-input"
                @keydown.enter="saveEditField()"
                @keydown.escape="editingFieldIndex = null"
                @blur="saveEditField()"
              />
            </template>
          </div>
        </div>
        <div class="confirm-add-row">
          <input
            v-model="newFieldName"
            class="field-add-input"
            :placeholder="$t('chat.addNewField')"
            @keydown.enter="addField()"
          />
          <button class="field-add-btn" :disabled="!newFieldName.trim()" @click="addField()"><Plus :size="14" /></button>
        </div>
        <div class="confirm-actions">
          <button class="confirm-btn-yes" @click="store.confirmExtraction()">{{ $t('chat.confirmExtraction') }}</button>
          <button class="confirm-btn-no" @click="store.cancelExtraction()">{{ $t('chat.modifyRequirement') }}</button>
        </div>
      </div>
    </div>
    <div class="chat-input-area">
      <InputBar
        v-model="inputText"
        :disabled="isStreaming || isAgentResponding"
        :placeholder="agentQuestion ? '回答 Agent 的问题...' : (isStreaming ? $t('chat.thinking') : $t('chat.inputPlaceholder'))"
        :agent-question="agentQuestion"
        @send="agentQuestion ? handleAgentQuestionAnswer($event) : onSend($event)"
        @file-selected="onFileSelected"
        @cancel-extraction="onCancelExtraction"
      />
    </div>
  </div>
</template>

<style scoped>
.chat-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  align-items: center;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding-bottom: var(--space-md);
  width: 50%;
  min-width: 400px;
}

.chat-input-area {
  flex-shrink: 0;
  width: 50%;
  min-width: 400px;
}

/* Inline field confirm card */
.confirm-card {
  margin: var(--space-sm) 0;
  padding: var(--space-md);
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
}
.confirm-label { font-size: 12px; color: var(--color-text-tertiary); margin-bottom: 8px; }
.confirm-fields {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-bottom: var(--space-sm);
}
.field-tag-row {
  display: flex;
  align-items: center;
  gap: 1px;
  background: var(--color-primary-soft);
  border-radius: var(--radius-full);
  padding-left: 10px;
}
.field-tag {
  font-size: 12px;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  color: var(--color-primary);
  cursor: pointer;
}
.field-tag:hover { text-decoration: underline; }
.field-del {
  width: 20px; height: 20px; display: flex; align-items: center; justify-content: center;
  border: none; background: transparent; color: var(--color-primary); cursor: pointer;
  border-radius: 50%; opacity: 0.6; padding: 0;
}
.field-del:hover { opacity: 1; background: rgba(0,0,0,0.05); }
.field-edit-input {
  width: 120px; padding: 2px 8px; border: 1px solid var(--color-primary);
  border-radius: var(--radius-full); font-size: 12px; outline: none;
  background: var(--color-surface);
}
.confirm-add-row {
  display: flex; gap: 4px; margin-bottom: var(--space-md); align-items: center;
}
.field-add-input {
  flex: 1; padding: 4px 10px; border: 1px dashed var(--color-border);
  border-radius: var(--radius-sm); font-size: 12px; outline: none;
  background: transparent; color: var(--color-text);
}
.field-add-input:focus { border-color: var(--color-primary); border-style: solid; }
.field-add-btn {
  width: 26px; height: 26px; display: flex; align-items: center; justify-content: center;
  border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  background: var(--color-surface); color: var(--color-text-secondary); cursor: pointer;
}
.field-add-btn:disabled { opacity: 0.4; cursor: default; }
.field-add-btn:not(:disabled):hover { background: var(--color-bg-soft); color: var(--color-text); }
.confirm-actions {
  display: flex;
  gap: var(--space-sm);
}
.confirm-btn-yes,
.confirm-btn-no {
  padding: 6px 16px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-size: 13px;
  cursor: pointer;
  transition: all var(--transition-fast);
}
.confirm-btn-yes {
  background: var(--color-primary);
  color: #fff;
  border-color: var(--color-primary);
}
.confirm-btn-yes:hover { opacity: 0.85; }
.confirm-btn-no {
  background: var(--color-surface);
  color: var(--color-text-secondary);
}
.confirm-btn-no:hover { background: var(--color-bg-soft); color: var(--color-text); }

/* Algorithm guide card */
.guide-card {
  margin: var(--space-sm) 0;
  padding: var(--space-lg);
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
  border-radius: var(--radius-lg);
}
.guide-progress-bar {
  width: 100%;
  height: 6px;
  background: #e5e7eb;
  border-radius: 3px;
  margin-bottom: 8px;
  overflow: hidden;
}
.guide-progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #10b981, #34d399);
  border-radius: 3px;
  transition: width 0.4s ease;
}
.guide-progress-label {
  font-size: 12px;
  color: #059669;
  font-weight: 600;
  margin-bottom: 12px;
  text-align: right;
}
.guide-reply {
  font-size: 14px;
  color: var(--color-text);
  line-height: 1.7;
  white-space: pre-wrap;
}
.guide-actions {
  display: flex;
  gap: var(--space-sm);
  justify-content: flex-end;
  margin-top: var(--space-md);
}
.btn-guide-cancel, .btn-guide-back, .btn-guide-submit {
  padding: 6px 16px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-size: 13px;
  cursor: pointer;
  transition: all var(--transition-fast);
}
.btn-guide-cancel { background: var(--color-bg-mute); color: var(--color-text-secondary); }
.btn-guide-cancel:hover { background: #e5e7eb; color: var(--color-text); }
.btn-guide-back { background: #fef3c7; color: #92400e; border-color: #fcd34d; }
.btn-guide-back:hover { background: #fde68a; }
.btn-guide-submit { background: #10b981; color: #fff; border-color: #10b981; font-weight: 600; }
.btn-guide-submit:hover { background: #059669; }
.btn-guide-submit:disabled { opacity: 0.5; cursor: default; }
</style>
