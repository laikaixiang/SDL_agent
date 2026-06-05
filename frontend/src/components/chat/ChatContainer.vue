<script setup lang="ts">
import { ref, shallowRef, watch, nextTick, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { storeToRefs } from 'pinia'
import { useChatStore, MODE_PREFIX } from '@/stores/chat'
import { useAnalysisStore } from '@/stores/analysis'
import { useExperimentStore } from '@/stores/experiment'
import { useLayoutStore } from '@/stores/layout'
import { uploadPDF, sendAgentMessage, respondToAgent } from '@/api/chat'
import MessageBubble from './MessageBubble.vue'
import InputBar from './InputBar.vue'
import type {
  ToolCallInfo,
  TeamAgentInfo,
} from '@/types/chat'

const { t } = useI18n()
const store = useChatStore()
const analysisStore = useAnalysisStore()
const expStore = useExperimentStore()
const layoutStore = useLayoutStore()
const { messages, isStreaming, fieldConfirm } = storeToRefs(store)
const { showGuide, guideReply, guideProgress, guideDone, generating } = storeToRefs(analysisStore)
const inputText = ref('')
const chatEl = ref<HTMLDivElement>()
const newFieldName = ref('')
const editingFieldIndex = ref<number | null>(null)
const editFieldValue = ref('')

// Agent session
const sessionId = ref('')

// Per-turn agent state — these are the LIVE working buffers for the
// current agent run. When the run finishes (onDone) we commit them
// onto the AI's Message so they remain visible afterwards (like a
// normal conversation).
const liveToolCalls = shallowRef<Map<number, ToolCallInfo>>(new Map())
const livePendingQuestion = ref<{ question: string; options?: string } | null>(null)
const liveTeamAgents = ref<TeamAgentInfo[]>([])
// 心跳: 记录最后收到事件的时间(用于显示"agent 仍在工作..."指示)
const lastEventTime = ref(Date.now())
// 1s 轮询检查是否超过 8s 没有事件
const isAgentAlive = ref(false)
let aliveTimer: number | null = null

watch([() => store.isStreaming, lastEventTime], () => {
  if (aliveTimer) {
    clearInterval(aliveTimer)
    aliveTimer = null
  }
  if (store.isStreaming) {
    lastEventTime.value = Date.now()
    isAgentAlive.value = true
    aliveTimer = window.setInterval(() => {
      const gap = Date.now() - lastEventTime.value
      // 超过 8s 没有事件, 仍认为 agent 活着(只是慢/在等子 agent),
      // 但显示"仍在工作"指示
      isAgentAlive.value = gap < 60000  // 超过 60s 视为真正卡死
    }, 1000)
  } else {
    isAgentAlive.value = false
  }
}, { immediate: true })

onUnmounted(() => {
  if (aliveTimer) clearInterval(aliveTimer)
})

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
  // Agent mode: route to sendToAgent (bypasses prefix-based handlers)
  if (store.chatEngine === 'agent') {
    await sendToAgent(text)
  } else {
    await store.send(text)
  }
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

function buildAgentCallbacks(aiMsg: import('@/types/chat').Message) {
  let currentToolArgs: Record<string, unknown> = {}
  let thinkingStart = 0

  const cbs: import('@/types/chat').AgentCallbacks = {
    onThinkingChunk(t: string) {
      if (!thinkingStart) thinkingStart = Date.now()
      aiMsg.thinking = t
    },
    onThinkingComplete(t: string) {
      aiMsg.thinking = t
      aiMsg.thinking_duration = thinkingStart ? Math.round((Date.now() - thinkingStart) / 1000) : 0
    },
    onToolCallStart(i: number, n: string, cid: string) {
      const calls = new Map(liveToolCalls.value)
      calls.set(i, { index: i, name: n, callId: cid, arguments: {}, status: 'running' })
      liveToolCalls.value = calls
      currentToolArgs = {}
    },
    onToolCallArgs(i: number, d: string) {
      try {
        const parsed = JSON.parse(d)
        currentToolArgs = { ...currentToolArgs, ...parsed }
        const calls = new Map(liveToolCalls.value)
        const existing = calls.get(i)
        if (existing) {
          calls.set(i, { ...existing, arguments: { ...currentToolArgs } })
          liveToolCalls.value = calls
        }
      } catch { /* partial JSON, skip */ }
    },
    onToolCallEnd(i: number, n: string, a: Record<string, unknown>) {
      const calls = new Map(liveToolCalls.value)
      const existing = calls.get(i)
      const callId = existing?.callId ?? ''
      calls.set(i, { index: i, name: n, callId, arguments: a, status: 'running' })
      liveToolCalls.value = calls
    },
    onToolResult(i: number, n: string, r: string, s: string) {
      const calls = new Map(liveToolCalls.value)
      calls.set(i, {
        index: i,
        name: n,
        callId: calls.get(i)?.callId ?? '',
        arguments: calls.get(i)?.arguments ?? {},
        result: r,
        status: s === 'error' ? 'error' : 'done',
      })
      liveToolCalls.value = calls
      // If design_experiment tool returned JSON, load it into experiment store
      if (n === 'design_experiment' && s === 'success') {
        try {
          const resultJson = JSON.parse(r)
          if (resultJson.experiment_name || resultJson.steps) {
            expStore.loadFromJSON(resultJson)
            layoutStore.updateTaskStatus('experiment', 'completed')
          }
        } catch { /* not JSON or no experiment data */ }
      }
    },
    onAgentQuestion(q: string, o?: string) {
      // 把问题文本追加到 AI 消息正文(看起来像正常对话, 不再有独立卡片)
      const formatted = `\n\n${q}`
      if (aiMsg.content && !aiMsg.content.endsWith(q)) {
        aiMsg.content = aiMsg.content + formatted
      } else if (!aiMsg.content) {
        aiMsg.content = q
      }
      aiMsg.pendingQuestion = { question: q, options: o }
    },
    onTeamSpawn(_mode: string, agents: TeamAgentInfo[]) {
      liveTeamAgents.value = agents.map(a => ({ ...a, status: 'spawning' as const }))
    },
    onTeamProgress(agentId: string, status: string, summary?: string) {
      const idx = liveTeamAgents.value.findIndex(a => a.id === agentId)
      if (idx !== -1) {
        const updated = [...liveTeamAgents.value]
        updated[idx] = {
          ...updated[idx],
          status: status as TeamAgentInfo['status'],
          ...(summary ? { summary } : {}),
        }
        liveTeamAgents.value = updated
      }
    },
    onTeamDone(_mode: string, _results: unknown[]) {
      liveTeamAgents.value = liveTeamAgents.value.map(a => ({
        ...a,
        status: a.status === 'running' || a.status === 'spawning' ? 'done' : a.status,
      }))
    },
    onKeepalive(_timestamp: number) {
      // 心跳 — 只更新最后事件时间, 不渲染新消息
      lastEventTime.value = Date.now()
    },
    onToolProgress(name: string, current: number, total: number, message?: string) {
      // 把进度附加到当前 AI 消息(用 systemNote 显示,不占对话正文)
      aiMsg.systemNote = {
        kind: 'info',
        text: message || `🔧 ${name} 进度: ${current}/${total}`,
      }
    },
    onCompactionStart(message: string) {
      aiMsg.systemNote = { kind: 'compaction', text: `🔄 ${message}` }
    },
    onCompactionComplete(compactedCount: number, message: string) {
      aiMsg.systemNote = {
        kind: 'compaction',
        text: `✓ ${message} (${compactedCount} 条 → 压缩摘要)`,
      }
    },
    onCompactionError(error: string) {
      aiMsg.systemNote = { kind: 'info', text: `⚠ 压缩失败: ${error}` }
    },
    onTimeoutSummary(summary: string, timeoutSec: number) {
      // 项目总结作为新的 AI 消息显示(标注是超时自动生成的)
      store.addMessage('ai', '')
      const summaryMsg = store.messages[store.messages.length - 1]
      summaryMsg.content = summary
      summaryMsg.systemNote = {
        kind: 'timeout_summary',
        text: `⏱ 用户未在 ${timeoutSec}s 内响应, 已自动生成项目对话总结`,
      }
    },
    onError(m: string) {
      store.addMessage('ai', m)
    },
    onDone() {
      // Commit per-turn buffers to the AI message so they remain
      // visible after this turn ends. Live buffers reset for next turn.
      aiMsg.toolCalls = Array.from(liveToolCalls.value.values())
      aiMsg.teamAgents = liveTeamAgents.value.slice()
      // pendingQuestion is cleared when user answers (or remains
      // visible in the AI bubble if no answer was given)
      liveToolCalls.value = new Map()
      liveTeamAgents.value = []
    },
  }
  return cbs
}

async function sendToAgent(message: string) {
  // Reset per-turn buffers
  liveToolCalls.value = new Map()
  livePendingQuestion.value = null
  liveTeamAgents.value = []

  // Add user message first, then AI placeholder for incremental updates
  store.addMessage('user', message)
  store.addMessage('ai', '')
  const aiMsg = store.messages[store.messages.length - 1]
  let agentFullText = ''

  const callbacks = buildAgentCallbacks(aiMsg)
  callbacks.onTextChunk = (t: string) => {
    agentFullText = t
    aiMsg.content = t
  }
  callbacks.onTextComplete = (t: string) => {
    agentFullText = t
    aiMsg.content = t
  }

  const history = store.messages.slice(0, -1).map(m => ({ role: m.role, content: m.content }))
  // 传当前 chat_mode 给 agent endpoint, 让后端按 mode 过滤工具集
  // 详见 app.py:_get_mode_system_prompt
  const chatMode = store.currentMode || 'normal'
  await sendAgentMessage({
    message,
    session_id: sessionId.value,
    history,
    chat_mode: chatMode,
  }, callbacks)
}

async function handleAgentQuestionAnswer(answer: string) {
  // 清除待回答问题(答案会作为下一条 user 消息出现, 视觉上形成 Q&A 链)
  livePendingQuestion.value = null

  store.addMessage('user', answer)

  try {
    const result = await respondToAgent(sessionId.value, answer)
    if (result.type === 'text' || result.type === 'reply') {
      // After user answers, append a final AI summary message so the
      // Q&A pair is followed by the agent's continued response.
      store.addMessage('ai', result.reply)
    }
    // 'agent_continue' type means the agent will stream more events;
    // those will arrive via the next sendToAgent call.
  } catch (err) {
    store.addMessage('ai', (err as Error).message)
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
        :tool-calls="msg.toolCalls"
        :pending-question="msg.pendingQuestion"
        :system-note="msg.systemNote"
        :team-agents="msg.teamAgents"
        @question-select="handleAgentQuestionAnswer"
      />

      <!-- Live (in-progress) agent attachments for the current turn.
           They live alongside the messages list and are merged into the
           AI's Message on completion (onDone callback). -->
      <template v-if="liveToolCalls.size > 0 || liveTeamAgents.length > 0">
        <div v-if="liveToolCalls.size > 0" class="live-attachments">
          <MessageBubble
            v-for="[idx, tc] in liveToolCalls"
            :key="`live-tc-${idx}`"
            role="ai"
            content=""
            :tool-calls="[tc]"
          />
        </div>
        <div v-if="liveTeamAgents.length > 0" class="live-attachments">
          <MessageBubble
            role="ai"
            content=""
            :team-agents="liveTeamAgents"
          />
        </div>
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

      <!-- Agent 仍在工作指示 (长时间无事件但 isStreaming 为 true 时) -->
      <div v-if="isAgentAlive && isStreaming && liveToolCalls.size === 0 && !livePendingQuestion && liveTeamAgents.length === 0" class="alive-indicator">
        <span class="alive-indicator__pulse"></span>
        <span>Agent 仍在工作...</span>
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
        :disabled="isStreaming"
        :placeholder="livePendingQuestion ? '回答 Agent 的问题...' : (isStreaming ? $t('chat.thinking') : $t('chat.inputPlaceholder'))"
        :agent-question="livePendingQuestion"
        @send="livePendingQuestion ? handleAgentQuestionAnswer($event) : onSend($event)"
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

.live-attachments {
  width: 100%;
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
/* Agent 仍在工作指示器 (无活动输出时) */
.alive-indicator {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  margin: 4px 0;
  background: var(--color-bg-soft);
  border: 1px dashed var(--color-border);
  border-radius: var(--radius-full);
  font-size: 12px;
  color: var(--color-text-secondary);
}
.alive-indicator__pulse {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--color-primary);
  animation: alive-pulse 1.5s ease-in-out infinite;
}
@keyframes alive-pulse {
  0%, 100% { opacity: 0.4; transform: scale(0.9); }
  50% { opacity: 1; transform: scale(1.2); }
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
