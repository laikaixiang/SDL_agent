import { ref, computed } from 'vue'
import { defineStore } from 'pinia'
import type { Message } from '@/types/chat'
import { sendChatMessage } from '@/api/chat'
import { generateExperimentStream } from '@/api/experiment'
import { saveHistoryBatch } from '@/api/history'
import { isTimeoutError } from '@/api/client'
import { useSSE } from '@/composables/useSSE'
import { useLayoutStore } from '@/stores/layout'
import { useExperimentStore } from '@/stores/experiment'

export type ChatMode = 'normal' | 'extraction' | 'hardware' | 'experiment' | 'analysis'

export const MODE_PREFIX: Record<ChatMode, string> = {
  normal: '',
  extraction: '帮我搜寻：',
  hardware: '硬件控制：',
  experiment: '实验设计：',
  analysis: '数据分析',
}

export const MODE_LABEL: Record<ChatMode, string> = {
  normal: '',
  extraction: '📄 文献提取',
  hardware: '⚙️ 硬件控制',
  experiment: '🧪 实验设计',
  analysis: '📈 数据分析',
}

export interface PageReading {
  filename: string
  page: number
  image: string
}

export const useChatStore = defineStore('chat', () => {
  const messages = ref<Message[]>([])
  const isStreaming = ref(false)
  const abortController = ref<AbortController | null>(null)
  const currentMode = ref<ChatMode>('normal')
  const extractionRunning = ref(false)
  let extractionDisconnect: (() => void) | null = null

  // Two-round extraction: field_confirm → start_extraction
  const fieldConfirm = ref<{ task_desc: string; fields: string[] } | null>(null)

  // PDF page preview during extraction
  const currentPage = ref<PageReading | null>(null)
  const extractionPdfPath = ref<string | null>(null)
  const extractionFilename = ref<string | null>(null)

  const streamingMessage = computed(() =>
    isStreaming.value ? messages.value[messages.value.length - 1] : null
  )

  const isModeActive = computed(() => currentMode.value !== 'normal')

  function setMode(mode: ChatMode) {
    if (currentMode.value === mode) {
      currentMode.value = 'normal'
    } else {
      currentMode.value = mode
    }
  }

  function enableExtraction() { currentMode.value = 'extraction' }
  function disableExtraction() { currentMode.value = 'normal' }

  function addMessage(role: 'user' | 'ai', content: string) {
    messages.value.push({
      role,
      content,
      timestamp: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }),
    })
  }

  // 持久化当前消息列表到服务端会话文件夹
  async function persistHistory() {
    try {
      const payload = messages.value.map(m => ({
        role: m.role,
        content: m.content,
        timestamp: m.timestamp,
        mode: currentMode.value,
      }))
      await saveHistoryBatch(payload)
    } catch {
      // 静默失败，不干扰用户操作
    }
  }

  function connectExtractionSSE() {
    const layout = useLayoutStore()
    extractionRunning.value = true
    const { connect, disconnect } = useSSE('/api/task_stream', {
      onMessage(msg) {
        switch (msg.type) {
          case 'info':
          case 'progress':
            layout.updateTaskStatus('extraction', 'running')
            break
          case 'reading_start':
            currentPage.value = null
            break
          case 'reading_chunk':
            // LLM streaming for current page — shown inline
            break
          case 'page_reading': {
            const d = msg.data as { filename: string; pdf_path?: string; page: number; image: string }
            currentPage.value = { filename: d.filename, page: d.page, image: d.image }
            if (d.pdf_path && !extractionPdfPath.value) extractionPdfPath.value = d.pdf_path
            if (d.filename && !extractionFilename.value) extractionFilename.value = d.filename
            break
          }
          case 'finding': {
            const f = msg.data as { page: number; filename: string; details: Record<string, string> }
            let text = `🎯 新发现 (第${f.page}页 · ${f.filename})\n`
            for (const [k, v] of Object.entries(f.details)) {
              if (k !== '_source_doc') text += `  ${k}: ${v}\n`
            }
            addMessage('ai', text.trim())
            break
          }
          case 'complete': {
            extractionRunning.value = false
            extractionDisconnect = null
            currentPage.value = null
            extractionPdfPath.value = null
            extractionFilename.value = null
            layout.updateTaskStatus('extraction', 'completed')
            const d = msg.data as Record<string, unknown> | undefined
            if (d?.message) addMessage('ai', d.message as string)
            if (d?.error) addMessage('ai', d.error as string)
            persistHistory()
            disconnect()
            break
          }
          case 'error':
            extractionRunning.value = false
            extractionDisconnect = null
            currentPage.value = null
            extractionPdfPath.value = null
            extractionFilename.value = null
            addMessage('ai', msg.data as string)
            persistHistory()
            disconnect()
            break
        }
      },
      onError(err) {
        extractionDisconnect = null
        currentPage.value = null
        extractionRunning.value = false
        addMessage('ai', err.message)
        layout.updateTaskStatus('extraction', 'completed')
      },
    })
    extractionDisconnect = disconnect
    connect()
  }

  async function confirmExtraction() {
    const pending = fieldConfirm.value
    if (!pending) return

    fieldConfirm.value = null
    addMessage('user', '✅ 确认使用上述字段提取')

    const controller = new AbortController()
    abortController.value = controller

    try {
      const result = await sendChatMessage(
        {
          message: '确认提取',
          action: 'start_extraction',
          task_desc: pending.task_desc,
          fields: pending.fields,
        },
        () => {},
        controller.signal,
      )

      addMessage('ai', result.text)

      if (result.type === 'task_trigger') {
        connectExtractionSSE()
      }
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        addMessage('ai', (err as Error).message)
      }
    }
  }

  function cancelExtraction() {
    fieldConfirm.value = null
    currentMode.value = 'extraction'
  }

  function removeConfirmField(index: number) {
    if (fieldConfirm.value && index >= 0 && index < fieldConfirm.value.fields.length) {
      fieldConfirm.value.fields.splice(index, 1)
    }
  }

  function addConfirmField(name: string) {
    if (fieldConfirm.value && name.trim()) {
      fieldConfirm.value.fields.push(name.trim())
    }
  }

  function updateConfirmField(index: number, value: string) {
    if (fieldConfirm.value && index >= 0 && index < fieldConfirm.value.fields.length) {
      fieldConfirm.value.fields[index] = value
    }
  }

  async function send(text: string) {
    if (isStreaming.value) return

    const mode = currentMode.value
    const prefix = MODE_PREFIX[mode] || ''
    const finalText = prefix ? `${prefix}${text}` : text

    addMessage('user', finalText)
    addMessage('ai', '') // placeholder

    const aiMsg = messages.value[messages.value.length - 1]

    const controller = new AbortController()
    abortController.value = controller
    isStreaming.value = true

    try {
      const actionMap: Record<ChatMode, string> = {
        normal: 'chat',
        extraction: '',
        hardware: '',
        experiment: '',
        analysis: '',
      }

      const history = messages.value.slice(0, -2).map(m => ({ role: m.role, content: m.content }))

      let thinkingStartTime = 0

      const result = await sendChatMessage(
        {
          message: finalText,
          action: actionMap[mode] || 'chat',
          history,
        },
        {
          onThinkingChunk(text) {
            if (!thinkingStartTime) thinkingStartTime = Date.now()
            aiMsg.thinking = text
          },
          onThinkingComplete(text) {
            aiMsg.thinking = text
            aiMsg.thinking_duration = thinkingStartTime
              ? Math.round((Date.now() - thinkingStartTime) / 1000)
              : 0
          },
          onTextChunk(text) {
            aiMsg.content = text
          },
          onTextComplete(text) {
            aiMsg.content = text
          },
          onError(msg) {
            aiMsg.content = msg
          },
          onDone() {
            // stream ended normally
          },
        },
        controller.signal,
      )

      if (mode === 'extraction') {
        currentMode.value = 'normal'
        if (result.type === 'field_confirm' && result.task_desc && result.fields) {
          fieldConfirm.value = { task_desc: result.task_desc, fields: result.fields }
        } else if (result.type === 'task_trigger') {
          connectExtractionSSE()
        }
      } else if (mode === 'experiment') {
        currentMode.value = 'normal'
        if (result.type === 'experiment_design_mode') {
          const cmd = result.command || text
          try {
            aiMsg.content = '⏳ AI 正在分析实验需求...'
            let streamedText = ''
            let expThinkingStart = 0
            const expData = await generateExperimentStream(
              cmd,
              {
                onThinkingChunk(text) {
                  if (!expThinkingStart) expThinkingStart = Date.now()
                  aiMsg.thinking = text
                  aiMsg.thinking_duration = Math.round((Date.now() - expThinkingStart) / 100) / 10
                },
                onThinkingComplete(text) {
                  aiMsg.thinking = text
                  aiMsg.thinking_duration = expThinkingStart
                    ? Math.round((Date.now() - expThinkingStart) / 1000)
                    : 0
                },
                onChunk(chunk) {
                  streamedText += chunk
                  if (streamedText.length % 50 < chunk.length || streamedText.length < 50) {
                    aiMsg.content = '⏳ AI 正在生成实验方案...\n\n```json\n' + streamedText + '\n```'
                  }
                },
              },
              controller.signal,
            )
            if (expData.type === 'experiment_design') {
              const expStore = useExperimentStore()
              expStore.loadFromJSON(expData.experiment_json)
              useLayoutStore().updateTaskStatus('experiment', 'completed')
            }
            aiMsg.content = expData.reply || ''
          } catch (err: unknown) {
            const msg = isTimeoutError(err)
              ? '实验设计生成超时，请重试或简化需求描述'
              : (err as Error).message || '网络请求失败'
            aiMsg.content = msg
          }
        }
      } else if (mode !== 'normal') {
        currentMode.value = 'normal'
      }
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        aiMsg.content = (err as Error).message
      }
      if (mode !== 'normal') currentMode.value = 'normal'
    } finally {
      isStreaming.value = false
      abortController.value = null
      persistHistory()
    }
  }

  function stop() {
    abortController.value?.abort()
  }

  async function cancelExtractionTask() {
    try {
      await fetch('/api/cancel_task', { method: 'POST' })
    } catch {
      // silently fail
    }
    extractionDisconnect?.()
    extractionDisconnect = null
    extractionRunning.value = false
    currentPage.value = null
    extractionPdfPath.value = null
    extractionFilename.value = null
    addMessage('ai', '提取任务已取消。')
  }

  // 页面关闭/刷新时自动保存（sendBeacon 保证可靠发送）
  if (typeof window !== 'undefined') {
    window.addEventListener('beforeunload', () => {
      if (messages.value.length === 0) return
      const payload = messages.value.map(m => ({
        role: m.role,
        content: m.content,
        timestamp: m.timestamp,
        mode: currentMode.value,
      }))
      const blob = new Blob([JSON.stringify({ messages: payload })], { type: 'application/json' })
      navigator.sendBeacon('/api/history/save_batch', blob)
    })
  }

  function loadMessages(msgs: { role: string; content: string; timestamp?: string }[]) {
    messages.value = msgs.map(m => ({
      role: m.role as 'user' | 'ai',
      content: m.content,
      timestamp: m.timestamp,
    }))
  }

  function clear() {
    messages.value = []
    currentMode.value = 'normal'
    fieldConfirm.value = null
    currentPage.value = null
    extractionPdfPath.value = null
    extractionFilename.value = null
    extractionRunning.value = false
    extractionDisconnect?.()
    extractionDisconnect = null
  }

  return {
    messages,
    isStreaming,
    currentMode,
    isModeActive,
    extractionRunning,
    fieldConfirm,
    currentPage,
    extractionPdfPath,
    extractionFilename,
    streamingMessage,
    setMode,
    enableExtraction,
    disableExtraction,
    addMessage,
    send,
    stop,
    clear,
    cancelExtractionTask,
    confirmExtraction,
    cancelExtraction,
    removeConfirmField,
    addConfirmField,
    updateConfirmField,
    persistHistory,
    loadMessages,
  }
})
