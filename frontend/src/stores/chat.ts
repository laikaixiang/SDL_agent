import { ref, computed } from 'vue'
import { defineStore } from 'pinia'
import type { Message } from '@/types/chat'
import { sendChatMessage } from '@/api/chat'

export type ChatMode = 'normal' | 'extraction' | 'hardware' | 'experiment' | 'analysis'

export const useChatStore = defineStore('chat', () => {
  const messages = ref<Message[]>([])
  const isStreaming = ref(false)
  const abortController = ref<AbortController | null>(null)
  const currentMode = ref<ChatMode>('normal')

  const streamingMessage = computed(() =>
    isStreaming.value ? messages.value[messages.value.length - 1] : null
  )

  function setMode(mode: ChatMode) {
    currentMode.value = mode
  }

  function addMessage(role: 'user' | 'ai', content: string) {
    messages.value.push({
      role,
      content,
      timestamp: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }),
    })
  }

  async function send(text: string) {
    if (isStreaming.value) return

    addMessage('user', text)
    addMessage('ai', '') // placeholder that will be filled by streaming

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

      await sendChatMessage(
        {
          message: text,
          action: actionMap[currentMode.value] || 'chat',
          history: messages.value.slice(0, -1).map(m => ({ role: m.role, content: m.content })),
        },
        (chunk) => {
          aiMsg.content += chunk
        },
        controller.signal,
      )
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        aiMsg.content = `错误: ${(err as Error).message}`
      }
    } finally {
      isStreaming.value = false
      abortController.value = null
    }
  }

  function stop() {
    abortController.value?.abort()
  }

  function clear() {
    messages.value = []
  }

  return {
    messages,
    isStreaming,
    currentMode,
    streamingMessage,
    setMode,
    send,
    stop,
    clear,
  }
})
