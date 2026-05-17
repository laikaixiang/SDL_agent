import type { Message } from '@/types/chat'
import i18n from '@/i18n'

interface ChatRequest {
  message: string
  action?: string
  history?: Pick<Message, 'role' | 'content'>[]
  task_desc?: string
  fields?: string[]
}

interface JsonResponse {
  type: 'text' | 'system' | 'task_trigger' | 'field_confirm' | 'experiment_design_mode' | 'error'
  reply: string
  task_desc?: string
  fields?: string[]
  command?: string
}

interface ChatResult {
  text: string
  type?: string
  task_desc?: string
  fields?: string[]
  command?: string
}

interface StreamEvent {
  type: string
  text?: string
}

export interface StreamCallbacks {
  onTextChunk?: (text: string) => void
  onTextComplete?: (text: string) => void
  onThinkingChunk?: (text: string) => void
  onThinkingComplete?: (text: string) => void
  onError?: (message: string) => void
  onDone?: () => void
  /** @deprecated Legacy callback — use onTextChunk instead */
  onChunk?: (text: string) => void
}

/**
 * Send a chat message. Routes JSON responses (system/task triggers) directly,
 * SSE streams (text/event-stream) through typed callbacks.
 */
export async function sendChatMessage(
  body: ChatRequest,
  callbacks: StreamCallbacks | ((text: string) => void),
  signal?: AbortSignal,
): Promise<ChatResult> {
  // Normalize legacy onChunk callback
  const cb: StreamCallbacks =
    typeof callbacks === 'function' ? { onChunk: callbacks } : callbacks

  const resp = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })

  const contentType = resp.headers.get('Content-Type') || ''

  // --- JSON response (special actions) ---
  if (contentType.includes('application/json')) {
    const data: JsonResponse = await resp.json()
    cb.onTextChunk?.(data.reply)
    cb.onDone?.()
    return { text: data.reply, type: data.type, task_desc: data.task_desc, fields: data.fields, command: data.command }
  }

  // --- SSE stream ---
  let fullText = ''
  const reader = resp.body?.getReader()
  if (!reader) return { text: '' }

  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          const event: StreamEvent = JSON.parse(line.slice(6))

          switch (event.type) {
            case 'text_start':
              fullText = ''
              break
            case 'text_delta':
              fullText = event.text || ''
              cb.onTextChunk?.(fullText)
              cb.onChunk?.(fullText)
              break
            case 'text_end':
              fullText = event.text || fullText
              cb.onTextComplete?.(fullText)
              cb.onChunk?.(fullText)
              break
            case 'thinking_start':
              break
            case 'thinking_delta':
              cb.onThinkingChunk?.(event.text || '')
              break
            case 'thinking_end':
              cb.onThinkingComplete?.(event.text || '')
              break
            case 'error':
              cb.onError?.(event.text || '')
              break
            case 'done':
              cb.onDone?.()
              break
          }
        } catch {
          // skip malformed JSON
        }
      }
    }
  } catch (err) {
    if ((err as Error).name === 'AbortError') {
      fullText += '\n' + i18n.global.t('chat.generationStopped')
      cb.onTextChunk?.(fullText)
    } else {
      throw err
    }
  } finally {
    reader.releaseLock()
  }

  return { text: fullText }
}

export async function uploadPDF(file: File): Promise<{ success: boolean; filename: string }> {
  const formData = new FormData()
  formData.append('file', file)
  const resp = await fetch('/api/upload', { method: 'POST', body: formData })
  return resp.json()
}
