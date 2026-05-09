import type { Message } from '@/types/chat'

interface ChatRequest {
  message: string
  action?: string
  history?: Pick<Message, 'role' | 'content'>[]
  task_desc?: string
  fields?: string[]
}

interface JsonResponse {
  type: 'text' | 'system' | 'task_trigger' | 'field_confirm' | 'error'
  reply: string
  task_desc?: string
  fields?: string[]
}

interface ChatResult {
  text: string
  type?: string
  task_desc?: string
  fields?: string[]
}

/**
 * Send a chat message. If the backend returns JSON (system message, task trigger),
 * we parse it directly. If plain text, we stream it.
 */
export async function sendChatMessage(
  body: ChatRequest,
  onChunk: (text: string) => void,
  signal?: AbortSignal,
): Promise<ChatResult> {
  const resp = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })

  const contentType = resp.headers.get('Content-Type') || ''

  if (contentType.includes('application/json')) {
    const data: JsonResponse = await resp.json()
    onChunk(data.reply)
    return { text: data.reply, type: data.type, task_desc: data.task_desc, fields: data.fields }
  }

  // Streaming text response
  let fullText = ''
  const reader = resp.body?.getReader()
  if (!reader) return { text: '' }

  const decoder = new TextDecoder('utf-8')

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      const chunk = decoder.decode(value, { stream: true })
      fullText += chunk
      onChunk(chunk)
    }
  } catch (err) {
    if ((err as Error).name === 'AbortError') {
      fullText += '\n(已停止生成)'
      onChunk('\n(已停止生成)')
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
