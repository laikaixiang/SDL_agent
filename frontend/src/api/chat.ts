import type { Message } from '@/types/chat'

interface ChatRequest {
  message: string
  action?: string
  history?: Pick<Message, 'role' | 'content'>[]
}

interface JsonResponse {
  type: 'text' | 'system' | 'task_trigger' | 'error'
  reply: string
}

/**
 * Send a chat message. If the backend returns JSON (system message, task trigger),
 * we parse it directly. If plain text, we stream it.
 */
export async function sendChatMessage(
  body: ChatRequest,
  onChunk: (text: string) => void,
  signal?: AbortSignal,
): Promise<string> {
  const resp = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })

  const contentType = resp.headers.get('Content-Type') || ''

  if (contentType.includes('application/json')) {
    const data: JsonResponse = await resp.json()
    if (data.type === 'task_trigger') {
      onChunk(data.reply)
      return data.reply
    }
    onChunk(data.reply)
    return data.reply
  }

  // Streaming text response
  let fullText = ''
  const reader = resp.body?.getReader()
  if (!reader) return fullText

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

  return fullText
}
