import { ref, onUnmounted } from 'vue'

interface SSEMessage {
  type: string
  data?: unknown
}

interface UseSSEOptions {
  onMessage: (msg: SSEMessage) => void
  onComplete?: (data: unknown) => void
  onError?: (err: Error) => void
}

export function useSSE(url: string, opts: UseSSEOptions) {
  const connected = ref(false)
  const controller = new AbortController()

  async function connect() {
    try {
      const resp = await fetch(url, { signal: controller.signal })
      connected.value = true
      const reader = resp.body?.getReader()
      if (!reader) return

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        let dataLine = ''
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            dataLine = line.slice(6)
          } else if (line === '') {
            if (dataLine) {
              try {
                const msg: SSEMessage = JSON.parse(dataLine)
                opts.onMessage(msg)
                if (msg.type === 'complete') {
                  opts.onComplete?.(msg.data)
                  return
                }
              } catch {
                // skip unparseable
              }
              dataLine = ''
            }
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        opts.onError?.(err as Error)
      }
    } finally {
      connected.value = false
    }
  }

  function disconnect() {
    controller.abort()
  }

  onUnmounted(disconnect)

  return { connected, connect, disconnect }
}
