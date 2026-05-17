import i18n from '@/i18n'

const BASE = '' // All /api routes are on same origin or proxied by Vite

interface RequestOptions {
  method?: string
  body?: unknown
  signal?: AbortSignal
  timeout?: number
}

async function request<T = unknown>(url: string, opts: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, signal, timeout = 30000 } = opts

  const controller = new AbortController()
  const linked = signal
  if (linked) linked.addEventListener('abort', () => controller.abort(linked.reason))

  const timer = setTimeout(() => controller.abort(new DOMException('请求超时', 'TimeoutError')), timeout)

  try {
    const resp = await fetch(BASE + url, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    })

    if (!resp.ok) {
      const text = await resp.text().catch(() => '')
      throw new ApiError(resp.status, text || resp.statusText)
    }

    return (await resp.json()) as T
  } catch (err) {
    // 将内部超时 AbortError 转换为有意义的 ApiError
    if (controller.signal.aborted && controller.signal.reason instanceof DOMException
        && controller.signal.reason.name === 'TimeoutError') {
      throw new ApiError(408, i18n.global.t('api.requestTimeout'))
    }
    // 外部信号触发的 abort，透传原始错误
    if (linked?.aborted) {
      throw err
    }
    throw err
  } finally {
    clearTimeout(timer)
  }
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export function isTimeoutError(err: unknown): boolean {
  if (err instanceof ApiError && err.status === 408) return true
  if (err instanceof DOMException && err.name === 'TimeoutError') return true
  if (err instanceof DOMException && err.name === 'AbortError') return true
  return false
}

export { request }
