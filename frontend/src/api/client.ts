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
  if (linked) linked.addEventListener('abort', () => controller.abort())

  const timer = setTimeout(() => controller.abort(), timeout)

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
  } finally {
    clearTimeout(timer)
  }
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

export { request }
