import { request } from './client'

export interface SessionEntry {
  timestamp: string
  started_at: string
  saved_at: string
  message_count: number
  title: string | null
  path: string
}

export interface SessionsIndex {
  sessions: SessionEntry[]
}

export async function fetchSessions(): Promise<SessionsIndex> {
  return request<SessionsIndex>('/api/history/sessions')
}

export async function saveHistoryBatch(messages: unknown[]): Promise<{ success: boolean; saved_count: number }> {
  return request('/api/history/save_batch', {
    method: 'POST',
    body: { messages },
    timeout: 10000,
  })
}
