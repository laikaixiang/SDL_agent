import { request } from './client'

export interface SessionEntry {
  timestamp: string
  started_at: string
  saved_at: string
  message_count: number
  title: string | null
  path: string
  folder_id?: string
}

export interface SessionsIndex {
  sessions: SessionEntry[]
}

export interface SessionData {
  title: string
  messages: { role: string; content: string; timestamp?: string; mode?: string }[]
  outputs: Record<string, string[]>
}

export interface Folder {
  id: string
  name: string
  created_at: string
}

// ── Sessions ──

export async function fetchSessions(): Promise<SessionsIndex> {
  return request<SessionsIndex>('/api/history/sessions')
}

export async function fetchSession(timestamp: string): Promise<{ success: boolean; data: SessionData }> {
  return request(`/api/history/session/${timestamp}`)
}

export async function saveHistoryBatch(messages: unknown[]): Promise<{ success: boolean; saved_count: number }> {
  return request('/api/history/save_batch', {
    method: 'POST',
    body: { messages },
    timeout: 10000,
  })
}

export async function deleteSession(timestamp: string): Promise<{ success: boolean }> {
  return request(`/api/history/session/${timestamp}`, { method: 'DELETE' })
}

export async function updateSessionTitle(timestamp: string, title: string): Promise<{ success: boolean; title: string }> {
  return request(`/api/history/session/${timestamp}/title`, {
    method: 'PUT',
    body: { title },
  })
}

export async function moveSession(timestamp: string, folder_id: string | null): Promise<{ success: boolean }> {
  return request(`/api/history/session/${timestamp}/move`, {
    method: 'PUT',
    body: { folder_id },
  })
}

// ── Folders ──

export async function fetchFolders(): Promise<{ success: boolean; folders: Folder[] }> {
  return request('/api/history/folders')
}

export async function createFolder(name: string): Promise<{ success: boolean; folder: Folder }> {
  return request('/api/history/folders', {
    method: 'POST',
    body: { name },
  })
}

export async function renameFolder(id: string, name: string): Promise<{ success: boolean; folder: Folder }> {
  return request(`/api/history/folders/${id}`, {
    method: 'PUT',
    body: { name },
  })
}

export async function deleteFolder(id: string): Promise<{ success: boolean }> {
  return request(`/api/history/folders/${id}`, { method: 'DELETE' })
}
