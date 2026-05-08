import { request } from './client'

export async function uploadPDF(file: File): Promise<{ success: boolean; filename: string }> {
  const formData = new FormData()
  formData.append('file', file)
  const resp = await fetch('/api/upload', { method: 'POST', body: formData })
  return resp.json()
}

export async function startExtraction(taskDesc: string, fields: string[]): Promise<{ type: string; reply: string }> {
  return request('/api/chat', {
    method: 'POST',
    body: { message: '', action: 'start_extraction', task_desc: taskDesc, fields },
  })
}

export async function getExtractionFields(taskDesc: string): Promise<{ type: string; reply: string }> {
  return request('/api/chat', {
    method: 'POST',
    body: { message: `帮我搜寻：${taskDesc}` },
  })
}

export async function cancelTask(): Promise<{ status: string }> {
  return request('/api/cancel_task', { method: 'POST' })
}
