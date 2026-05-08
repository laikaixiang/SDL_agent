import { request } from './client'
import type { HardwareTool } from '@/types/hardware'

export async function getHardwareTools(): Promise<HardwareTool[]> {
  const resp = await request<{ tools: HardwareTool[] }>('/api/hardware_tools')
  return resp.tools || []
}

interface ToolCall {
  name: string
  params: Record<string, unknown>
}

export async function sendHardwareCommand(cmd: string): Promise<{
  type: string
  tool_calls: ToolCall[]
  reply: string
  task_desc: string
}> {
  return request('/api/chat', {
    method: 'POST',
    body: { message: `硬件控制：${cmd}` },
  })
}

export async function executeHardware(toolCalls: ToolCall[]): Promise<{ type: string; reply: string }> {
  return request('/api/chat', {
    method: 'POST',
    body: { message: '', action: 'start_hardware', tool_calls: toolCalls },
  })
}

export async function cancelTask(): Promise<{ status: string }> {
  return request('/api/cancel_task', { method: 'POST' })
}
