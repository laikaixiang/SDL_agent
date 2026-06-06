import { request } from './client'
import type { ExperimentPlan, VariableDefinition } from '@/types/experiment'
import i18n from '@/i18n'

export interface ImportCSVResult {
  type: string
  variables: Record<string, VariableDefinition>
  batch_data: Record<string, unknown>[]
  reply: string
}

export async function importCSV(csvContent: string): Promise<ImportCSVResult> {
  return request('/api/variables/import_csv', {
    method: 'POST',
    body: { csv_content: csvContent },
  })
}

export async function generateExperiment(desc: string): Promise<{
  type: string
  experiment_json: ExperimentPlan
  reply: string
}> {
  return request('/api/experiment_chat', {
    method: 'POST',
    body: { message: desc, history: [] },
    timeout: 240000,
  })
}

export interface StreamCallbacks {
  onChunk?: (text: string) => void
  onThinkingChunk?: (text: string) => void
  onThinkingComplete?: (text: string) => void
}

export async function generateExperimentStream(
  desc: string,
  callbacks: StreamCallbacks,
  signal?: AbortSignal,
): Promise<{
  type: string
  experiment_json: ExperimentPlan
  reply: string
}> {
  const resp = await fetch('/api/experiment_chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: desc, history: [], stream: true }),
    signal,
  })

  if (!resp.ok) {
    const text = await resp.text().catch(() => '')
    throw new Error(text || `HTTP ${resp.status}`)
  }

  const reader = resp.body?.getReader()
  if (!reader) throw new Error(i18n.global.t('api.cannotReadStream'))

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
          const msg = JSON.parse(line.slice(6))

          switch (msg.type) {
            case 'thinking_start':
              break
            case 'thinking_delta':
              callbacks.onThinkingChunk?.(msg.data as string)
              break
            case 'thinking_end':
              callbacks.onThinkingComplete?.(msg.data as string)
              break
            case 'chunk':
              callbacks.onChunk?.(msg.data as string)
              break
            case 'complete':
              return {
                type: 'experiment_design',
                experiment_json: msg.data.experiment_json,
                reply: msg.data.reply,
              }
            case 'error':
              throw new Error(msg.data as string)
          }
        } catch {
          // skip malformed JSON (same as chat)
        }
      }
    }
  } finally {
    reader.releaseLock()
  }

  throw new Error(i18n.global.t('api.streamEndedUnexpectedly'))
}

export async function compileExperiment(json: ExperimentPlan): Promise<{
  success: boolean
  code: string
  message?: string
}> {
  return request('/api/compile_experiment', {
    method: 'POST',
    body: { experiment_json: json },
  })
}

export async function logCompileError(error: string, experimentJson?: unknown): Promise<{
  success: boolean
  log_path?: string
  message?: string
}> {
  return request('/api/log_compile_error', {
    method: 'POST',
    body: { error, experiment_json: experimentJson },
  })
}

export async function compileAndRun(json: ExperimentPlan): Promise<{
  success: boolean
  code: string
  output: string
  error: string
}> {
  return request('/api/compile_and_run_experiment', {
    method: 'POST',
    body: { experiment_json: json },
    timeout: 300000,
  })
}

export async function executeExperiment(json: ExperimentPlan): Promise<{
  type: string
  reply: string
}> {
  return request('/api/execute_experiment_design', {
    method: 'POST',
    body: json,
  })
}

export async function saveExperiment(json: ExperimentPlan): Promise<{
  success: boolean
  filepath: string
  message: string
}> {
  return request('/api/save_experiment_design', {
    method: 'POST',
    body: json,
  })
}

export async function exportExperimentJSON(
  jsonData: ExperimentPlan,
  filepath: string,
): Promise<{ success: boolean; filepath: string; message: string }> {
  return request('/api/export_experiment_json', {
    method: 'POST',
    body: { json_data: jsonData, filepath },
  })
}
