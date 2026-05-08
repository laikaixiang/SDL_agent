import { request } from './client'
import type { ExperimentPlan } from '@/types/experiment'

export async function generateExperiment(desc: string): Promise<{
  type: string
  experiment_json: ExperimentPlan
  reply: string
}> {
  return request('/api/experiment_chat', {
    method: 'POST',
    body: { message: desc, history: [] },
    timeout: 120000,
  })
}

export async function compileExperiment(json: ExperimentPlan): Promise<{
  success: boolean
  code: string
}> {
  return request('/api/compile_experiment', {
    method: 'POST',
    body: { experiment_json: json },
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
