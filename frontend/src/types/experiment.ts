export type StepType = 'tool' | 'software' | 'helper'
export type HelperType = 'LOOP' | 'GROUP' | 'WAIT' | 'CONDITION' | 'END' | 'USER_INPUT'

export interface ExperimentStep {
  type: StepType
  name: string
  params: Record<string, unknown>
  description?: string
  input_file?: string
  output_file?: string
}

export interface ExperimentPlan {
  experiment_name: string
  description?: string
  steps: ExperimentStep[]
  created_at?: string
  notes?: string
}
