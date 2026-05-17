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

export interface VariableConstraint {
  min?: number
  max?: number
  step?: number
  required?: boolean
  options?: string[]
}

export interface VariableDefinition {
  name: string
  type: 'int' | 'float' | 'str' | 'bool'
  default_value: number | string | boolean
  constraints?: VariableConstraint
  used_in_steps?: string[]
}

export interface ExperimentPlan {
  experiment_name: string
  description?: string
  steps: ExperimentStep[]
  created_at?: string
  notes?: string
  variables?: Record<string, VariableDefinition>
  batch_data?: Record<string, unknown>[]
  batch_mode?: boolean
}
