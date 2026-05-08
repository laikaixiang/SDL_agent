export interface ExperimentStep {
  type: 'tool' | 'helper' | 'software'
  name: string
  params: Record<string, unknown>
  description?: string
}

export interface ExperimentPlan {
  experiment_name: string
  steps: ExperimentStep[]
}
