export interface ApiResponse<T = unknown> {
  success: boolean
  message?: string
  data?: T
}

export interface TaskStreamEvent {
  type: 'info' | 'progress' | 'result' | 'error' | 'done'
  message?: string
  data?: unknown
}
