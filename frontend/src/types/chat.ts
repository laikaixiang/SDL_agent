export interface Message {
  role: 'user' | 'ai'
  content: string
  timestamp?: string
}

export interface StreamChunk {
  type: 'text' | 'thinking' | 'tool_call' | 'error' | 'done'
  content?: string
  toolName?: string
  toolResult?: string
}

export interface ChatRequest {
  message: string
  mode: 'normal' | 'extraction' | 'hardware' | 'experiment' | 'analysis'
  history?: Message[]
}
