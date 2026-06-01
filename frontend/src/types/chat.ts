export interface Message {
  role: 'user' | 'ai'
  content: string
  thinking?: string
  thinking_duration?: number
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

// Agent types
export interface ToolCallInfo {
  index: number
  name: string
  callId: string
  arguments: Record<string, unknown>
  result?: string
  status: 'running' | 'done' | 'error'
}

export interface AgentQuestion {
  question: string
  options?: string
}

export interface AgentEvent {
  type: string
  text?: string
  index?: number
  name?: string
  call_id?: string
  arguments?: Record<string, unknown>
  delta?: string
  result?: string
  status?: string
  question?: string
  options?: string
  message?: string
  mode?: string
  agents?: Array<{ id: string; template: string; task: string }>
  results?: unknown[]
  agent_id?: string
  summary?: string
}

export interface TeamAgentInfo {
  id: string
  template: string
  task: string
  status: 'spawning' | 'running' | 'done' | 'error'
  summary?: string
}

export interface AgentCallbacks {
  onTextChunk?: (t: string) => void
  onTextComplete?: (t: string) => void
  onThinkingChunk?: (t: string) => void
  onThinkingComplete?: (t: string) => void
  onToolCallStart?: (i: number, n: string, cid: string) => void
  onToolCallArgs?: (i: number, d: string) => void
  onToolCallEnd?: (i: number, n: string, a: Record<string, unknown>) => void
  onToolResult?: (i: number, n: string, r: string, s: string) => void
  onAgentQuestion?: (q: string, o?: string) => void
  onTeamSpawn?: (mode: string, agents: TeamAgentInfo[]) => void
  onTeamProgress?: (agentId: string, status: string, summary?: string) => void
  onTeamDone?: (mode: string, results: unknown[]) => void
  onError?: (m: string) => void
  onDone?: () => void
}

export interface AgentResult {
  text: string
  error?: string
}
