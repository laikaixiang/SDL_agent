export interface Message {
  role: 'user' | 'ai'
  content: string
  thinking?: string
  thinking_duration?: number
  timestamp?: string
  // Agent-specific (per-turn) data — attached when agent run completes
  // so tool calls / team progress remain visible after the AI's final
  // reply (i.e. look like a normal conversation log).
  toolCalls?: ToolCallInfo[]
  pendingQuestion?: AgentQuestion
  // 系统消息: 项目总结 (ask_user 300s 超时时自动生成) / 压缩通知
  systemNote?: {
    kind: 'compaction' | 'timeout_summary' | 'info'
    text: string
  }
  teamAgents?: TeamAgentInfo[]
}

export interface AgentQuestionWithAnswer extends AgentQuestion {
  answer?: string
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
  // tool progress (extract_from_pdf 每页推送)
  current?: number
  total?: number
  // compaction (120s 触发)
  compacted_count?: number
  // timeout summary (300s 触发)
  timeout_sec?: number
  // keepalive timestamp
  timestamp?: number
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
  // 心跳 (每 5s 推送, 用于前端显示"agent 仍在工作")
  onKeepalive?: (timestamp: number) => void
  // 工具进度 (extract_from_pdf 每页推送)
  onToolProgress?: (name: string, current: number, total: number, message?: string) => void
  // 压缩 (120s)
  onCompactionStart?: (message: string) => void
  onCompactionComplete?: (compactedCount: number, message: string) => void
  onCompactionError?: (error: string) => void
  // 超时总结 (300s)
  onTimeoutSummary?: (summary: string, timeoutSec: number) => void
}

export interface AgentResult {
  text: string
  error?: string
}
