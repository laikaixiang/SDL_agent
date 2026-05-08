export interface HardwareTool {
  name: string
  description: string
  params: Record<string, { type: string; description: string; required: boolean; default?: unknown }>
}
