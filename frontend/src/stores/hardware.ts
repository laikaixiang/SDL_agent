import { ref } from 'vue'
import { defineStore } from 'pinia'
import type { HardwareTool } from '@/types/hardware'
import { getHardwareTools, sendHardwareCommand, executeHardware, cancelTask as apiCancel } from '@/api/hardware'
import { useSSE } from '@/composables/useSSE'

interface ToolCall {
  name: string
  params: Record<string, unknown>
}

export const useHardwareStore = defineStore('hardware', () => {
  const tools = ref<HardwareTool[]>([])
  const isRunning = ref(false)
  const pendingCalls = ref<ToolCall[]>([])
  const logMessages = ref<string[]>([])
  const confirmMessage = ref('')
  const error = ref('')
  const expandedTool = ref<string | null>(null)
  const toolParams = ref<Record<string, string>>({})
  const statusMessage = ref('')

  async function loadTools() {
    try {
      tools.value = await getHardwareTools()
    } catch {
      // tools unavailable
    }
  }

  function addLog(msg: string) {
    logMessages.value.push(msg)
  }

  function toggleExpand(toolName: string) {
    expandedTool.value = expandedTool.value === toolName ? null : toolName
  }

  function setToolParam(toolName: string, paramName: string, value: string) {
    toolParams.value[`${toolName}.${paramName}`] = value
  }

  function getToolParam(toolName: string, paramName: string): string {
    return toolParams.value[`${toolName}.${paramName}`] ?? ''
  }

  function collectToolParams(toolName: string): Record<string, unknown> {
    const tool = tools.value.find(t => t.name === toolName)
    if (!tool) return {}

    const params: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(tool.params)) {
      const raw = toolParams.value[`${toolName}.${k}`]
      const val = raw !== undefined ? raw : (v.default !== undefined ? String(v.default) : '')
      if (v.type === 'int' || v.type === 'number' || v.type === 'float') {
        params[k] = Number(val)
      } else {
        params[k] = val
      }
    }
    return params
  }

  async function runSingleTool(toolName: string) {
    const tool = tools.value.find(t => t.name === toolName)
    if (!tool || isRunning.value) return

    // Validate required params
    const params = collectToolParams(toolName)
    for (const [k, v] of Object.entries(tool.params)) {
      if (v.required && !params[k] && params[k] !== 0) {
        addLog(`缺少必填参数: ${k}`)
        setStatus(`缺少必填参数: ${k}`, true)
        return
      }
    }

    isRunning.value = true
    logMessages.value = []
    addLog(`执行: ${tool.description || tool.name}`)
    setStatus('执行中...', true)

    try {
      const resp = await executeHardware([{ name: toolName, params }])

      if (resp.type === 'task_trigger') {
        addLog(resp.reply || '任务已触发')
        const { connect } = useSSE('/api/task_stream', {
          onMessage(msg) {
            if (msg.type === 'info' || msg.type === 'progress') {
              addLog(msg.data as string)
            } else if (msg.type === 'error') {
              addLog(msg.data as string)
              error.value = msg.data as string
            } else if (msg.type === 'complete') {
              isRunning.value = false
              const d = msg.data as Record<string, unknown> | undefined
              if (d?.message) {
                addLog(d.message as string)
                setStatus('完成: ' + (d.message as string), false)
              }
              if (d?.error) {
                error.value = d.error as string
                addLog(d.error as string)
                setStatus('失败: ' + (d.error as string), false)
              }
            }
          },
          onError(err) {
            error.value = err.message
            isRunning.value = false
            setStatus('错误: ' + err.message, false)
          },
        })
        connect()
      } else {
        isRunning.value = false
        setStatus('完成', false)
      }
    } catch (err) {
      addLog('执行失败: ' + (err as Error).message)
      setStatus('失败: ' + (err as Error).message, false)
      isRunning.value = false
    }
  }

  function setStatus(msg: string, running: boolean) {
    statusMessage.value = msg
  }

  async function sendCommand(cmd: string) {
    error.value = ''
    try {
      const resp = await sendHardwareCommand(cmd)
      if (resp.type === 'hardware_confirm') {
        confirmMessage.value = resp.reply
        pendingCalls.value = resp.tool_calls || []
      } else if (resp.type === 'system') {
        addLog(resp.reply)
      } else if (resp.type === 'experiment_design_mode') {
        addLog(resp.reply)
      }
    } catch (err) {
      error.value = (err as Error).message
    }
  }

  async function execute() {
    if (!pendingCalls.value.length) return
    isRunning.value = true
    logMessages.value = []

    await executeHardware(pendingCalls.value)
    pendingCalls.value = []

    const { connect } = useSSE('/api/task_stream', {
      onMessage(msg) {
        if (msg.type === 'info' || msg.type === 'progress') {
          addLog(msg.data as string)
        } else if (msg.type === 'error') {
          addLog(msg.data as string)
          error.value = msg.data as string
        } else if (msg.type === 'complete') {
          isRunning.value = false
          const d = msg.data as Record<string, unknown> | undefined
          if (d?.message) addLog(d.message as string)
          if (d?.error) { error.value = d.error as string; addLog(d.error as string) }
        }
      },
      onError(err) {
        error.value = err.message
        isRunning.value = false
      },
    })
    connect()
  }

  async function cancel() {
    await apiCancel()
    isRunning.value = false
  }

  function dismissConfirm() {
    confirmMessage.value = ''
    pendingCalls.value = []
  }

  return {
    tools, isRunning, pendingCalls, logMessages, confirmMessage, error,
    expandedTool, toolParams, statusMessage,
    loadTools, sendCommand, execute, cancel, dismissConfirm, addLog,
    toggleExpand, setToolParam, getToolParam, collectToolParams, runSingleTool, setStatus,
  }
})
