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

    // Connect to SSE for execution progress
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
    loadTools, sendCommand, execute, cancel, dismissConfirm, addLog,
  }
})
