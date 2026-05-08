import { ref } from 'vue'
import { defineStore } from 'pinia'
import { startExtraction, getExtractionFields, cancelTask } from '@/api/extraction'
import { useSSE } from '@/composables/useSSE'

export interface Finding {
  tag: string
  value: string
}

export interface PageReading {
  page_id: string
  page_num: number
  pdf_name: string
}

export const useExtractionStore = defineStore('extraction', () => {
  const taskDesc = ref('')
  const fields = ref<string[]>([])
  const taskId = ref('')
  const isRunning = ref(false)
  const findings = ref<Finding[]>([])
  const logMessages = ref<string[]>([])
  const currentPage = ref<PageReading | null>(null)
  const llmStream = ref('')
  const readingActive = ref(false)
  const fileCount = ref(0)
  const summary = ref<{ field_count: number; pdf_count: number; file: string } | null>(null)
  const error = ref('')

  function addLog(msg: string) {
    logMessages.value.push(msg)
  }

  async function requestFields(desc: string) {
    taskDesc.value = desc
    const resp = await getExtractionFields(desc)
    // LLM response contains field list, try to extract
    try {
      const json = JSON.parse(resp.reply)
      if (json.fields) fields.value = json.fields
    } catch {
      // fallback: use default fields
      fields.value = ['钝化剂名称', '原文原句', '作用机理', '文献来源']
    }
    return fields.value
  }

  function start(desc: string, flds: string[]) {
    taskDesc.value = desc
    fields.value = flds
    findings.value = []
    logMessages.value = []
    llmStream.value = ''
    currentPage.value = null
    readingActive.value = false
    summary.value = null
    error.value = ''
    isRunning.value = true

    startExtraction(desc, flds).then(resp => {
      addLog(resp.reply)
    })
  }

  function connectSSE() {
    const { connect, disconnect } = useSSE('/api/task_stream', {
      onMessage(msg) {
        switch (msg.type) {
          case 'info':
            addLog(msg.data as string)
            break
          case 'progress':
            addLog(msg.data as string)
            break
          case 'error':
            addLog(msg.data as string)
            error.value = msg.data as string
            break
          case 'page_reading':
            currentPage.value = msg.data as PageReading
            llmStream.value = ''
            readingActive.value = true
            break
          case 'finding':
            findings.value.push(msg.data as Finding)
            readingActive.value = false
            break
          case 'reading_start':
            llmStream.value = ''
            readingActive.value = true
            break
          case 'reading_chunk':
            llmStream.value += msg.data as string
            break
          case 'complete':
            isRunning.value = false
            if (msg.data) {
              const d = msg.data as Record<string, unknown>
              if (d.message) addLog(d.message as string)
              if (d.error) error.value = d.error as string
              summary.value = d.field_count !== undefined ? d as typeof summary.value : null
            }
            disconnect()
            break
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
    await cancelTask()
    isRunning.value = false
  }

  return {
    taskDesc, fields, taskId, isRunning,
    findings, logMessages, currentPage, llmStream, readingActive,
    fileCount, summary, error,
    requestFields, start, connectSSE, cancel, addLog,
  }
})
