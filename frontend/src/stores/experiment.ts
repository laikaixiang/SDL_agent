import { ref } from 'vue'
import { defineStore } from 'pinia'
import type { ExperimentPlan, ExperimentStep } from '@/types/experiment'

export const useExperimentStore = defineStore('experiment', () => {
  const plan = ref<ExperimentPlan | null>(null)
  const codeViewMode = ref<'json' | 'python'>('json')
  const pythonCode = ref('')
  const loading = ref(false)
  const error = ref('')
  const logMessages = ref<string[]>([])

  function setPlan(name: string, steps: ExperimentStep[]) {
    plan.value = { experiment_name: name, steps }
  }

  async function sendDesignRequest(desc: string) {
    loading.value = true
    error.value = ''
    logMessages.value = []

    try {
      const resp = await fetch('/api/experiment_chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: desc, history: [] }),
      })
      const data = await resp.json()

      if (data.experiment_json) {
        let json = data.experiment_json
        if (typeof json === 'string') {
          json = JSON.parse(json)
        }
        plan.value = json
      } else if (data.type === 'system') {
        error.value = data.reply || '设计失败'
      }
    } catch (err) {
      error.value = (err as Error).message
    } finally {
      loading.value = false
    }
  }

  async function compile() {
    if (!plan.value) return
    try {
      const resp = await fetch('/api/compile_experiment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ experiment_json: plan.value }),
      })
      const data = await resp.json()
      pythonCode.value = data.python_code || ''
      codeViewMode.value = 'python'
    } catch {
      pythonCode.value = '# 编译失败'
    }
  }

  return {
    plan, codeViewMode, pythonCode, loading, error, logMessages,
    setPlan, sendDesignRequest, compile,
  }
})
