import { ref, computed } from 'vue'
import { defineStore } from 'pinia'
import type { ExperimentStep, ExperimentPlan, HelperType } from '@/types/experiment'
import type { HardwareTool } from '@/types/hardware'
import {
  generateExperiment,
  compileExperiment,
  compileAndRun,
  executeExperiment,
  saveExperiment,
} from '@/api/experiment'
import { getHardwareTools } from '@/api/hardware'
import { useSSE } from '@/composables/useSSE'
import { useLayoutStore } from '@/stores/layout'

interface AlgorithmDef {
  name: string
  chinese_name: string
  description: string
  params_schema?: Record<string, string>
}

const HELPER_TEMPLATES: Record<HelperType, Omit<ExperimentStep, 'description'>> = {
  LOOP: { type: 'helper', name: 'LOOP', params: { iterations: 3 } },
  GROUP: { type: 'helper', name: 'GROUP', params: { name: '步骤组' } },
  WAIT: { type: 'helper', name: 'WAIT', params: { duration: 5000 } },
  CONDITION: { type: 'helper', name: 'CONDITION', params: { condition: 'temperature > 100' } },
  END: { type: 'helper', name: 'END', params: {} },
  USER_INPUT: { type: 'helper', name: 'USER_INPUT', params: { prompt: '请输入参数', variable_name: 'user_value' } },
}

const HELPER_LABELS: Record<HelperType, string> = {
  LOOP: '循环执行',
  GROUP: '步骤组',
  WAIT: '等待',
  CONDITION: '条件判断',
  END: '结束标记',
  USER_INPUT: '用户输入',
}

export const useExperimentStore = defineStore('experiment', () => {
  const experimentName = ref('未命名实验')
  const steps = ref<ExperimentStep[]>([])
  const codeViewMode = ref<'json' | 'python'>('json')
  const pythonCode = ref('')
  const editingStepIndex = ref<number | null>(null)
  const draggedStepIndex = ref<number | null>(null)
  const codeAreaMinimized = ref(false)
  const codeAreaFullscreen = ref(false)
  const hardwareTools = ref<HardwareTool[]>([])
  const algorithms = ref<AlgorithmDef[]>([])
  const loading = ref(false)
  const running = ref(false)
  const error = ref('')
  const logMessages = ref<string[]>([])

  const plan = computed<ExperimentPlan>(() => ({
    experiment_name: experimentName.value,
    steps: steps.value,
    created_at: new Date().toISOString(),
  }))

  const jsonCode = computed(() => JSON.stringify(plan.value, null, 2))

  function addLog(msg: string) {
    logMessages.value.push(msg)
  }

  // --- Step CRUD ---

  function addStep(step: ExperimentStep) {
    steps.value.push({ ...step })
  }

  function removeStep(index: number) {
    steps.value.splice(index, 1)
    if (editingStepIndex.value === index) {
      editingStepIndex.value = null
    } else if (editingStepIndex.value !== null && editingStepIndex.value > index) {
      editingStepIndex.value--
    }
  }

  function moveStepUp(index: number) {
    if (index <= 0) return
    ;[steps.value[index - 1], steps.value[index]] = [steps.value[index], steps.value[index - 1]]
  }

  function moveStepDown(index: number) {
    if (index >= steps.value.length - 1) return
    ;[steps.value[index + 1], steps.value[index]] = [steps.value[index], steps.value[index + 1]]
  }

  function moveStep(fromIdx: number, toIdx: number) {
    const [item] = steps.value.splice(fromIdx, 1)
    steps.value.splice(toIdx, 0, item)
  }

  function updateStep(index: number, step: ExperimentStep) {
    steps.value[index] = { ...step }
    editingStepIndex.value = null
  }

  function toggleEdit(index: number) {
    editingStepIndex.value = editingStepIndex.value === index ? null : index
  }

  // --- Element adders ---

  function addToolStep(tool: HardwareTool, params: Record<string, unknown> = {}) {
    const initParams: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(tool.params || {})) {
      initParams[k] = params[k] ?? v.default
    }
    addStep({
      type: 'tool',
      name: tool.name,
      description: tool.description,
      params: initParams,
    })
  }

  function addAlgorithmStep(algo: AlgorithmDef) {
    addStep({
      type: 'software',
      name: algo.name,
      description: algo.chinese_name || algo.description,
      params: {},
      input_file: '',
      output_file: '',
    })
  }

  function addHelperFunction(fnType: HelperType) {
    const t = HELPER_TEMPLATES[fnType]
    addStep({ ...t, description: HELPER_LABELS[fnType], params: { ...t.params } })
  }

  // --- Load helpers ---

  async function loadHardwareTools() {
    try {
      hardwareTools.value = await getHardwareTools()
    } catch { /* offline */ }
  }

  async function loadAlgorithms() {
    try {
      const resp = await fetch('/api/list_algorithms')
      const data = await resp.json()
      algorithms.value = data.algorithms || []
    } catch { /* offline */ }
  }

  // --- AI generation ---

  async function generateFromAI(desc: string) {
    loading.value = true
    error.value = ''

    try {
      const data = await generateExperiment(desc)
      if (data.experiment_json) {
        let json = data.experiment_json
        if (typeof json === 'string') {
          json = JSON.parse(json) as ExperimentPlan
        }
        loadFromJSON(json)
        useLayoutStore().updateTaskStatus('experiment', 'completed')
      } else if ((data as unknown as { type: string }).type === 'error') {
        error.value = data.reply || 'AI 设计失败'
      }
    } catch (err) {
      error.value = (err as Error).message
    } finally {
      loading.value = false
    }
  }

  // --- Code ---

  async function compile() {
    if (!steps.value.length) return
    try {
      const data = await compileExperiment(plan.value)
      pythonCode.value = data.code || ''
      codeViewMode.value = 'python'
    } catch {
      pythonCode.value = '# 编译失败'
    }
  }

  async function compileAndRunExperiment() {
    if (!steps.value.length) return
    loading.value = true
    try {
      const data = await compileAndRun(plan.value)
      pythonCode.value = data.code || ''
      codeViewMode.value = 'python'
      if (data.output) {
        addLog('--- 执行输出 ---')
        addLog(data.output)
      }
      if (data.error) {
        addLog('--- 错误 ---')
        addLog(data.error)
      }
    } catch (err) {
      addLog('执行失败: ' + (err as Error).message)
    } finally {
      loading.value = false
    }
  }

  function syncFromCode() {
    // Implemented in CodeArea component via direct JSON editing
  }

  // --- Save / Load / Import / Clear ---

  async function save() {
    try {
      // Server backup
      await saveExperiment(plan.value)

      // Local save via File System Access API
      const blob = new Blob([jsonCode.value], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${experimentName.value}.json`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      addLog('保存失败: ' + (err as Error).message)
    }
  }

  function loadFromJSON(json: ExperimentPlan) {
    experimentName.value = json.experiment_name || '未命名实验'
    steps.value = (json.steps || []).map((s) => ({
      ...s,
      type: s.type || 'tool',
      params: s.params || {},
    }))
    editingStepIndex.value = null
    addLog(`已加载实验: ${experimentName.value}，共 ${steps.value.length} 步`)
  }

  async function importFile(file: File) {
    try {
      const text = await file.text()
      const json = JSON.parse(text) as ExperimentPlan
      loadFromJSON(json)
    } catch (err) {
      addLog('导入失败: ' + (err as Error).message)
    }
  }

  function clear() {
    experimentName.value = '未命名实验'
    steps.value = []
    pythonCode.value = ''
    logMessages.value = []
    editingStepIndex.value = null
    codeViewMode.value = 'json'
  }

  // --- Execute ---

  async function execute() {
    if (!steps.value.length) return
    running.value = true
    logMessages.value = ['开始执行实验...']

    try {
      const resp = await executeExperiment(plan.value)
      if (resp.type === 'task_trigger') {
        addLog(resp.reply)
        const { connect } = useSSE('/api/task_stream', {
          onMessage(msg) {
            switch (msg.type) {
              case 'info':
              case 'progress':
                addLog(msg.data as string)
                break
              case 'error':
                addLog(msg.data as string)
                error.value = msg.data as string
                break
              case 'complete':
                running.value = false
                const d = msg.data as Record<string, unknown> | undefined
                if (d?.message) addLog(d.message as string)
                if (d?.error) {
                  error.value = d.error as string
                  addLog(d.error as string)
                }
                useLayoutStore().updateTaskStatus('experiment', 'completed')
                break
            }
          },
          onError(err) {
            error.value = err.message
            running.value = false
          },
        })
        connect()
      }
    } catch (err) {
      addLog('执行失败: ' + (err as Error).message)
      running.value = false
    }
  }

  return {
    experimentName, steps, codeViewMode, pythonCode, editingStepIndex,
    draggedStepIndex, codeAreaMinimized, codeAreaFullscreen,
    hardwareTools, algorithms, loading, running, error, logMessages,
    plan, jsonCode,
    addStep, removeStep, moveStepUp, moveStepDown, moveStep, updateStep, toggleEdit,
    addToolStep, addAlgorithmStep, addHelperFunction,
    loadHardwareTools, loadAlgorithms,
    generateFromAI, compile, compileAndRunExperiment, syncFromCode,
    save, loadFromJSON, importFile, clear, execute, addLog,
  }
})
