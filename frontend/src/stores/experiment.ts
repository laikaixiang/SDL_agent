import { ref, computed, watch } from 'vue'
import { defineStore } from 'pinia'
import type { ExperimentStep, ExperimentPlan, HelperType } from '@/types/experiment'
import type { HardwareTool } from '@/types/hardware'
import {
  generateExperimentStream,
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
  const editableJsonCode = ref('')
  const editablePythonCode = ref('')
  const compileStatus = ref<'idle' | 'compiling' | 'error'>('idle')
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
  const thinking = ref('')
  const thinkingDuration = ref(0)
  let thinkingTimer: ReturnType<typeof setInterval> | null = null

  const plan = computed<ExperimentPlan>(() => ({
    experiment_name: experimentName.value,
    steps: steps.value,
    created_at: new Date().toISOString(),
  }))

  const jsonCode = computed(() => JSON.stringify(plan.value, null, 2))

  // Keep editableJsonCode in sync with computed jsonCode (only when not focused)
  let jsonFocusCount = 0
  watch(jsonCode, (val) => {
    if (jsonFocusCount === 0) {
      editableJsonCode.value = val
    }
  }, { immediate: true })

  // Keep editablePythonCode in sync with pythonCode
  let pyFocusCount = 0
  watch(pythonCode, (val) => {
    if (pyFocusCount === 0) {
      editablePythonCode.value = val || '# 自动生成的实验执行代码'
    }
  }, { immediate: true })

  // Debounced auto-compile: watch jsonCode and compile after 600ms of no changes
  let compileTimer: ReturnType<typeof setTimeout> | null = null
  watch(jsonCode, () => {
    if (!steps.value.length) return
    if (compileTimer) clearTimeout(compileTimer)
    compileTimer = setTimeout(async () => {
      compileStatus.value = 'compiling'
      try {
        const data = await compileExperiment(plan.value)
        pythonCode.value = data.code || ''
        compileStatus.value = 'idle'
      } catch {
        pythonCode.value = '# 编译失败'
        compileStatus.value = 'error'
      }
    }, 600)
  })

  // Focus tracking helpers for editability
  function onJsonFocus() { jsonFocusCount++ }
  function onJsonBlur() { jsonFocusCount = Math.max(0, jsonFocusCount - 1) }
  function onPyFocus() { pyFocusCount++ }
  function onPyBlur() { pyFocusCount = Math.max(0, pyFocusCount - 1) }

  // --- Nesting & block structure ---

  interface StepNestingInfo {
    level: number
    isBlockStart: boolean
    isBlockEnd: boolean
    blockType?: 'LOOP' | 'GROUP' | 'CONDITION'
    guideLines: number[]
  }

  const BLOCK_OPENERS = new Set(['LOOP', 'GROUP', 'CONDITION'])

  const nestingInfo = computed<StepNestingInfo[]>(() => {
    const result: StepNestingInfo[] = []
    const stack: string[] = []

    for (const step of steps.value) {
      const currentLevel = stack.length
      const isHelper = step.type === 'helper'
      const isBlockStart = isHelper && BLOCK_OPENERS.has(step.name)
      const isBlockEnd = isHelper && step.name === 'END'

      if (isBlockStart) {
        result.push({
          level: currentLevel,
          isBlockStart: true,
          isBlockEnd: false,
          blockType: step.name as 'LOOP' | 'GROUP' | 'CONDITION',
          guideLines: Array.from({ length: currentLevel }, (_, j) => j),
        })
        stack.push(step.name)
      } else if (isBlockEnd) {
        if (stack.length > 0) {
          stack.pop()
          result.push({
            level: stack.length,
            isBlockStart: false,
            isBlockEnd: true,
            guideLines: Array.from({ length: stack.length }, (_, j) => j),
          })
        } else {
          result.push({
            level: 0,
            isBlockStart: false,
            isBlockEnd: true,
            guideLines: [],
          })
        }
      } else {
        result.push({
          level: currentLevel,
          isBlockStart: false,
          isBlockEnd: false,
          guideLines: Array.from({ length: currentLevel }, (_, j) => j),
        })
      }
    }

    return result
  })

  const blockErrors = computed(() => {
    const stack: string[] = []
    let orphanedEnd: number | null = null

    for (let i = 0; i < steps.value.length; i++) {
      const step = steps.value[i]
      if (step.type === 'helper' && BLOCK_OPENERS.has(step.name)) {
        stack.push(step.name)
      } else if (step.type === 'helper' && step.name === 'END') {
        if (stack.length > 0) {
          stack.pop()
        } else if (orphanedEnd === null) {
          orphanedEnd = i
        }
      }
    }

    return {
      unclosed: stack.length > 0 ? stack[stack.length - 1] : null,
      lastIndex: stack.length > 0 ? steps.value.length - 1 : null,
      orphanedEnd,
    }
  })

  // --- Collapse state ---

  const collapsedBlocks = ref<Set<number>>(new Set())

  function toggleCollapse(index: number) {
    const s = new Set(collapsedBlocks.value)
    if (s.has(index)) {
      s.delete(index)
    } else {
      s.add(index)
    }
    collapsedBlocks.value = s
  }

  const hiddenStepIndices = computed(() => {
    const hidden = new Set<number>()
    for (const startIdx of collapsedBlocks.value) {
      const step = steps.value[startIdx]
      if (!step || step.type !== 'helper' || !BLOCK_OPENERS.has(step.name)) {
        continue
      }
      let depth = 0
      for (let i = startIdx + 1; i < steps.value.length; i++) {
        const s = steps.value[i]
        if (s.type === 'helper' && BLOCK_OPENERS.has(s.name)) {
          depth++
        } else if (s.type === 'helper' && s.name === 'END') {
          if (depth === 0) {
            for (let j = startIdx + 1; j <= i; j++) {
              hidden.add(j)
            }
            break
          }
          depth--
        }
      }
    }
    return hidden
  })

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

  function _clearThinkingTimer() {
    if (thinkingTimer) {
      clearInterval(thinkingTimer)
      thinkingTimer = null
    }
  }

  function _startThinkingTimer() {
    _clearThinkingTimer()
    const startTime = Date.now()
    thinkingDuration.value = 0
    thinkingTimer = setInterval(() => {
      thinkingDuration.value = Math.round((Date.now() - startTime) / 10) / 100
    }, 100)
  }

  async function generateFromAI(desc: string) {
    loading.value = true
    error.value = ''
    thinking.value = ''
    thinkingDuration.value = 0
    let thinkingDone = false

    try {
      const data = await generateExperimentStream(desc, {
        onThinkingChunk(text) {
          console.log('[exp] thinking chunk:', text.length, 'chars')
          thinking.value = text
          if (!thinkingDone && !thinkingTimer) {
            _startThinkingTimer()
          }
        },
        onThinkingComplete(text) {
          console.log('[exp] thinking complete:', text.length, 'chars')
          thinking.value = text
          thinkingDone = true
          _clearThinkingTimer()
        },
        onChunk(text) {
          console.log('[exp] content chunk:', text.length, 'chars')
          if (!thinkingDone && !thinking.value) {
            // No thinking phase — show content as it streams
            if (!thinkingTimer) _startThinkingTimer()
            thinking.value = '正在直接生成实验方案...'
          }
        },
      })
      console.log('[exp] stream complete, got json:', !!data.experiment_json)
      if (data.experiment_json) {
        let json = data.experiment_json
        if (typeof json === 'string') {
          json = JSON.parse(json) as ExperimentPlan
        }
        loadFromJSON(json)
        useLayoutStore().updateTaskStatus('experiment', 'completed')
      }
    } catch (err) {
      console.error('[exp] generateFromAI error:', err)
      if ((err as Error).message) {
        error.value = (err as Error).message
      }
    } finally {
      _clearThinkingTimer()
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
      if (data.output) addLog(data.output)
      if (data.error) addLog(data.error)
    } catch (err) {
      addLog((err as Error).message)
    } finally {
      loading.value = false
    }
  }

  function syncFromCode() {
    try {
      const json = JSON.parse(editableJsonCode.value) as ExperimentPlan
      loadFromJSON(json)
      addLog('已从 JSON 同步实验步骤')
    } catch (err) {
      addLog(`JSON 解析失败: ${(err as Error).message}`)
    }
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
      addLog((err as Error).message)
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
      addLog((err as Error).message)
    }
  }

  function clear() {
    experimentName.value = '未命名实验'
    steps.value = []
    pythonCode.value = ''
    logMessages.value = []
    editingStepIndex.value = null
    codeViewMode.value = 'json'
    collapsedBlocks.value = new Set()
    thinking.value = ''
    thinkingDuration.value = 0
    _clearThinkingTimer()
  }

  // --- Execute ---

  async function execute() {
    if (!steps.value.length) return
    running.value = true
    logMessages.value = []

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
      addLog((err as Error).message)
      running.value = false
    }
  }

  return {
    experimentName, steps, codeViewMode, pythonCode, editableJsonCode, editablePythonCode,
    compileStatus, editingStepIndex,
    draggedStepIndex, codeAreaMinimized, codeAreaFullscreen,
    hardwareTools, algorithms, loading, running, error, logMessages, thinking, thinkingDuration,
    plan, jsonCode, nestingInfo, blockErrors,
    collapsedBlocks, toggleCollapse, hiddenStepIndices,
    addStep, removeStep, moveStepUp, moveStepDown, moveStep, updateStep, toggleEdit,
    addToolStep, addAlgorithmStep, addHelperFunction,
    loadHardwareTools, loadAlgorithms,
    generateFromAI, compile, compileAndRunExperiment, syncFromCode,
    onJsonFocus, onJsonBlur, onPyFocus, onPyBlur,
    save, loadFromJSON, importFile, clear, execute, addLog,
  }
})
