import { ref } from 'vue'
import { defineStore } from 'pinia'
import { listAlgorithms, browseCSV, runAlgorithm, previewCsv } from '@/api/analysis'
import type { useExperimentStore as _ExpStore } from '@/stores/experiment'
import type { Algorithm, PreviewData } from '@/types/analysis'
import i18n from '@/i18n'

export const useAnalysisStore = defineStore('analysis', () => {
  const algorithms = ref<Algorithm[]>([])
  const csvFiles = ref<string[]>([])
  const selectedAlgo = ref<Algorithm | null>(null)
  const selectedFile = ref('')
  const loading = ref(false)
  const result = ref<{ output_path: string; message: string } | null>(null)
  const error = ref('')
  const expandedAlgo = ref<string | null>(null)
  const algoInputFiles = ref<Record<string, string>>({})
  const algoOutputDirs = ref<Record<string, string>>({})
  const generating = ref(false)

  async function loadAlgorithms() {
    try { algorithms.value = await listAlgorithms() } catch { /* empty */ }
  }

  async function loadFiles() {
    try {
      const data = await browseCSV()
      csvFiles.value = data.csv_files || []
    } catch { /* empty */ }
  }

  function toggleDetail(algoName: string) {
    expandedAlgo.value = expandedAlgo.value === algoName ? null : algoName
  }

  function setInputFile(algoName: string, path: string) {
    algoInputFiles.value[algoName] = path
  }

  function setOutputDir(algoName: string, path: string) {
    algoOutputDirs.value[algoName] = path
  }

  async function addToExperiment(algo: Algorithm) {
    const { useExperimentStore } = await import('@/stores/experiment')
    const expStore = useExperimentStore()
    expStore.addAlgorithmStep({
      name: algo.name,
      chinese_name: algo.chinese_name,
      description: algo.description,
      params_schema: algo.params_schema as Record<string, string> | undefined,
    })
  }

  async function generateAlgorithm(desc: string) {
    generating.value = true
    error.value = ''
    try {
      const resp = await fetch('/api/generate_algorithm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description: desc }),
      })
      const data = await resp.json()
      if (data.success) {
        await loadAlgorithms()
      } else {
        error.value = data.message || ''
      }
    } catch (err) {
      error.value = (err as Error).message
    } finally {
      generating.value = false
    }
  }

  // 引导式算法生成状态
  const showGuide = ref(false)
  const guideReply = ref('')
  const guideProgress = ref('')
  const guideSessionId = ref('')
  const guideDone = ref(false)

  async function startGuide() {
    showGuide.value = true
    guideDone.value = false
    generating.value = true
    error.value = ''
    try {
      const resp = await fetch('/api/algorithm_gen/guide', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      const data = await resp.json()
      if (data.stage === 'question') {
        guideReply.value = data.reply
        guideProgress.value = data.progress
        guideSessionId.value = data.session_id
      } else {
        guideReply.value = data.reply || i18n.global.t('analysis.startFailed')
        guideDone.value = true
      }
    } catch (err) {
      error.value = (err as Error).message
      showGuide.value = false
    } finally {
      generating.value = false
    }
  }

  async function submitGuideAnswer(answer: string) {
    if (guideDone.value || !guideSessionId.value) return
    generating.value = true
    error.value = ''
    try {
      const resp = await fetch('/api/algorithm_gen/guide', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: guideSessionId.value, answer, action: 'answer' }),
      })
      const data = await resp.json()
      if (data.stage === 'question') {
        guideReply.value = data.reply
        guideProgress.value = data.progress
      } else {
        guideReply.value = data.reply
        guideProgress.value = 'complete'
        guideDone.value = true
        if (data.success) {
          await loadAlgorithms()
        }
      }
    } catch (err) {
      error.value = (err as Error).message
    } finally {
      generating.value = false
    }
  }

  async function guideGoBack() {
    if (!guideSessionId.value) return
    try {
      const resp = await fetch('/api/algorithm_gen/guide', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: guideSessionId.value, action: 'back' }),
      })
      const data = await resp.json()
      if (data.stage === 'question') {
        guideReply.value = data.reply
        guideProgress.value = data.progress
        return data.previous_answer || ''
      }
    } catch { /* ignore */ }
    return ''
  }

  function cancelGuide() {
    if (guideSessionId.value && !guideDone.value) {
      fetch('/api/algorithm_gen/guide', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: guideSessionId.value, action: 'cancel' }),
      }).catch(() => {})
    }
    showGuide.value = false
    guideSessionId.value = ''
    guideReply.value = ''
  }

  async function run() {
    if (!selectedAlgo.value || !selectedFile.value) return
    loading.value = true
    error.value = ''
    result.value = null
    try {
      result.value = await runAlgorithm(selectedAlgo.value.name, selectedFile.value)
    } catch (err) {
      error.value = (err as Error).message
    } finally {
      loading.value = false
    }
  }

  return {
    algorithms, csvFiles, selectedAlgo, selectedFile, loading, result, error,
    expandedAlgo, algoInputFiles, algoOutputDirs, generating,
    showGuide, guideReply, guideProgress, guideSessionId, guideDone,
    loadAlgorithms, loadFiles, toggleDetail, setInputFile, setOutputDir,
    addToExperiment, generateAlgorithm, startGuide, submitGuideAnswer,
    guideGoBack, cancelGuide, run,
  }
})
