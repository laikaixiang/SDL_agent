import { ref } from 'vue'
import { defineStore } from 'pinia'
import { listAlgorithms, browseCSV, runAlgorithm } from '@/api/analysis'
import type { useExperimentStore as _ExpStore } from '@/stores/experiment'

interface Algorithm {
  name: string
  chinese_name: string
  description: string
  params_schema?: Record<string, string>
}

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
      params_schema: algo.params_schema,
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
        error.value = data.message || '生成失败'
      }
    } catch (err) {
      error.value = (err as Error).message
    } finally {
      generating.value = false
    }
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
    loadAlgorithms, loadFiles, toggleDetail, setInputFile, setOutputDir,
    addToExperiment, generateAlgorithm, run,
  }
})
