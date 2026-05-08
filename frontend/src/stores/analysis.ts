import { ref } from 'vue'
import { defineStore } from 'pinia'
import { listAlgorithms, browseCSV, runAlgorithm } from '@/api/analysis'

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

  async function loadAlgorithms() {
    try { algorithms.value = await listAlgorithms() } catch { /* empty */ }
  }

  async function loadFiles() {
    try {
      const data = await browseCSV()
      csvFiles.value = data.csv_files || []
    } catch { /* empty */ }
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
    loadAlgorithms, loadFiles, run,
  }
})
