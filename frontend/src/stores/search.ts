import { ref } from 'vue'
import { defineStore } from 'pinia'
import type { SearchResult, PagePreview, LiteratureEntry } from '@/types/search'
import { semanticSearch, getPageImage } from '@/api/search'
import { getLiteratureList, getLiteratureDetail } from '@/api/literature'

export const useSearchStore = defineStore('search', () => {
  const query = ref('')
  const results = ref<SearchResult[]>([])
  const loading = ref(false)
  const totalPages = ref(0)
  const preview = ref<PagePreview | null>(null)
  const previewLoading = ref(false)
  const error = ref('')
  const hasSearched = ref(false)

  // Literature list (all PDFs from registry)
  const literatureList = ref<LiteratureEntry[]>([])
  const literatureLoading = ref(false)
  const literatureTotal = ref(0)
  const selectedLiterature = ref<LiteratureEntry | null>(null)
  const abstractLoading = ref(false)

  // PDF viewer state (shown in ChatContainer)
  const viewPdfFile = ref<{ pdfPath: string; filename: string } | null>(null)
  const pdfPanelOpen = ref(false)

  async function search(q?: string) {
    const searchQuery = q || query.value.trim()
    if (!searchQuery) return

    query.value = searchQuery
    loading.value = true
    error.value = ''
    hasSearched.value = true

    try {
      const data = await semanticSearch(searchQuery, 10)
      results.value = data.results || []
      totalPages.value = data.total_pages_indexed
    } catch (err) {
      error.value = (err as Error).message
      results.value = []
    } finally {
      loading.value = false
    }
  }

  async function viewPage(pdfPath: string, pageNum: number) {
    previewLoading.value = true
    try {
      const data = await getPageImage(pdfPath, pageNum)
      preview.value = {
        imageBase64: data.image_base64,
        pageNum: data.page_num,
      }
    } catch {
      // silently fail
    } finally {
      previewLoading.value = false
    }
  }

  function closePreview() {
    preview.value = null
  }

  // Literature list actions
  async function loadLiteratureList() {
    literatureLoading.value = true
    try {
      const data = await getLiteratureList(1, 50)
      literatureList.value = data.entries || []
      literatureTotal.value = data.total
    } catch {
      literatureList.value = []
    } finally {
      literatureLoading.value = false
    }
  }

  async function viewAbstract(id: string) {
    abstractLoading.value = true
    selectedLiterature.value = null
    try {
      const data = await getLiteratureDetail(id)
      selectedLiterature.value = data.entry
    } catch {
      selectedLiterature.value = null
    } finally {
      abstractLoading.value = false
    }
  }

  function closeAbstract() {
    selectedLiterature.value = null
  }

  function openPdfViewer(pdfPath: string, filename: string) {
    console.log('[searchStore] openPdfViewer called:', pdfPath, filename)
    viewPdfFile.value = { pdfPath, filename }
    pdfPanelOpen.value = true
    console.log('[searchStore] after set, pdfPanelOpen:', pdfPanelOpen.value, 'viewPdfFile:', JSON.stringify(viewPdfFile.value))
  }

  function closePdfViewer() {
    pdfPanelOpen.value = false
    viewPdfFile.value = null
  }

  return {
    query, results, loading, totalPages, preview, previewLoading, error, hasSearched,
    search, viewPage, closePreview,
    literatureList, literatureLoading, literatureTotal, selectedLiterature, abstractLoading,
    loadLiteratureList, viewAbstract, closeAbstract,
    viewPdfFile, pdfPanelOpen, openPdfViewer, closePdfViewer,
  }
})
