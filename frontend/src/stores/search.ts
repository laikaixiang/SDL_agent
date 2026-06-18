import { ref, computed } from 'vue'
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

  // PDF viewer state (multi-tab, app-level panel)
  interface OpenPdfTab { id: string; pdfPath: string; filename: string }
  let _nextTabId = 0
  const openPdfTabs = ref<OpenPdfTab[]>([])
  const activePdfId = ref<string | null>(null)
  const pdfPanelOpen = ref(false)
  const activePdfTab = computed<OpenPdfTab | null>(() =>
    activePdfId.value ? (openPdfTabs.value.find(t => t.id === activePdfId.value) || null) : null
  )

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
    const existing = openPdfTabs.value.find(t => t.pdfPath === pdfPath)
    if (existing) {
      activePdfId.value = existing.id
      pdfPanelOpen.value = true
      return
    }
    const id = `pdf-${++_nextTabId}`
    openPdfTabs.value.push({ id, pdfPath, filename })
    activePdfId.value = id
    pdfPanelOpen.value = true
  }

  function closePdfViewer() {
    pdfPanelOpen.value = false
    openPdfTabs.value = []
    activePdfId.value = null
  }

  function closePdfTab(id: string) {
    const idx = openPdfTabs.value.findIndex(t => t.id === id)
    if (idx === -1) return
    openPdfTabs.value.splice(idx, 1)
    if (activePdfId.value === id) {
      if (openPdfTabs.value.length > 0) {
        const nextIdx = Math.min(idx, openPdfTabs.value.length - 1)
        activePdfId.value = openPdfTabs.value[nextIdx].id
      } else {
        activePdfId.value = null
        pdfPanelOpen.value = false
      }
    }
  }

  function setActivePdf(id: string) {
    if (openPdfTabs.value.some(t => t.id === id)) {
      activePdfId.value = id
    }
  }

  /**
   * Step 6: 跳转到 PDF 原文位置.
   *
   * 1) 复用 openPdfViewer 打开 tab
   * 2) 通过自定义事件通知 PdfViewer 跳转 + 高亮
   *
   * PdfViewer 监听 'pdf-jump' 事件, payload: { page (0-based), offset, length }
   */
  async function jumpToSource(
    doc: string,
    page: number,
    offset?: number | null,
    length?: number | null,
  ) {
    if (!doc) return
    openPdfViewer(doc, doc)
    // 等待 viewer 挂载, 然后发跳转事件
    setTimeout(() => {
      window.dispatchEvent(new CustomEvent('pdf-jump', {
        detail: { page: page - 1, offset, length }
      }))
    }, 350)
  }

  return {
    query, results, loading, totalPages, preview, previewLoading, error, hasSearched,
    search, viewPage, closePreview,
    literatureList, literatureLoading, literatureTotal, selectedLiterature, abstractLoading,
    loadLiteratureList, viewAbstract, closeAbstract,
    openPdfTabs, activePdfId, activePdfTab, pdfPanelOpen,
    openPdfViewer, closePdfViewer, closePdfTab, setActivePdf, jumpToSource,
  }
})
