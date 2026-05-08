import { ref } from 'vue'
import { defineStore } from 'pinia'
import type { SearchResult, PagePreview } from '@/types/search'
import { semanticSearch, getPageImage } from '@/api/search'

export const useSearchStore = defineStore('search', () => {
  const query = ref('')
  const results = ref<SearchResult[]>([])
  const loading = ref(false)
  const totalPages = ref(0)
  const preview = ref<PagePreview | null>(null)
  const previewLoading = ref(false)
  const error = ref('')
  const hasSearched = ref(false)

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

  return {
    query, results, loading, totalPages, preview, previewLoading, error, hasSearched,
    search, viewPage, closePreview,
  }
})
