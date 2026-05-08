import type { SearchResponse } from '@/types/search'
import { request } from './client'

export async function semanticSearch(query: string, topK = 10): Promise<SearchResponse> {
  return request<SearchResponse>('/api/semantic_search', {
    method: 'POST',
    body: { query, top_k: topK },
  })
}

export async function getPageImage(pdfPath: string, pageNum: number): Promise<{ success: boolean; image_base64: string; page_num: number }> {
  return request('/api/page_image', {
    method: 'POST',
    body: { pdf_path: pdfPath, page_num: pageNum },
  })
}
