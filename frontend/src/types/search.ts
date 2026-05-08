export interface SearchResult {
  page_id: string
  pdf_path: string
  pdf_name: string
  page_num: number
  text_snippet: string
  similarity: number
}

export interface SearchResponse {
  success: boolean
  query: string
  total_pages_indexed: number
  results: SearchResult[]
}

export interface PagePreview {
  imageBase64: string
  pageNum: number
}
