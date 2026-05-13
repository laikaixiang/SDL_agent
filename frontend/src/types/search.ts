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

export interface LiteratureEntry {
  id: string
  title: string
  authors: string
  abstract_summary: string
  innovation_points: string[]
  key_image_desc: string | null
  doi: string
  current_filename: string
  pdf_path: string
  extraction_status: string
  created_at: string
  updated_at: string
}

export interface LiteratureListResponse {
  success: boolean
  entries: LiteratureEntry[]
  total: number
  page: number
  limit: number
}

export interface LiteratureDetailResponse {
  success: boolean
  entry: LiteratureEntry
}
