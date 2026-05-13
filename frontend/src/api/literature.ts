import type { LiteratureListResponse, LiteratureDetailResponse } from '@/types/search'
import { request } from './client'

export async function getLiteratureList(page = 1, limit = 50): Promise<LiteratureListResponse> {
  return request<LiteratureListResponse>(`/api/literature/list?page=${page}&limit=${limit}`)
}

export async function getLiteratureDetail(id: string): Promise<LiteratureDetailResponse> {
  return request<LiteratureDetailResponse>(`/api/literature/detail/${encodeURIComponent(id)}`)
}
