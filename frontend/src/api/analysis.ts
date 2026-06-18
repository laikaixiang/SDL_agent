import type { Algorithm, PreviewResponse, ColumnsResponse, RecommendResponse } from '@/types/analysis'

export async function listAlgorithms(): Promise<Algorithm[]> {
  const resp = await fetch('/api/list_algorithms')
  const data = await resp.json()
  return data.algorithms || []
}

export async function browseCSV(subdir?: string): Promise<{ csv_files: string[]; files?: { path: string; name: string; folder: string }[] }> {
  const params = subdir ? `?subdir=${encodeURIComponent(subdir)}` : ''
  const resp = await fetch(`/api/browse_csv${params}`)
  return resp.json()
}

export async function runAlgorithm(algo: string, inputFile: string, params: Record<string, unknown> = {}): Promise<{ output_path: string; message: string }> {
  const resp = await fetch('/api/run_algorithm', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ algorithm_name: algo, input_file: inputFile, params }),
  })
  return resp.json()
}

// =============================================================================
// Phase 1 新增: CSV 预览 API
// =============================================================================

/** 拉取 CSV 前 N 行预览 + 列类型推断 */
export async function previewCsv(path: string, n: number = 20): Promise<PreviewResponse> {
  const params = new URLSearchParams({ path, n: String(n) })
  const resp = await fetch(`/api/csv/preview?${params.toString()}`)
  return resp.json()
}

/** 仅取 CSV 列名 (轻量) */
export async function getCsvColumns(path: string): Promise<ColumnsResponse> {
  const params = new URLSearchParams({ path })
  const resp = await fetch(`/api/csv/columns?${params.toString()}`)
  return resp.json()
}

// =============================================================================
// Phase 3 新增: 算法推荐 API
// =============================================================================

/** 让 LLM 根据 CSV 列名推荐最合适的算法 */
export async function recommendAlgorithm(path: string): Promise<RecommendResponse> {
  const params = new URLSearchParams({ path })
  const resp = await fetch(`/api/algorithm/recommend?${params.toString()}`)
  return resp.json()
}
