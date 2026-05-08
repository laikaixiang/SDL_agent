interface Algorithm {
  name: string
  chinese_name: string
  description: string
  params_schema?: Record<string, string>
}

export async function listAlgorithms(): Promise<Algorithm[]> {
  const resp = await fetch('/api/list_algorithms')
  const data = await resp.json()
  return data.algorithms || []
}

export async function browseCSV(subdir?: string): Promise<{ csv_files: string[] }> {
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
