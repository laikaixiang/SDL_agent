/**
 * software 模块前端类型契约 (frontend/src/types/analysis.ts)
 *
 * 这是 CSV 预览 + 结果可视化的"合同"文件。
 * - 后端 app.py 的 /api/csv/preview, /api/csv/columns, /api/list_algorithms 响应结构
 * - BaseAlgorithm.result_schema 字段类型
 * - ResultRenderer 派发所需的子类型
 *
 * 修改此文件需同步: app.py 路由, software/algorithms/base.py, 各算法 result_schema
 */

// =============================================================================
// CSV 预览 (Phase 1)
// =============================================================================

/** 列类型推断结果 */
export type ColumnType = 'int' | 'float' | 'bool' | 'date' | 'str'

/** 单列元数据 */
export interface ColumnInfo {
  name: string
  type: ColumnType
  /** 前 3 个非空值样本 (字符串形式) */
  sample: string[]
  non_null_count: number
  null_count: number
}

/** /api/csv/preview 响应 data */
export interface PreviewData {
  path: string
  columns: ColumnInfo[]
  /** 实际读出的行数 (≤ n) */
  row_count: number
  /** 估算的总行数 */
  total_rows: number
  /** 文件字节数 */
  file_size: number
}

/** /api/csv/preview 完整响应 */
export interface PreviewResponse {
  success: boolean
  data?: PreviewData
  message?: string
}

/** /api/csv/columns 响应 */
export interface ColumnsResponse {
  success: boolean
  data?: { columns: string[] }
  message?: string
}

// =============================================================================
// Algorithm 元数据 (Phase 1 + Phase 2)
// =============================================================================

/** params_schema 单字段定义 */
export interface ParamSchema {
  type: 'int' | 'float' | 'bool' | 'str' | 'list' | 'columns'
  description: string
  default?: unknown
  required?: boolean
  /** str 类型的候选值 (下拉框) */
  options?: string[]
}

export interface Algorithm {
  name: string
  chinese_name: string
  description: string
  params_schema?: Record<string, ParamSchema>
  /** 结果渲染 schema (Phase 2 新增, 可选) */
  result_schema?: ResultSchema
  tags?: string[]
  icon?: string
}

// =============================================================================
// Result Schema (Phase 2)
// =============================================================================

/**
 * result_schema 顶层类型:
 *  - "table"  : 通用表格 (statistics 类)
 *  - "kv"     : 键值对 (spectrum_analysis 类)
 *  - "chart"  : 折线/柱状图
 *  - "matrix" : 矩阵 (correlation 类, 通常与 table 组合)
 *  - "list"   : 列表
 */
export type ResultSchemaType = 'table' | 'kv' | 'chart' | 'matrix' | 'list'

/** 单列表头定义 */
export interface TableColumn {
  key: string
  label: string
  /** 格式化规则: "decimal:3" / "integer" / "percent:2" / "scientific" */
  format?: string
}

/** result_schema.sections 单项 */
export interface ResultSection {
  title?: string
  /** 段类型: table | matrix | chart, 默认继承顶层 type */
  type?: 'table' | 'matrix' | 'chart' | 'kv'
  /** 表格列定义 (type=table/matrix 用) */
  columns?: TableColumn[]
  /**
   * 数据源路径, 从 run() 返回的 result 字典中取值
   * 例: "result.statistics" → result["result"]["statistics"]
   */
  rows_from: string
  /** 矩阵值格式 (type=matrix 用) */
  value_format?: string
}

/** kv 类型单项 */
export interface KvItem {
  key: string
  label: string
  unit?: string
  format?: string
}

/** chart 类型配置 */
export interface ChartConfig {
  x_from: string
  y_from: string
  /** 折线 (line) / 柱状 (bar) */
  chart_type: 'line' | 'bar'
  title?: string
}

/** 顶层 result_schema */
export interface ResultSchema {
  type: ResultSchemaType
  /** table 模式: 多段表格 */
  sections?: ResultSection[]
  /** kv 模式: 键值对列表 */
  items?: KvItem[]
  /** chart 模式: 图表配置 */
  config?: ChartConfig
  /** 任意附加元数据 */
  meta?: Record<string, unknown>
}

// =============================================================================
// Run Result 包装 (前端用)
// =============================================================================

/** runAlgorithm / auto_analyze 返回结果 */
export interface RunResult {
  success: boolean
  algorithm: string
  result?: unknown
  message: string
  output_path?: string
  /** 兼容 SSE 推送的 output_path_latest / output_path_archive */
  output_path_latest?: string
  output_path_archive?: string
}

// =============================================================================
// Algorithm Recommend (Phase 3)
// =============================================================================

export interface AlgorithmRecommend {
  algorithm: string
  read_function: string
  read_params: Record<string, unknown>
  reasoning: string
}

export interface RecommendResponse {
  success: boolean
  data?: AlgorithmRecommend
  message?: string
}
