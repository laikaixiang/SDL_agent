import { marked } from 'marked'
import katex from 'katex'

marked.setOptions({
  breaks: true,
  gfm: true,
})

/**
 * Render LaTeX formulas to HTML via KaTeX, then render markdown via marked.
 * Supports: $$...$$, \[...\] for display math; $...$, \(...\) for inline math.
 */
export function markdownToHtml(text: string): string {
  const displayMath: string[] = []
  const inlineMath: string[] = []

  let processed = text

  // 1. Extract display math $$...$$
  processed = processed.replace(/\$\$([\s\S]*?)\$\$/g, (_match, formula: string) => {
    displayMath.push(formula.trim())
    return `\x00DM${displayMath.length - 1}\x00`
  })

  // 1b. Extract display math \[...\]
  processed = processed.replace(/\\\[([\s\S]*?)\\\]/g, (_match, formula: string) => {
    displayMath.push(formula.trim())
    return `\x00DM${displayMath.length - 1}\x00`
  })

  // 2. Extract inline math $...$ (single $, content must not contain $ or newline)
  processed = processed.replace(/\$([^$\n]+?)\$/g, (_match, formula: string) => {
    inlineMath.push(formula.trim())
    return `\x00IM${inlineMath.length - 1}\x00`
  })

  // 2b. Extract inline math \(...\)
  processed = processed.replace(/\\\(([\s\S]*?)\\\)/g, (_match, formula: string) => {
    inlineMath.push(formula.trim())
    return `\x00IM${inlineMath.length - 1}\x00`
  })

  // 3. Render markdown
  let html = marked.parse(processed) as string

  // 4. Replace display math placeholders
  html = html.replace(/\x00DM(\d+)\x00/g, (_match, idx: string) => {
    const formula = displayMath[Number(idx)]
    try {
      return katex.renderToString(formula, { displayMode: true, throwOnError: false })
    } catch {
      return `<pre>${formula}</pre>`
    }
  })

  // 5. Replace inline math placeholders
  html = html.replace(/\x00IM(\d+)\x00/g, (_match, idx: string) => {
    const formula = inlineMath[Number(idx)]
    try {
      return katex.renderToString(formula, { displayMode: false, throwOnError: false })
    } catch {
      return formula
    }
  })

  return html
}
