<script setup lang="ts">
import { FileText, Eye, Download } from 'lucide-vue-next'
import type { SearchResult } from '@/types/search'

defineProps<{ result: SearchResult }>()
const emit = defineEmits<{ preview: [pdfPath: string, pageNum: number]; extract: [pdfPath: string, pageNum: number] }>()

function pct(sim: number) {
  return Math.round(sim * 100) + '%'
}
</script>

<template>
  <div class="result-card">
    <div class="card-left">
      <div class="similarity-ring" :style="{ '--pct': pct(result.similarity) }">
        <span class="sim-val">{{ pct(result.similarity) }}</span>
      </div>
    </div>
    <div class="card-body">
      <div class="card-header">
        <FileText :size="15" class="file-icon" />
        <span class="pdf-name">{{ result.pdf_name || 'Unknown' }}</span>
        <span class="page-num">p.{{ result.page_num }}</span>
      </div>
      <p class="snippet">{{ result.text_snippet }}</p>
      <div class="card-actions">
        <button class="action-btn" @click="emit('preview', result.pdf_path, result.page_num)">
          <Eye :size="14" /><span>查看页面</span>
        </button>
        <button class="action-btn primary" @click="emit('extract', result.pdf_path, result.page_num)">
          <Download :size="14" /><span>提取此页</span>
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.result-card { display: flex; gap: var(--space-lg); padding: var(--space-lg); background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-md); transition: box-shadow var(--transition-fast); }
.result-card:hover { box-shadow: var(--shadow-md); }
.card-left { flex-shrink: 0; }
.similarity-ring {
  width: 52px; height: 52px; border-radius: 50%;
  background: conic-gradient(var(--color-primary) calc(var(--pct, 0%) * 3.6), var(--color-bg-mute) 0);
  display: flex; align-items: center; justify-content: center; position: relative;
}
.similarity-ring::after {
  content: ''; position: absolute; width: 40px; height: 40px; border-radius: 50%; background: var(--color-surface);
}
.sim-val { position: relative; z-index: 1; font-size: 12px; font-weight: 700; color: var(--color-primary); }
.card-body { flex: 1; min-width: 0; }
.card-header { display: flex; align-items: center; gap: 6px; margin-bottom: var(--space-sm); }
.file-icon { color: var(--color-text-tertiary); flex-shrink: 0; }
.pdf-name { font-size: 14px; font-weight: 600; color: var(--color-text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.page-num { font-size: 12px; color: var(--color-text-tertiary); background: var(--color-bg-soft); padding: 1px 6px; border-radius: var(--radius-sm); white-space: nowrap; }
.snippet { font-size: 13px; color: var(--color-text-secondary); line-height: 1.6; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; margin-bottom: var(--space-md); }
.card-actions { display: flex; gap: var(--space-sm); }
.action-btn {
  display: flex; align-items: center; gap: 4px; padding: 5px 12px;
  border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  background: var(--color-surface); color: var(--color-text-secondary); font-size: 13px;
  cursor: pointer; transition: all var(--transition-fast);
}
.action-btn:hover { background: var(--color-bg-soft); color: var(--color-text); }
.action-btn.primary { background: var(--color-primary); color: #fff; border-color: var(--color-primary); }
.action-btn.primary:hover { opacity: 0.85; }
</style>
