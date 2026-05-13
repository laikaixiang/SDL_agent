<script setup lang="ts">
import { X, FileText, Lightbulb, Eye } from 'lucide-vue-next'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import type { LiteratureEntry } from '@/types/search'

defineProps<{ entry: LiteratureEntry | null; loading: boolean }>()
const emit = defineEmits<{ close: []; viewPdf: [pdfPath: string, filename: string] }>()
</script>

<template>
  <Transition name="slide-left">
    <div v-if="entry || loading" class="abstract-panel">
      <div class="abstract-header">
        <span>文献摘要</span>
        <button class="close-btn" @click="$emit('close')"><X :size="18" /></button>
      </div>
      <div class="abstract-body">
        <LoadingSpinner v-if="loading" :size="28" label="加载摘要..." />

        <template v-else-if="entry">
          <div class="abstract-section">
            <h2 class="abstract-title">{{ entry.title }}</h2>
            <div v-if="entry.authors" class="abstract-authors">
              <FileText :size="13" />
              <span>{{ entry.authors }}</span>
            </div>
          </div>

          <div v-if="entry.abstract_summary" class="abstract-section">
            <h3 class="section-label">摘要</h3>
            <p class="abstract-text">{{ entry.abstract_summary }}</p>
          </div>

          <div v-if="entry.innovation_points?.length" class="abstract-section">
            <h3 class="section-label">
              <Lightbulb :size="14" />
              创新点
            </h3>
            <ul class="innovation-list">
              <li v-for="(point, i) in entry.innovation_points" :key="i">{{ point }}</li>
            </ul>
          </div>

          <div class="abstract-section abstract-meta">
            <span v-if="entry.current_filename" class="meta-item">文件: {{ entry.current_filename }}</span>
            <span v-if="entry.doi" class="meta-item">DOI: {{ entry.doi }}</span>
          </div>
        </template>
      </div>

      <div v-if="entry" class="abstract-footer">
        <button
          class="btn-view-pdf"
          @click="$emit('viewPdf', entry.pdf_path, entry.current_filename || entry.title)"
        >
          <Eye :size="14" />
          <span>查看PDF</span>
        </button>
      </div>
    </div>
  </Transition>
</template>

<style scoped>
.abstract-panel {
  width: 380px;
  flex-shrink: 0;
  background: var(--color-surface);
  border-right: 1px solid var(--color-border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.abstract-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-md) var(--space-lg);
  border-bottom: 1px solid var(--color-border);
  font-size: 14px;
  font-weight: 600;
  flex-shrink: 0;
}
.close-btn {
  display: flex;
  border: none;
  background: none;
  color: var(--color-text-secondary);
  cursor: pointer;
  padding: 4px;
  border-radius: var(--radius-sm);
}
.close-btn:hover { background: var(--color-bg-soft); }
.abstract-body {
  flex: 1;
  overflow-y: auto;
  padding: var(--space-xl);
}
.abstract-section {
  margin-bottom: var(--space-xl);
}
.abstract-title {
  font-size: 16px;
  font-weight: 700;
  color: var(--color-text);
  line-height: 1.5;
  margin-bottom: var(--space-sm);
}
.abstract-authors {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: var(--color-text-secondary);
}
.section-label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-tertiary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: var(--space-sm);
}
.abstract-text {
  font-size: 13px;
  color: var(--color-text-secondary);
  line-height: 1.7;
}
.innovation-list {
  list-style: none;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: var(--space-sm);
}
.innovation-list li {
  font-size: 13px;
  color: var(--color-text-secondary);
  line-height: 1.5;
  padding-left: var(--space-md);
  border-left: 2px solid var(--color-primary);
}
.abstract-meta {
  padding-top: var(--space-md);
  border-top: 1px solid var(--color-border);
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.meta-item {
  font-size: 11px;
  color: var(--color-text-tertiary);
  font-family: monospace;
}

.abstract-footer {
  padding: var(--space-md) var(--space-xl);
  border-top: 1px solid var(--color-border);
  flex-shrink: 0;
}
.btn-view-pdf {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  width: 100%;
  padding: 10px 0;
  background: var(--color-primary);
  color: #fff;
  border: none;
  border-radius: var(--radius-sm);
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all var(--transition-fast);
}
.btn-view-pdf:hover { opacity: 0.9; transform: translateY(-1px); }

.slide-left-enter-active,
.slide-left-leave-active { transition: all var(--transition-slow); }
.slide-left-enter-from,
.slide-left-leave-to { width: 0; opacity: 0; transform: translateX(-20px); }
</style>
