<script setup lang="ts">
import { ref, watch } from 'vue'
import { useSearchStore } from '@/stores/search'
import { useChatStore } from '@/stores/chat'
import { useLayoutStore } from '@/stores/layout'
import SearchBar from '@/components/search/SearchBar.vue'
import SearchResultList from '@/components/search/SearchResultList.vue'
import PagePreview from '@/components/search/PagePreview.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import { FileText, X } from 'lucide-vue-next'

const store = useSearchStore()
const chat = useChatStore()
const layout = useLayoutStore()

const showPdfPreview = ref(false)

watch(() => store.loading, (val) => {
  if (val) {
    layout.updateTaskStatus('extraction', 'running', 10)
  } else {
    layout.updateTaskStatus('extraction', 'completed')
  }
})

// Auto-show PDF preview when extraction starts
watch(() => chat.extractionRunning, (val) => {
  if (val) showPdfPreview.value = true
})

function onPreview(pdfPath: string, pageNum: number) {
  store.viewPage(pdfPath, pageNum)
}

function onExtract(_pdfPath: string, _pageNum: number) {
  chat.enableExtraction()
}
</script>

<template>
  <div class="extraction-page">
    <!-- PDF reading panel (during extraction) -->
    <div v-if="showPdfPreview && chat.extractionRunning" class="pdf-reader">
      <div class="pdf-reader-header">
        <span>AI 正在阅读...</span>
        <div class="pdf-reader-info">
          <FileText :size="13" />
          <span v-if="chat.currentPage">{{ chat.currentPage.filename }} — 第 {{ chat.currentPage.page }} 页</span>
          <span v-else>等待连接...</span>
          <button class="pdf-reader-close" title="关闭预览" @click="showPdfPreview = false"><X :size="16" /></button>
        </div>
      </div>
      <div class="pdf-reader-body" :class="{ scanning: !chat.currentPage }">
        <div class="scan-line" />
        <img
          v-if="chat.currentPage"
          :src="'data:image/jpeg;base64,' + chat.currentPage.image"
          alt="PDF page"
          class="pdf-reader-img"
        />
        <span v-else class="pdf-waiting">正在读取第一页...</span>
      </div>
    </div>

    <!-- Re-open preview button when closed but extraction is running -->
    <button
      v-if="!showPdfPreview && chat.extractionRunning"
      class="pdf-reopen-btn"
      @click="showPdfPreview = true"
    >
      <FileText :size="13" />
      <span>显示 PDF 预览</span>
    </button>

    <SearchBar v-model="store.query" :loading="store.loading" @search="store.search(store.query)" />

    <div class="search-results">
      <!-- Loading -->
      <div v-if="store.loading" class="state-center">
        <LoadingSpinner :size="28" label="搜索中..." />
      </div>

      <!-- Error -->
      <div v-else-if="store.error" class="state-center">
        <EmptyState title="搜索失败" :description="store.error" />
      </div>

      <!-- Empty state before search -->
      <div v-else-if="!store.hasSearched" class="state-center">
        <EmptyState title="文献检索" description="输入自然语言查询，在所有已索引的文献页面中搜索相关内容" />
      </div>

      <!-- No results -->
      <div v-else-if="!store.results.length" class="state-center">
        <EmptyState title="无结果" :description="'未找到与「' + store.query + '」相关的页面'" />
      </div>

      <!-- Results -->
      <div v-else class="results-area">
        <div class="results-header">
          共 {{ store.totalPages }} 页已索引，找到 {{ store.results.length }} 条结果
        </div>
        <SearchResultList
          :results="store.results"
          @preview="onPreview"
          @extract="onExtract"
        />
      </div>
    </div>

    <!-- Page preview panel -->
    <PagePreview
      :image-base64="store.preview?.imageBase64 || null"
      :page-num="store.preview?.pageNum || 0"
      :loading="store.previewLoading"
      @close="store.closePreview()"
    />
  </div>
</template>

<style scoped>
.extraction-page { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
.search-results { flex: 1; overflow-y: auto; }
.state-center { display: flex; align-items: center; justify-content: center; height: 100%; }
.results-area { padding-top: var(--space-sm); }
.results-header { font-size: 13px; color: var(--color-text-tertiary); padding: 0 var(--space-xl); margin-bottom: var(--space-md); }

/* PDF reader panel */
.pdf-reader {
  flex-shrink: 0;
  border-bottom: 1px solid var(--color-border);
  overflow: hidden;
}
.pdf-reader-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 14px;
  background: #1f2937;
  color: #fff;
  font-size: 13px;
  font-weight: 600;
}
.pdf-reader-info {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: #9ca3af;
  font-weight: 400;
}
.pdf-reader-close {
  background: none;
  border: none;
  color: #9ca3af;
  cursor: pointer;
  padding: 2px;
  margin-left: 4px;
}
.pdf-reader-close:hover { color: #fff; }
.pdf-reader-body {
  position: relative;
  background: #f3f4f6;
  display: flex;
  justify-content: center;
  align-items: center;
  max-height: 300px;
  overflow: hidden;
}
.pdf-reader-img {
  max-width: 100%;
  max-height: 300px;
  object-fit: contain;
  box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.pdf-waiting {
  color: var(--color-text-tertiary);
  font-size: 13px;
}

/* Scan line animation */
.scan-line {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 4px;
  background: rgba(16, 185, 129, 0.7);
  box-shadow: 0 0 20px rgba(16, 185, 129, 1);
  display: none;
  z-index: 10;
  animation: scan 2.5s ease-in-out infinite alternate;
}
.scanning .scan-line { display: block; }
@keyframes scan {
  0% { top: 0; }
  100% { top: calc(100% - 4px); }
}

/* Re-open button */
.pdf-reopen-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  margin: 8px var(--space-xl);
  border: 1px dashed var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-bg-soft);
  color: var(--color-text-secondary);
  font-size: 12px;
  cursor: pointer;
  transition: all var(--transition-fast);
}
.pdf-reopen-btn:hover {
  border-color: var(--color-primary);
  color: var(--color-primary);
  background: var(--color-primary-mute);
}
</style>
