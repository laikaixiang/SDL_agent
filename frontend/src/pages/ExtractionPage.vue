<script setup lang="ts">
import { watch, onMounted } from 'vue'
import { useSearchStore } from '@/stores/search'
import { useChatStore } from '@/stores/chat'
import { useLayoutStore } from '@/stores/layout'
import SearchBar from '@/components/search/SearchBar.vue'
import SearchResultList from '@/components/search/SearchResultList.vue'
import PagePreview from '@/components/search/PagePreview.vue'
import LiteratureCard from '@/components/search/LiteratureCard.vue'
import AbstractPreview from '@/components/search/AbstractPreview.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import { BookOpen } from 'lucide-vue-next'

const store = useSearchStore()
const chat = useChatStore()
const layout = useLayoutStore()

onMounted(() => {
  store.loadLiteratureList()
})

watch(() => store.loading, (val) => {
  if (val) {
    layout.updateTaskStatus('extraction', 'running', 10)
  } else {
    layout.updateTaskStatus('extraction', 'completed')
  }
})

// Auto-open extraction PDF in the shared PdfPanel
watch(() => chat.extractionPdfPath, (newPath) => {
  if (newPath && chat.extractionFilename) {
    store.openPdfViewer(newPath, chat.extractionFilename)
  }
})

function onPreview(pdfPath: string, pageNum: number) {
  store.viewPage(pdfPath, pageNum)
}

function onExtract(_pdfPath: string, _pageNum: number) {
  chat.enableExtraction()
}

function onSelectLiterature(id: string) {
  store.viewAbstract(id)
}

function onViewPdf(pdfPath: string, filename: string) {
  store.openPdfViewer(pdfPath, filename)
}
</script>

<template>
  <div class="extraction-page">
    <SearchBar v-model="store.query" :loading="store.loading" @search="store.search(store.query)" />

    <div class="content-area">
      <AbstractPreview
        :entry="store.selectedLiterature"
        :loading="store.abstractLoading"
        @close="store.closeAbstract()"
        @view-pdf="onViewPdf"
      />

      <div class="search-results">
        <!-- Loading -->
        <div v-if="store.loading" class="state-center">
          <LoadingSpinner :size="28" label="搜索中..." />
        </div>

        <!-- Error -->
        <div v-else-if="store.error" class="state-center">
          <EmptyState title="搜索失败" :description="store.error" />
        </div>

        <!-- Literature list (before search) -->
        <div v-else-if="!store.hasSearched" class="literature-area">
          <div class="lit-header">
            <div class="lit-header-left">
              <BookOpen :size="16" />
              <span>文献库</span>
              <span v-if="!store.literatureLoading" class="lit-count">共 {{ store.literatureTotal }} 篇</span>
            </div>
          </div>

          <div v-if="store.literatureLoading" class="state-center">
            <LoadingSpinner :size="24" label="加载文献库..." />
          </div>

          <div v-else-if="!store.literatureList.length" class="state-center">
            <EmptyState title="文献库为空" description="暂无已索引的文献，请先将 PDF 放入文献库并执行索引" />
          </div>

          <div v-else class="lit-list">
            <LiteratureCard
              v-for="entry in store.literatureList"
              :key="entry.id"
              :entry="entry"
              @select="onSelectLiterature"
            />
          </div>
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
.content-area { flex: 1; display: flex; overflow: hidden; }
.search-results { flex: 1; overflow-y: auto; }
.state-center { display: flex; align-items: center; justify-content: center; height: 100%; }
.results-area { padding-top: var(--space-sm); }
.results-header { font-size: 13px; color: var(--color-text-tertiary); padding: 0 var(--space-xl); margin-bottom: var(--space-md); }

/* Literature list */
.literature-area { flex: 1; overflow-y: auto; display: flex; flex-direction: column; }
.lit-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-md) var(--space-xl);
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}
.lit-header-left {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text);
}
.lit-count {
  font-size: 12px;
  font-weight: 400;
  color: var(--color-text-tertiary);
  margin-left: var(--space-sm);
}
.lit-list {
  flex: 1;
  overflow-y: auto;
  padding: var(--space-md) var(--space-xl);
  display: flex;
  flex-direction: column;
  gap: var(--space-sm);
}
</style>
