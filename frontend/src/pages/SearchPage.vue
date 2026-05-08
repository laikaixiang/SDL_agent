<script setup lang="ts">
import { watch } from 'vue'
import { useSearchStore } from '@/stores/search'
import { useChatStore } from '@/stores/chat'
import { useLayoutStore } from '@/stores/layout'
import SearchBar from '@/components/search/SearchBar.vue'
import SearchResultList from '@/components/search/SearchResultList.vue'
import PagePreview from '@/components/search/PagePreview.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import EmptyState from '@/components/common/EmptyState.vue'

const store = useSearchStore()
const chat = useChatStore()
const layout = useLayoutStore()

watch(() => store.loading, (val) => {
  if (val) {
    layout.updateTaskStatus('search', 'running', 10)
  } else {
    layout.updateTaskStatus('search', 'completed')
  }
})

function onPreview(pdfPath: string, pageNum: number) {
  store.viewPage(pdfPath, pageNum)
}

function onExtract(_pdfPath: string, _pageNum: number) {
  chat.setMode('extraction')
  layout.openTaskPanel('extraction')
}
</script>

<template>
  <div class="search-page">
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
        <EmptyState title="语义搜索" description="输入自然语言查询，在所有已索引的文献页面中搜索相关内容" />
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
.search-page { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
.search-results { flex: 1; overflow-y: auto; }
.state-center { display: flex; align-items: center; justify-content: center; height: 100%; }
.results-area { padding-top: var(--space-sm); }
.results-header { font-size: 13px; color: var(--color-text-tertiary); padding: 0 var(--space-xl); margin-bottom: var(--space-md); max-width: 720px; margin-left: auto; margin-right: auto; }
</style>
