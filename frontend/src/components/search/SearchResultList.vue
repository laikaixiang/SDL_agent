<script setup lang="ts">
import type { SearchResult } from '@/types/search'
import SearchResultCard from './SearchResultCard.vue'

defineProps<{ results: SearchResult[] }>()
const emit = defineEmits<{ preview: [pdfPath: string, pageNum: number]; extract: [pdfPath: string, pageNum: number] }>()
</script>

<template>
  <div class="result-list">
    <SearchResultCard
      v-for="r in results"
      :key="r.page_id"
      :result="r"
      @preview="(pdf, pn) => emit('preview', pdf, pn)"
      @extract="(pdf, pn) => emit('extract', pdf, pn)"
    />
  </div>
</template>

<style scoped>
.result-list { display: flex; flex-direction: column; gap: var(--space-md); padding: 0 var(--space-xl) var(--space-xl); max-width: 720px; margin: 0 auto; }
</style>
