<script setup lang="ts">
import { ref, reactive, onMounted, onUnmounted, nextTick } from 'vue'
import { getPageImage } from '@/api/search'
import { ChevronUp, ChevronDown, Hash } from 'lucide-vue-next'

const props = defineProps<{ pdfPath: string; filename: string }>()

interface PageEntry {
  loading: boolean
  imageBase64: string
}

const totalPages = ref(0)
const pageMap = reactive<Record<number, PageEntry>>({})
const jumpPage = ref('')
let observer: IntersectionObserver | null = null

onMounted(async () => {
  // Guard: don't attempt to load if pdfPath is missing
  if (!props.pdfPath) {
    console.error('[PdfViewer] pdfPath is empty or undefined, cannot load PDF')
    totalPages.value = 0
    return
  }
  // Load page 0 first, which also gives us total_pages from the page_preview API
  const url = `/api/page_preview?doc=${encodeURIComponent(props.pdfPath)}&page=1`
  console.log('[PdfViewer] pdfPath:', props.pdfPath)
  console.log('[PdfViewer] fetching:', url)
  try {
    const resp = await fetch(url)
    console.log('[PdfViewer] page_preview status:', resp.status)
    const json = await resp.json()
    console.log('[PdfViewer] page_preview raw:', JSON.stringify(json).substring(0, 300))
    // API returns { success: true, data: { total_pages, image_base64, ... } }
    // 404 error returns { error: "..." }
    const d = json.data || json
    console.log('[PdfViewer] parsed total_pages:', d.total_pages, 'has image:', !!d.image_base64)
    if (d.error) {
      console.error('[PdfViewer] API error:', d.error)
    }
    if (d.total_pages) totalPages.value = d.total_pages
    const img = d.image_base64 || ''
    pageMap[0] = { loading: false, imageBase64: img.startsWith('data:') ? img.replace('data:image/jpeg;base64,', '') : img }
  } catch (err) {
    console.error('[PdfViewer] fetch error:', err)
    totalPages.value = 1
  }

  // If still no total, fallback
  if (!totalPages.value || totalPages.value < 1) totalPages.value = 1

  console.log('[PdfViewer] totalPages:', totalPages.value, 'pageMap[0] hasImage:', !!pageMap[0]?.imageBase64)

  await nextTick()
  setupObserver()
})

onUnmounted(() => {
  observer?.disconnect()
})

async function loadPage(pageNum: number) {
  if (pageMap[pageNum]) return
  pageMap[pageNum] = { loading: true, imageBase64: '' }
  try {
    console.log('[PdfViewer] loadPage:', pageNum)
    const data = await getPageImage(props.pdfPath, pageNum)
    console.log('[PdfViewer] loadPage result has image:', !!data.image_base64, 'len:', data.image_base64?.length || 0)
    pageMap[pageNum] = { loading: false, imageBase64: data.image_base64 }
  } catch (err) {
    console.error('[PdfViewer] loadPage error:', err)
    pageMap[pageNum] = { loading: false, imageBase64: '' }
  }
}

function setupObserver() {
  observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        const el = entry.target as HTMLElement
        const pn = parseInt(el.dataset.page || '0')
        loadPage(pn)
        if (pn + 1 < totalPages.value) loadPage(pn + 1)
      }
    }
  }, { rootMargin: '400px 0px' })

  // Observe all sentinels (may be none yet, that's OK — watch will re-run)
  document.querySelectorAll('.page-sentinel').forEach(el => observer?.observe(el))
}

// When a new page gets added to pageMap, observe its DOM element
function onPageRendered(el: Element | null) {
  if (el && observer) observer.observe(el)
}

function scrollToPage(pn: number) {
  const el = document.querySelector(`[data-page="${pn}"]`)
  el?.scrollIntoView({ behavior: 'smooth' })
}

function onJump() {
  const pn = parseInt(jumpPage.value)
  if (pn >= 1 && pn <= totalPages.value) {
    // Ensure page is loaded before scrolling
    loadPage(pn - 1)
    setTimeout(() => scrollToPage(pn - 1), 100)
    jumpPage.value = ''
  }
}

function goPrev() {
  const container = document.querySelector('.pdf-scroll')
  if (!container) return
  const pageEls = container.querySelectorAll('.page-sentinel')
  for (const el of pageEls) {
    const rect = el.getBoundingClientRect()
    const cr = container.getBoundingClientRect()
    if (rect.top >= cr.top && rect.top < cr.bottom) {
      const pn = parseInt((el as HTMLElement).dataset.page || '0')
      if (pn > 0) { loadPage(pn - 1); setTimeout(() => scrollToPage(pn - 1), 100) }
      return
    }
  }
}

function goNext() {
  const container = document.querySelector('.pdf-scroll')
  if (!container) return
  const pageEls = container.querySelectorAll('.page-sentinel')
  for (const el of pageEls) {
    const rect = el.getBoundingClientRect()
    const cr = container.getBoundingClientRect()
    if (rect.top >= cr.top && rect.top < cr.bottom) {
      const pn = parseInt((el as HTMLElement).dataset.page || '0')
      if (pn + 1 < totalPages.value) { loadPage(pn + 1); setTimeout(() => scrollToPage(pn + 1), 100) }
      return
    }
  }
}

function pageNumbers(): number[] {
  const nums: number[] = []
  for (let i = 0; i < totalPages.value; i++) nums.push(i)
  return nums
}
</script>

<template>
  <div class="pdf-viewer">
    <div class="pdf-toolbar">
      <span class="pdf-filename" :title="filename">{{ filename }}</span>
      <div class="pdf-nav">
        <button class="nav-btn" title="上一页" @click="goPrev"><ChevronUp :size="16" /></button>
        <div class="page-jump">
          <Hash :size="12" />
          <input
            v-model="jumpPage"
            class="jump-input"
            :placeholder="`1-${totalPages}`"
            @keydown.enter="onJump"
          />
          <span class="total-label">/ {{ totalPages }}</span>
        </div>
        <button class="nav-btn" title="下一页" @click="goNext"><ChevronDown :size="16" /></button>
      </div>
    </div>
    <div class="pdf-scroll" ref="scrollEl">
      <div
        v-for="pn in pageNumbers()"
        :key="pn"
        class="page-sentinel"
        :data-page="pn"
        :ref="(el: any) => onPageRendered(el as Element)"
      >
        <div v-if="!pageMap[pn]" class="page-placeholder">
          <span>{{ pn + 1 }}</span>
        </div>
        <div v-else-if="pageMap[pn].loading" class="page-loading">
          <span>加载第 {{ pn + 1 }} 页...</span>
        </div>
        <img
          v-else-if="pageMap[pn].imageBase64"
          :src="'data:image/jpeg;base64,' + pageMap[pn].imageBase64"
          :alt="`第 ${pn + 1} 页`"
          class="page-img"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
.pdf-viewer {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #1a1a1a;
}
.pdf-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background: #2d2d2d;
  border-bottom: 1px solid #404040;
  flex-shrink: 0;
  gap: 8px;
}
.pdf-filename {
  font-size: 12px;
  color: #ccc;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
  min-width: 0;
}
.pdf-nav {
  display: flex;
  align-items: center;
  gap: 3px;
  flex-shrink: 0;
}
.nav-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: 1px solid #555;
  border-radius: 4px;
  background: #3a3a3a;
  color: #ccc;
  cursor: pointer;
  transition: all 0.15s;
}
.nav-btn:hover { background: #4a4a4a; color: #fff; }
.page-jump {
  display: flex;
  align-items: center;
  gap: 4px;
  color: #999;
}
.jump-input {
  width: 40px;
  padding: 3px 4px;
  border: 1px solid #555;
  border-radius: 3px;
  background: #3a3a3a;
  color: #fff;
  font-size: 12px;
  text-align: center;
  outline: none;
}
.jump-input:focus { border-color: var(--color-primary); }
.total-label { font-size: 12px; color: #888; }
.pdf-scroll {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  scroll-behavior: smooth;
}
.page-placeholder, .page-loading {
  min-height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #666;
  font-size: 13px;
  background: #222;
}
.page-img {
  width: 100%;
  display: block;
  box-shadow: 0 2px 8px rgba(0,0,0,0.5);
}
</style>
