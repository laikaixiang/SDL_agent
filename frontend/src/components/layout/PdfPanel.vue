<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { storeToRefs } from 'pinia'
import { useSearchStore } from '@/stores/search'
import PdfViewer from '@/components/chat/PdfViewer.vue'
import { X, ChevronDown } from 'lucide-vue-next'

const store = useSearchStore()
const { openPdfTabs, activePdfId, activePdfTab, pdfPanelOpen } = storeToRefs(store)

const dropdownOpen = ref(false)
const dropdownRef = ref<HTMLElement | null>(null)

function onOutsideClick(e: MouseEvent) {
  if (dropdownRef.value && !dropdownRef.value.contains(e.target as Node)) {
    dropdownOpen.value = false
  }
}

onMounted(() => document.addEventListener('mousedown', onOutsideClick))
onUnmounted(() => document.removeEventListener('mousedown', onOutsideClick))
</script>

<template>
  <aside v-if="pdfPanelOpen && activePdfTab" class="pdf-panel">
    <div class="pdf-panel-header">
      <div
        ref="dropdownRef"
        class="pdf-tab-selector"
        @click="dropdownOpen = !dropdownOpen"
      >
        <span class="pdf-tab-label">{{ openPdfTabs.length }} 个文档</span>
        <ChevronDown :size="12" class="pdf-tab-chevron" :class="{ open: dropdownOpen }" />
        <div v-if="dropdownOpen && openPdfTabs.length > 0" class="pdf-tab-dropdown">
          <div
            v-for="tab in openPdfTabs"
            :key="tab.id"
            class="pdf-tab-item"
            :class="{ active: tab.id === activePdfId }"
            @click.stop="store.setActivePdf(tab.id); dropdownOpen = false"
          >
            <span class="pdf-tab-item-name">{{ tab.filename }}</span>
            <button
              class="pdf-tab-item-close"
              title="关闭此标签"
              @click.stop="store.closePdfTab(tab.id)"
            >&#x00d7;</button>
          </div>
        </div>
      </div>
      <button class="pdf-panel-close" title="关闭PDF面板" @click="store.closePdfViewer()">
        <X :size="16" />
      </button>
    </div>
    <div class="pdf-panel-body">
      <PdfViewer
        :key="activePdfId ?? undefined"
        :pdf-path="activePdfTab.pdfPath"
        :filename="activePdfTab.filename"
      />
    </div>
  </aside>
</template>

<style scoped>
.pdf-panel {
  width: 520px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  border-left: 2px solid #404040;
  background: #1a1a1a;
  overflow: hidden;
}

.pdf-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #2d2d2d;
  border-bottom: 1px solid #404040;
  flex-shrink: 0;
}

.pdf-tab-selector {
  position: relative;
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 10px;
  cursor: pointer;
  user-select: none;
  flex: 1;
  min-width: 0;
}
.pdf-tab-selector:hover { background: #3a3a3a; }

.pdf-tab-label {
  font-size: 12px;
  color: #ccc;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
  min-width: 0;
}

.pdf-tab-chevron {
  color: #888;
  flex-shrink: 0;
  transition: transform 0.15s;
}
.pdf-tab-chevron.open { transform: rotate(180deg); }

.pdf-tab-dropdown {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background: #2d2d2d;
  border: 1px solid #404040;
  border-top: none;
  z-index: 100;
  max-height: 240px;
  overflow-y: auto;
}

.pdf-tab-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 10px;
  cursor: pointer;
  border-top: 1px solid #3a3a3a;
}
.pdf-tab-item:hover { background: #3a3a3a; }
.pdf-tab-item.active { background: #404040; }

.pdf-tab-item-name {
  font-size: 12px;
  color: #ccc;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
  min-width: 0;
}

.pdf-tab-item-close {
  background: none;
  border: none;
  color: #888;
  font-size: 16px;
  cursor: pointer;
  padding: 2px 4px;
  line-height: 1;
  flex-shrink: 0;
  border-radius: 3px;
  margin-left: 6px;
}
.pdf-tab-item-close:hover { background: #555; color: #f87171; }

.pdf-panel-close {
  display: flex;
  align-items: center;
  justify-content: center;
  background: none;
  border: none;
  color: #999;
  cursor: pointer;
  padding: 6px 8px;
  flex-shrink: 0;
}
.pdf-panel-close:hover { background: #444; color: #fff; }

.pdf-panel-body {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
</style>
