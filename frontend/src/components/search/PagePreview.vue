<script setup lang="ts">
import { X } from 'lucide-vue-next'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'

defineProps<{ imageBase64: string | null; pageNum: number; loading: boolean }>()
defineEmits<{ close: [] }>()
</script>

<template>
  <Transition name="slide-right">
    <div v-if="imageBase64 || loading" class="preview-panel">
      <div class="preview-header">
        <span>{{ $t('search.pagePreview', { n: pageNum }) }}</span>
        <button class="close-btn" @click="$emit('close')"><X :size="18" /></button>
      </div>
      <div class="preview-body">
        <LoadingSpinner v-if="loading" :size="32" :label="$t('search.loadingPage')" />
        <img v-else-if="imageBase64" :src="'data:image/jpeg;base64,' + imageBase64" alt="PDF page" class="preview-img" />
      </div>
    </div>
  </Transition>
</template>

<style scoped>
.preview-panel {
  position: fixed; top: 0; right: 0; z-index: 900;
  width: var(--panel-width); height: 100vh;
  background: var(--color-surface); border-left: 1px solid var(--color-border);
  box-shadow: var(--shadow-lg); display: flex; flex-direction: column;
}
.preview-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: var(--space-lg); border-bottom: 1px solid var(--color-border);
  font-size: 14px; font-weight: 600; flex-shrink: 0;
}
.close-btn { display: flex; border: none; background: none; color: var(--color-text-secondary); cursor: pointer; padding: 4px; border-radius: var(--radius-sm); }
.close-btn:hover { background: var(--color-bg-soft); }
.preview-body { flex: 1; overflow-y: auto; display: flex; align-items: center; justify-content: center; padding: var(--space-lg); }
.preview-img { max-width: 100%; height: auto; border-radius: var(--radius-sm); box-shadow: var(--shadow-md); }

.slide-right-enter-active,
.slide-right-leave-active { transition: transform var(--transition-slow); }
.slide-right-enter-from,
.slide-right-leave-to { transform: translateX(100%); }
</style>
