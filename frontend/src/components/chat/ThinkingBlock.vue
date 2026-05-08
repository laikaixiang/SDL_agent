<script setup lang="ts">
import { ref } from 'vue'
import { Brain } from 'lucide-vue-next'

defineProps<{ content: string; duration?: number }>()
const collapsed = ref(false)
</script>

<template>
  <div class="thinking-block">
    <button class="thinking-toggle" @click="collapsed = !collapsed">
      <Brain :size="14" />
      <span>思考{{ duration ? `了 ${duration}s` : '中' }}...</span>
      <span class="chevron">{{ collapsed ? '▶' : '▼' }}</span>
    </button>
    <div class="thinking-content" v-show="!collapsed">{{ content }}</div>
  </div>
</template>

<style scoped>
.thinking-block { margin-bottom: var(--space-md); border-radius: var(--radius-md); background: var(--color-bg-soft); border: 1px solid var(--color-border); overflow: hidden; }
.thinking-toggle { display: flex; align-items: center; gap: var(--space-xs); width: 100%; padding: 8px 12px; border: none; background: none; color: var(--color-text-secondary); font-size: 13px; cursor: pointer; }
.chevron { font-size: 10px; margin-left: auto; }
.thinking-content { padding: 0 12px 12px 30px; font-size: 13px; color: var(--color-text-secondary); white-space: pre-wrap; line-height: 1.6; }
</style>
