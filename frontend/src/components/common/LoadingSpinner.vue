<script setup lang="ts">
import { computed } from 'vue'
const props = defineProps<{ size?: number; label?: string }>()
const px = computed(() => `${props.size || 24}px`)
</script>
<template>
  <div class="spinner-wrap">
    <svg class="spinner" :style="{ width: px, height: px }" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="10" fill="none" stroke="var(--color-border)" stroke-width="3" />
      <path d="M12 2a10 10 0 0 1 10 10" fill="none" stroke="var(--color-primary)" stroke-width="3" stroke-linecap="round" />
    </svg>
    <span v-if="label" class="label">{{ label }}</span>
  </div>
</template>
<style scoped>
.spinner-wrap { display: flex; align-items: center; gap: var(--space-sm); color: var(--color-text-secondary); }
.spinner { animation: spin 1s linear infinite; }
.label { font-size: 13px; }
</style>
