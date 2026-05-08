<script setup lang="ts">
defineProps<{ variant?: 'default' | 'success' | 'warning' | 'error' }>()
</script>
<template>
  <span class="badge" :class="variant || 'default'"><slot /></span>
</template>
<style scoped>
.badge { display: inline-flex; align-items: center; padding: 2px 8px; border-radius: var(--radius-full); font-size: 12px; font-weight: 500; }
.default { background: var(--color-bg-mute); color: var(--color-text-secondary); }
.success { background: rgba(16,185,129,0.15); color: var(--color-success); }
.warning { background: rgba(245,158,11,0.15); color: var(--color-warning); }
.error   { background: rgba(239,68,68,0.15); color: var(--color-error); }
</style>
