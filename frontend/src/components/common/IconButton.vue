<script setup lang="ts">
defineProps<{ title?: string }>()
defineEmits<{ click: [] }>()
</script>
<template>
  <button class="icon-btn" :title="title" @click="$emit('click')"><slot /></button>
</template>
<style scoped>
.icon-btn {
  display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px;
  border: none; border-radius: var(--radius-sm); background: transparent; color: var(--color-text-secondary);
  transition: background var(--transition-fast), color var(--transition-fast);
}
.icon-btn:hover { background: var(--color-bg-soft); color: var(--color-text); }
</style>
