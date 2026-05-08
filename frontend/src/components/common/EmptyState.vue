<script setup lang="ts">
import { Inbox } from 'lucide-vue-next'
defineProps<{ title?: string; description?: string }>()
</script>
<template>
  <div class="empty">
    <Inbox :size="48" stroke-width="1" class="empty-icon" />
    <h3 v-if="title">{{ title }}</h3>
    <p v-if="description">{{ description }}</p>
  </div>
</template>
<style scoped>
.empty { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: var(--space-2xl); color: var(--color-text-tertiary); text-align: center; }
.empty-icon { margin-bottom: var(--space-md); }
h3 { font-size: 16px; color: var(--color-text-secondary); margin-bottom: var(--space-xs); }
p { font-size: 14px; }
</style>
