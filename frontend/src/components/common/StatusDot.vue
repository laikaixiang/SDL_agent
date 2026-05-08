<script setup lang="ts">
import { computed } from 'vue'
const props = defineProps<{ status: 'online' | 'offline' | 'warning' | 'loading'; size?: number }>()
const px = computed(() => `${props.size || 8}px`)
</script>

<template>
  <span class="status-dot" :class="status" :style="{ width: px, height: px }" />
</template>

<style scoped>
.status-dot {
  display: inline-block;
  border-radius: 50%;
  flex-shrink: 0;
}
.online  { background: var(--color-success); }
.offline { background: var(--color-text-tertiary); }
.warning { background: var(--color-warning); }
.loading { background: var(--color-primary); animation: pulse 1.5s infinite; }
</style>
