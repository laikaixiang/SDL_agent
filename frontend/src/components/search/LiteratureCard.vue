<script setup lang="ts">
import { FileText } from 'lucide-vue-next'
import type { LiteratureEntry } from '@/types/search'

defineProps<{ entry: LiteratureEntry }>()
const emit = defineEmits<{ select: [id: string] }>()
</script>

<template>
  <div class="lit-card" @click="emit('select', entry.id)">
    <div class="lit-card-icon">
      <FileText :size="18" />
    </div>
    <div class="lit-card-body">
      <div class="lit-card-title">{{ entry.title || entry.current_filename }}</div>
      <div class="lit-card-meta">
        <span v-if="entry.authors" class="lit-card-authors">{{ entry.authors }}</span>
        <span v-if="entry.extraction_status" class="lit-card-status" :class="entry.extraction_status">
          {{ entry.extraction_status === 'done' ? '已提取' : entry.extraction_status }}
        </span>
      </div>
      <div v-if="entry.abstract_summary" class="lit-card-abstract">
        {{ entry.abstract_summary }}
      </div>
    </div>
  </div>
</template>

<style scoped>
.lit-card {
  display: flex;
  gap: var(--space-md);
  padding: var(--space-md) var(--space-lg);
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all var(--transition-fast);
}
.lit-card:hover {
  border-color: var(--color-primary);
  box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08);
}
.lit-card-icon {
  flex-shrink: 0;
  color: var(--color-text-tertiary);
  padding-top: 2px;
}
.lit-card-body {
  flex: 1;
  min-width: 0;
}
.lit-card-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text);
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.lit-card-meta {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  margin-top: 4px;
  flex-wrap: wrap;
}
.lit-card-authors {
  font-size: 12px;
  color: var(--color-text-secondary);
}
.lit-card-status {
  font-size: 11px;
  padding: 1px 6px;
  border-radius: 3px;
  background: var(--color-bg-soft);
  color: var(--color-text-tertiary);
}
.lit-card-status.done {
  background: #d1fae5;
  color: #065f46;
}
.lit-card-abstract {
  margin-top: 6px;
  font-size: 12px;
  color: var(--color-text-secondary);
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
</style>
