<script setup lang="ts">
import { useI18n } from 'vue-i18n'
const { t } = useI18n()
defineProps<{
  text: string
  duration?: number
}>()
</script>

<template>
  <details class="thinking-bubble">
    <summary class="thinking-bubble__summary">
      <span>🧠 {{ t('chat.thinkingProcess') }}</span>
      <span v-if="duration" class="thinking-bubble__duration">{{ duration }}s</span>
    </summary>
    <pre class="thinking-bubble__content">{{ text }}</pre>
  </details>
</template>

<style scoped>
.thinking-bubble {
  margin: var(--space-sm) 0;
  border-radius: var(--radius-md);
  background: var(--color-bg-soft);
  border: 1px solid var(--color-border);
  overflow: hidden;
  font-size: 13px;
}

.thinking-bubble__summary {
  display: flex;
  align-items: center;
  gap: var(--space-xs);
  padding: 8px 12px;
  cursor: pointer;
  color: var(--color-text-secondary);
  user-select: none;
}

.thinking-bubble__summary::-webkit-details-marker {
  display: none;
}

.thinking-bubble__duration {
  margin-left: auto;
  font-size: 12px;
  color: var(--color-text-tertiary);
}

.thinking-bubble__content {
  margin: 0;
  padding: 0 12px 12px;
  font-size: 12px;
  line-height: 1.6;
  color: var(--color-text-secondary);
  white-space: pre-wrap;
  word-break: break-word;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
}
</style>
