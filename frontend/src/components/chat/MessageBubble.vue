<script setup lang="ts">
import { computed } from 'vue'
import { markdownToHtml } from '@/utils/markdown'

const props = defineProps<{
  role: 'user' | 'ai'
  content: string
  timestamp?: string
}>()

const renderedContent = computed(() => {
  if (props.role === 'ai' && props.content) {
    return markdownToHtml(props.content)
  }
  return ''
})
</script>

<template>
  <div class="msg-row" :class="role">
    <div class="msg-bubble" :class="role">
      <slot />
      <div v-if="role === 'ai'" class="msg-text markdown-body" v-html="renderedContent"></div>
      <div v-else class="msg-text">{{ content }}</div>
    </div>
  </div>
</template>

<style scoped>
.msg-row {
  display: flex;
  padding: var(--space-sm) var(--space-xl);
  animation: messageIn 0.3s ease-out;
}
.msg-row.user { justify-content: flex-end; }
.msg-row.ai   { justify-content: flex-start; }

.msg-bubble {
  max-width: 75%;
  padding: 12px 18px;
  border-radius: 20px;
  font-size: 14px;
  line-height: 1.65;
  word-break: break-word;
}
.msg-bubble.user {
  background: var(--color-primary);
  color: #fff;
  border-bottom-right-radius: 6px;
  white-space: pre-wrap;
}
.msg-bubble.ai {
  background: var(--color-surface);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-bottom-left-radius: 6px;
}
.msg-text { }

/* Markdown rendered content */
.markdown-body :deep(h1) { font-size: 1.4em; font-weight: 700; margin: 0.6em 0 0.3em; }
.markdown-body :deep(h2) { font-size: 1.2em; font-weight: 700; margin: 0.5em 0 0.25em; }
.markdown-body :deep(h3) { font-size: 1.05em; font-weight: 600; margin: 0.4em 0 0.2em; }
.markdown-body :deep(p) { margin: 0.3em 0; }
.markdown-body :deep(ul), .markdown-body :deep(ol) { padding-left: 1.5em; margin: 0.3em 0; }
.markdown-body :deep(li) { margin: 0.15em 0; }
.markdown-body :deep(code) {
  background: rgba(0,0,0,0.08);
  padding: 1px 5px;
  border-radius: 4px;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 0.9em;
}
.markdown-body :deep(pre) {
  background: rgba(0,0,0,0.06);
  padding: 10px 14px;
  border-radius: 8px;
  overflow-x: auto;
  margin: 0.4em 0;
}
.markdown-body :deep(pre code) {
  background: none;
  padding: 0;
}
.markdown-body :deep(blockquote) {
  border-left: 3px solid var(--color-primary);
  padding-left: 12px;
  color: var(--color-text-secondary);
  margin: 0.4em 0;
}
.markdown-body :deep(table) { border-collapse: collapse; margin: 0.4em 0; width: 100%; }
.markdown-body :deep(th), .markdown-body :deep(td) {
  border: 1px solid var(--color-border);
  padding: 4px 10px;
  text-align: left;
  font-size: 0.9em;
}
.markdown-body :deep(th) { background: rgba(0,0,0,0.04); font-weight: 600; }
.markdown-body :deep(a) { color: var(--color-primary); }
.markdown-body :deep(hr) { border: none; border-top: 1px solid var(--color-border); margin: 0.6em 0; }
.markdown-body :deep(strong) { font-weight: 700; }
.markdown-body :deep(em) { font-style: italic; }
</style>
