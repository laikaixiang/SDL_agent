<script setup lang="ts">
import { User, Bot } from 'lucide-vue-next'

defineProps<{
  role: 'user' | 'ai'
  content: string
  timestamp?: string
}>()
</script>

<template>
  <div class="msg-row" :class="role">
    <div class="msg-bubble" :class="role">
      <slot />
      <div class="msg-text">{{ content }}</div>
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
  white-space: pre-wrap;
  word-break: break-word;
}
.msg-bubble.user {
  background: var(--color-primary);
  color: #fff;
  border-bottom-right-radius: 6px;
}
.msg-bubble.ai {
  background: var(--color-surface);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-bottom-left-radius: 6px;
}
.msg-text { }
</style>
