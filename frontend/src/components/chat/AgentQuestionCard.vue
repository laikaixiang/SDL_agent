<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  question: string
  options?: string
  answer?: string
}>()

const emit = defineEmits<{
  select: [answer: string]
}>()

const parsedOptions = computed<string[]>(() => {
  if (!props.options) return []
  try {
    const parsed = JSON.parse(props.options)
    if (Array.isArray(parsed)) return parsed
  } catch {
    // Not valid JSON array
  }
  return []
})

const isAnswered = computed(() => !!props.answer)
</script>

<template>
  <div class="question-card" :class="{ 'question-card--answered': isAnswered }">
    <div class="question-card__header">
      <span>{{ isAnswered ? '✅' : '❓' }}</span>
      <span>{{ isAnswered ? 'Agent 提问（已回答）' : 'Agent 需要你确认' }}</span>
    </div>
    <p class="question-card__text">{{ question }}</p>
    <div v-if="!isAnswered && parsedOptions.length > 0" class="question-card__options">
      <button
        v-for="(opt, i) in parsedOptions"
        :key="i"
        class="question-card__option"
        @click="emit('select', opt)"
      >{{ opt }}</button>
    </div>
    <p v-if="!isAnswered && parsedOptions.length > 0" class="question-card__hint">或直接在下方输入框回答</p>
    <div v-if="isAnswered" class="question-card__answer">
      <span class="question-card__answer-label">你的回答：</span>
      <span class="question-card__answer-text">{{ answer }}</span>
    </div>
  </div>
</template>

<style scoped>
.question-card {
  margin: var(--space-sm) 0;
  padding: var(--space-md);
  background: #fffbeb;
  border: 1px solid #fde68a;
  border-radius: var(--radius-md);
  color: #92400e;
}

.question-card--answered {
  background: #f0fdf4;
  border-color: #bbf7d0;
  color: #065f46;
}

.question-card__header {
  display: flex;
  align-items: center;
  gap: var(--space-xs);
  font-size: 13px;
  font-weight: 600;
  margin-bottom: var(--space-sm);
}

.question-card__text {
  margin: 0 0 var(--space-md) 0;
  font-size: 14px;
  line-height: 1.6;
  white-space: pre-wrap;
}

.question-card--answered .question-card__text {
  margin-bottom: var(--space-sm);
  opacity: 0.85;
}

.question-card__options {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-sm);
  margin-bottom: var(--space-sm);
}

.question-card__option {
  padding: 6px 16px;
  border: 1px solid #fcd34d;
  border-radius: var(--radius-full);
  background: #fef3c7;
  color: #92400e;
  font-size: 13px;
  cursor: pointer;
  transition: all var(--transition-fast);
}

.question-card__option:hover {
  background: #fde68a;
  border-color: #f59e0b;
}

.question-card__hint {
  margin: 0;
  font-size: 12px;
  opacity: 0.7;
}

.question-card__answer {
  margin-top: var(--space-sm);
  padding: var(--space-xs) var(--space-sm);
  background: rgba(16, 185, 129, 0.1);
  border-radius: var(--radius-sm);
  font-size: 13px;
}

.question-card__answer-label {
  font-weight: 600;
  margin-right: var(--space-xs);
}

.question-card__answer-text {
  word-break: break-word;
}
</style>
