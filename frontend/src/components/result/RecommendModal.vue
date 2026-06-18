<script setup lang="ts">
/**
 * RecommendModal.vue — 智能推荐弹窗 (Phase 3)

 * 展示 LLM 根据 CSV 列名推荐的算法 + 读取函数 + 推荐理由。
 * 用户可选择"采纳"或"换一个"。
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { Sparkles, Check, RefreshCw, AlertCircle, Loader2 } from 'lucide-vue-next'
import ModalContainer from '@/components/modals/ModalContainer.vue'
import type { AlgorithmRecommend } from '@/types/analysis'

const { t } = useI18n()

const props = defineProps<{
  open: boolean
  recommendation: AlgorithmRecommend | null
  loading: boolean
  error: string
}>()

const emit = defineEmits<{
  (e: 'update:open', val: boolean): void
  (e: 'apply'): void
  (e: 'retry'): void
}>()

const algoDisplay = computed(() => {
  if (!props.recommendation) return ''
  return props.recommendation.algorithm
})

const readFnDisplay = computed(() => {
  if (!props.recommendation) return ''
  return props.recommendation.read_function
})

const reasoning = computed(() => {
  if (!props.recommendation) return ''
  return props.recommendation.reasoning
})
</script>

<template>
  <ModalContainer
    :open="open"
    :title="t('analysis.recommend')"
    width="480px"
    @update:open="$emit('update:open', $event)"
  >
    <div class="recommend-body">
      <!-- Loading -->
      <div v-if="loading" class="recommend-loading">
        <Loader2 :size="32" class="spin" />
        <span>{{ t('analysis.recommendRunning') }}</span>
      </div>

      <!-- Error -->
      <div v-else-if="error" class="recommend-error">
        <AlertCircle :size="16" />
        <span>{{ error }}</span>
        <button class="retry-btn" @click="$emit('retry')">
          <RefreshCw :size="14" /> {{ t('common.submit') }}
        </button>
      </div>

      <!-- Empty -->
      <div v-else-if="!recommendation" class="recommend-empty">
        <AlertCircle :size="16" />
        <span>{{ t('analysis.recommendFailed') }}</span>
        <button class="retry-btn" @click="$emit('retry')">
          <RefreshCw :size="14" /> {{ t('common.submit') }}
        </button>
      </div>

      <!-- Result -->
      <template v-else>
        <div class="recommend-result">
          <div class="result-header">
            <Sparkles :size="20" class="sparkle-icon" />
            <span class="result-title">{{ t('analysis.recommendDone') }}</span>
          </div>

          <div class="result-field">
            <span class="field-label">{{ t('analysis.selectAlgorithm') }}</span>
            <code class="field-value">{{ algoDisplay }}</code>
          </div>

          <div class="result-field">
            <span class="field-label">{{ t('analysis.inputFile') }}</span>
            <code class="field-value">{{ readFnDisplay }}</code>
          </div>

          <div v-if="reasoning" class="result-field">
            <span class="field-label">{{ t('analysis.recommendReasoning') }}</span>
            <p class="reasoning-text">{{ reasoning }}</p>
          </div>
        </div>

        <div class="recommend-actions">
          <button class="action-btn apply-btn" @click="$emit('apply')">
            <Check :size="16" /> {{ t('analysis.recommendApply') }}
          </button>
          <button class="action-btn retry-btn-outline" @click="$emit('retry')">
            <RefreshCw :size="16" /> {{ t('analysis.recommendReject') }}
          </button>
        </div>
      </template>
    </div>
  </ModalContainer>
</template>

<style scoped>
.recommend-body {
  display: flex; flex-direction: column; gap: var(--space-lg);
  min-height: 120px;
}

.recommend-loading,
.recommend-empty {
  display: flex; flex-direction: column; align-items: center; gap: var(--space-md);
  padding: var(--space-xl); color: var(--color-text-secondary); font-size: 14px;
}
.spin { animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

.recommend-error {
  display: flex; flex-direction: column; align-items: center; gap: var(--space-sm);
  padding: var(--space-md); color: var(--color-error); font-size: 13px;
  text-align: center;
}
.retry-btn {
  display: flex; align-items: center; gap: 4px;
  padding: 6px 16px; border: 1px solid var(--color-primary);
  border-radius: var(--radius-sm); background: var(--color-primary-soft);
  color: var(--color-primary); font-size: 13px; cursor: pointer;
  margin-top: var(--space-sm);
}
.retry-btn:hover { opacity: 0.85; }

.recommend-result {
  display: flex; flex-direction: column; gap: var(--space-md);
}
.result-header {
  display: flex; align-items: center; gap: var(--space-sm);
  color: var(--color-primary); font-size: 15px; font-weight: 600;
  margin-bottom: var(--space-sm);
}
.sparkle-icon { flex-shrink: 0; }
.result-title { font-size: 15px; }

.result-field {
  display: flex; flex-direction: column; gap: 4px;
}
.field-label {
  font-size: 12px; color: var(--color-text-secondary); font-weight: 500;
}
.field-value {
  font-size: 13px; background: var(--color-bg-soft);
  padding: 4px 8px; border-radius: var(--radius-sm);
  color: var(--color-text); word-break: break-all;
}
.reasoning-text {
  font-size: 13px; color: var(--color-text-secondary);
  line-height: 1.5; margin: 0;
}

.recommend-actions {
  display: flex; gap: var(--space-md); justify-content: flex-end;
  padding-top: var(--space-sm); border-top: 1px solid var(--color-border);
}
.action-btn {
  display: flex; align-items: center; gap: 6px;
  padding: 8px 20px; border-radius: var(--radius-sm);
  font-size: 14px; cursor: pointer; transition: opacity var(--transition-fast);
}
.action-btn:hover { opacity: 0.85; }
.apply-btn {
  border: none; background: var(--color-primary); color: #fff;
}
.retry-btn-outline {
  border: 1px solid var(--color-border);
  background: var(--color-surface); color: var(--color-text-secondary);
}
.retry-btn-outline:hover { border-color: var(--color-primary-soft); color: var(--color-primary); }
</style>
