<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { CheckCircle } from 'lucide-vue-next'
import ModalContainer from './ModalContainer.vue'

const { t } = useI18n()

defineProps<{
  open: boolean
  summary: { field_count: number; pdf_count: number; file: string } | null
}>()
defineEmits<{ 'update:open': [value: boolean] }>()
</script>

<template>
  <ModalContainer :open="open" width="420px" @update:open="$emit('update:open', $event)">
    <div class="summary">
      <div class="icon-wrap"><CheckCircle :size="48" stroke-width="1" class="icon" /></div>
      <h3>{{ $t('modals.extractionComplete') }}</h3>
      <div v-if="summary" class="stats">
        <div class="stat"><span class="val">{{ summary.field_count }}</span><span class="lbl">{{ $t('modals.dataItems') }}</span></div>
        <div class="stat"><span class="val">{{ summary.pdf_count }}</span><span class="lbl">{{ $t('modals.documents') }}</span></div>
      </div>
      <p v-if="summary" class="file">{{ $t('modals.savedTo') }}{{ summary.file }}</p>
      <button class="btn-close" @click="$emit('update:open', false)">{{ $t('common.close') }}</button>
    </div>
  </ModalContainer>
</template>

<style scoped>
.summary { display: flex; flex-direction: column; align-items: center; text-align: center; }
.icon-wrap { margin-bottom: var(--space-md); }
.icon { color: var(--color-success); }
h3 { font-size: 18px; margin-bottom: var(--space-lg); }
.stats { display: flex; gap: var(--space-2xl); margin-bottom: var(--space-lg); }
.stat { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.val { font-size: 28px; font-weight: 700; color: var(--color-primary); }
.lbl { font-size: 13px; color: var(--color-text-secondary); }
.file { font-size: 12px; color: var(--color-text-tertiary); margin-bottom: var(--space-xl); max-width: 100%; word-break: break-all; }
.btn-close { padding: 8px 32px; border: none; border-radius: var(--radius-md); background: var(--color-primary); color: #fff; font-size: 14px; cursor: pointer; transition: opacity var(--transition-fast); }
.btn-close:hover { opacity: 0.9; }
</style>
