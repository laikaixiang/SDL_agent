<script setup lang="ts">
import ModalContainer from './ModalContainer.vue'

defineProps<{
  open: boolean
  title: string
  message: string
  confirmText?: string
  cancelText?: string
  danger?: boolean
}>()
defineEmits<{
  'update:open': [value: boolean]
  confirm: []
  cancel: []
}>()
</script>

<template>
  <ModalContainer :open="open" :title="title" width="400px" @update:open="$emit('update:open', $event)">
    <p class="msg">{{ message }}</p>
    <div class="actions">
      <button class="btn-cancel" @click="$emit('cancel'); $emit('update:open', false)">{{ cancelText || '取消' }}</button>
      <button class="btn-confirm" :class="{ danger }" @click="$emit('confirm'); $emit('update:open', false)">{{ confirmText || '确认' }}</button>
    </div>
  </ModalContainer>
</template>

<style scoped>
.msg { font-size: 14px; color: var(--color-text-secondary); margin-bottom: var(--space-lg); }
.actions { display: flex; justify-content: flex-end; gap: var(--space-sm); }
.btn-cancel, .btn-confirm { padding: 8px 20px; border-radius: var(--radius-md); font-size: 14px; border: none; cursor: pointer; transition: background var(--transition-fast); }
.btn-cancel  { background: var(--color-bg-soft); color: var(--color-text); }
.btn-confirm { background: var(--color-primary); color: #fff; }
.btn-confirm.danger { background: var(--color-error); }
.btn-cancel:hover  { background: var(--color-bg-mute); }
.btn-confirm:hover { opacity: 0.9; }
</style>
