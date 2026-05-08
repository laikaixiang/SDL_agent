<script setup lang="ts">
import { X } from 'lucide-vue-next'
import IconButton from '@/components/common/IconButton.vue'

defineProps<{
  open: boolean
  title?: string
  width?: string
}>()
defineEmits<{ 'update:open': [value: boolean] }>()
</script>

<template>
  <Teleport to="body">
    <Transition name="fade">
      <div v-if="open" class="modal-overlay" @click.self="$emit('update:open', false)">
        <Transition name="slide-bottom">
          <div v-if="open" class="modal-panel" :style="{ maxWidth: width || '500px' }">
            <div class="modal-header" v-if="title">
              <h3>{{ title }}</h3>
              <IconButton title="关闭" @click="$emit('update:open', false)"><X :size="16" /></IconButton>
            </div>
            <div class="modal-body"><slot /></div>
          </div>
        </Transition>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.modal-overlay {
  position: fixed; inset: 0; z-index: 1000;
  display: flex; align-items: center; justify-content: center;
  background: rgba(0, 0, 0, 0.4); backdrop-filter: blur(4px);
}
.modal-panel {
  width: calc(100% - 40px); max-height: 80vh; overflow-y: auto;
  background: var(--color-surface); border-radius: var(--radius-lg);
  box-shadow: var(--shadow-lg); display: flex; flex-direction: column;
}
.modal-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: var(--space-lg) var(--space-lg) 0;
}
.modal-header h3 { font-size: 16px; }
.modal-body { padding: var(--space-lg); flex: 1; }
.slide-bottom-leave-active { position: absolute; }
</style>
