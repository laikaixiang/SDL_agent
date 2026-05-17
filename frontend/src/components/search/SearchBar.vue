<script setup lang="ts">
import { Search, X } from 'lucide-vue-next'

const modelValue = defineModel<string>('modelValue', { default: '' })
defineProps<{ loading?: boolean }>()
const emit = defineEmits<{ search: [] }>()

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter') emit('search')
}

function clear() {
  modelValue.value = ''
}
</script>

<template>
  <div class="search-bar-wrap">
    <div class="search-bar">
      <Search :size="20" class="search-icon" />
      <input
        :value="modelValue"
        @input="modelValue = ($event.target as HTMLInputElement).value"
        @keydown="onKeydown"
        type="text"
        class="search-input"
        :placeholder="$t('search.libraryPlaceholder')"
      />
      <button v-if="modelValue" class="clear-btn" @click="clear"><X :size="16" /></button>
      <button class="search-btn" :disabled="loading || !modelValue.trim()" @click="$emit('search')">
        <Search :size="16" />
        <span>{{ $t('search.search') }}</span>
      </button>
    </div>
    <p class="search-hint">{{ $t('search.hint') }}</p>
  </div>
</template>

<style scoped>
.search-bar-wrap { padding: var(--space-xl); max-width: 720px; margin: 0 auto; }
.search-bar { display: flex; align-items: center; gap: var(--space-sm); background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-lg); padding: 8px 12px; transition: border var(--transition-fast), box-shadow var(--transition-fast); }
.search-bar:focus-within { border-color: var(--color-primary); box-shadow: 0 0 0 3px var(--color-primary-mute); }
.search-icon { color: var(--color-text-tertiary); flex-shrink: 0; }
.search-input { flex: 1; border: none; background: none; font-size: 15px; color: var(--color-text); outline: none; min-width: 0; }
.search-input::placeholder { color: var(--color-text-tertiary); }
.clear-btn { display: flex; border: none; background: none; color: var(--color-text-tertiary); cursor: pointer; padding: 2px; }
.search-btn { display: flex; align-items: center; gap: 6px; padding: 8px 18px; border: none; border-radius: var(--radius-full); background: var(--color-primary); color: #fff; font-size: 14px; cursor: pointer; white-space: nowrap; transition: opacity var(--transition-fast); }
.search-btn:disabled { opacity: 0.4; cursor: default; }
.search-btn:not(:disabled):hover { opacity: 0.85; }
.search-hint { font-size: 12px; color: var(--color-text-tertiary); margin-top: var(--space-sm); padding-left: 44px; }
</style>
