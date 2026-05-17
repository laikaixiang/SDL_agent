<script setup lang="ts">
import { onMounted, watch } from 'vue'
import { useHardwareStore } from '@/stores/hardware'
import { useLayoutStore } from '@/stores/layout'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import Badge from '@/components/common/Badge.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import { Cpu, Wrench, ChevronDown, ChevronRight, Play } from 'lucide-vue-next'

const store = useHardwareStore()
const layout = useLayoutStore()

onMounted(() => {
  store.loadTools()
})

watch(() => store.isRunning, (val) => {
  if (val) {
    layout.updateTaskStatus('hardware', 'running', 10)
  } else {
    layout.updateTaskStatus('hardware', 'completed')
  }
})

watch(() => store.logMessages.length, (len) => {
  if (store.isRunning && len > 0) {
    layout.updateTaskStatus('hardware', 'running', Math.min(90, 10 + len * 5))
  }
})
</script>

<template>
  <div class="hardware-page">
    <div class="page-header">
      <h2><Cpu :size="18" /> {{ $t('modes.hardwareControl') }}</h2>
      <Badge v-if="store.isRunning" variant="warning">{{ $t('hardware.running') }}</Badge>
    </div>

    <div class="page-body">
      <!-- Tools -->
      <section class="tools-section">
        <h3><Wrench :size="14" /> {{ $t('hardware.availableTools') }} ({{ store.tools.length }})</h3>
        <div class="tool-rows">
          <div
            v-for="t in store.tools"
            :key="t.name"
            class="tool-row"
            :class="{ expanded: store.expandedTool === t.name }"
          >
            <div class="tool-row-main" @click="store.toggleExpand(t.name)">
              <component :is="store.expandedTool === t.name ? ChevronDown : ChevronRight"
                :size="14" class="tool-arrow" />
              <div class="tool-info">
                <span class="tool-label">{{ t.description || t.name }}</span>
                <span class="tool-name-en">{{ t.name }}</span>
              </div>
              <button
                class="tool-run-btn"
                :class="{ running: store.isRunning }"
                :disabled="store.isRunning"
                :title="$t('hardware.singleStep')"
                @click.stop="store.runSingleTool(t.name)"
              >
                <Play :size="12" />
              </button>
            </div>

            <div v-if="store.expandedTool === t.name" class="tool-params">
              <div class="param-grid">
                <div v-for="(v, k) in t.params" :key="k" class="param-field">
                  <div class="param-head">
                    <span class="param-label">{{ k }}</span>
                    <Badge v-if="v.required" variant="warning" class="param-badge">{{ $t('hardware.required') }}</Badge>
                    <Badge v-else variant="default" class="param-badge">{{ $t('hardware.optional') }}</Badge>
                  </div>
                  <span class="param-type">{{ v.type }}<span v-if="v.default !== undefined"> · {{ $t('hardware.defaultValue') }}: {{ v.default }}</span></span>
                  <input
                    class="param-input"
                    :placeholder="v.description || k"
                    :value="store.getToolParam(t.name, k)"
                    @input="(e) => store.setToolParam(t.name, k, (e.target as HTMLInputElement).value)"
                  />
                </div>
              </div>
              <div v-if="!Object.keys(t.params).length" class="param-empty">{{ $t('hardware.noParams') }}</div>
            </div>
          </div>
        </div>
        <div v-if="!store.tools.length" class="tools-empty">
          <LoadingSpinner :size="16" :label="$t('hardware.loadingTools')" />
        </div>
      </section>

      <!-- Confirm card -->
      <div v-if="store.confirmMessage" class="confirm-area">
        <ResultCard :title="$t('hardware.confirmOperation')">
          <pre class="confirm-text">{{ store.confirmMessage }}</pre>
          <div class="confirm-actions">
            <button class="btn-cancel" @click="store.dismissConfirm()">{{ $t('common.cancel') }}</button>
            <button class="btn-execute" @click="store.execute()">{{ $t('hardware.confirmExecute') }}</button>
          </div>
        </ResultCard>
      </div>

      <!-- Logs -->
      <div v-if="store.logMessages.length" class="log-panel">
        <div v-for="(log, i) in store.logMessages" :key="i" class="log-line">{{ log }}</div>
        <div v-if="store.isRunning" class="loading-line">
          <LoadingSpinner :size="16" :label="$t('hardware.executing')" />
        </div>
      </div>

      <div v-if="!store.logMessages.length && !store.confirmMessage" class="body-center">
        <EmptyState :title="$t('modes.hardwareControl')" :description="$t('hardware.expandToolHint')" />
      </div>
    </div>

    <!-- Status bar -->
    <div v-if="store.statusMessage" class="status-bar" :class="{ running: store.isRunning }">
      {{ store.statusMessage }}
    </div>
  </div>
</template>

<style scoped>
.hardware-page { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
.page-header { display: flex; align-items: center; gap: var(--space-md); padding: var(--space-lg) var(--space-xl) 0; flex-shrink: 0; }
.page-header h2 { font-size: 18px; display: flex; align-items: center; gap: var(--space-sm); }
.page-body { flex: 1; overflow-y: auto; padding: var(--space-lg) var(--space-xl); display: flex; flex-direction: column; gap: var(--space-lg); }
.body-center { flex: 1; display: flex; align-items: center; justify-content: center; }

.tools-section h3 { font-size: 13px; color: var(--color-text-secondary); margin-bottom: var(--space-md); display: flex; align-items: center; gap: 6px; }
.tools-empty { padding: var(--space-xl); display: flex; justify-content: center; }

.tool-rows { display: flex; flex-direction: column; gap: 2px; }

.tool-row { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-md); overflow: hidden; transition: border-color var(--transition-fast); }
.tool-row:hover { border-color: var(--color-primary-soft); }
.tool-row.expanded { border-color: var(--color-primary); }

.tool-row-main {
  display: flex; align-items: center; gap: var(--space-sm);
  padding: var(--space-md); cursor: pointer;
  user-select: none;
}

.tool-arrow { color: var(--color-text-tertiary); flex-shrink: 0; }

.tool-info { flex: 1; min-width: 0; }
.tool-label { font-size: 14px; font-weight: 600; color: var(--color-text); display: block; }
.tool-name-en { font-size: 11px; color: var(--color-text-tertiary); margin-top: 1px; display: block; }

.tool-run-btn {
  width: 32px; height: 32px;
  display: flex; align-items: center; justify-content: center;
  border: none; border-radius: 50%;
  background: var(--color-primary);
  color: #fff;
  cursor: pointer;
  flex-shrink: 0;
  transition: opacity var(--transition-fast), background var(--transition-fast);
}
.tool-run-btn:hover:not(:disabled) { opacity: 0.85; }
.tool-run-btn:disabled { background: var(--color-bg-mute); color: var(--color-text-tertiary); cursor: default; }
.tool-run-btn.running { background: var(--color-warning); }

.tool-params {
  border-top: 1px solid var(--color-border);
  padding: var(--space-md);
  background: var(--color-bg-soft);
}

.param-grid { display: flex; flex-direction: column; gap: var(--space-sm); }

.param-field { display: flex; flex-direction: column; gap: 2px; }
.param-head { display: flex; align-items: center; gap: var(--space-sm); }
.param-label { font-size: 13px; font-weight: 500; color: var(--color-text); }
.param-badge { font-size: 10px !important; }
.param-type { font-size: 11px; color: var(--color-text-tertiary); }

.param-input {
  padding: 6px 10px; border: 1px solid var(--color-border);
  border-radius: var(--radius-sm); background: var(--color-surface);
  color: var(--color-text); font-size: 13px; width: 100%;
  margin-top: 3px;
}
.param-input:focus { outline: none; border-color: var(--color-primary); }
.param-empty { font-size: 12px; color: var(--color-text-tertiary); padding: var(--space-sm) 0; }

.confirm-area { flex-shrink: 0; }
.confirm-text { font-size: 13px; color: var(--color-text-secondary); white-space: pre-wrap; margin-bottom: var(--space-md); background: var(--color-bg-soft); padding: var(--space-md); border-radius: var(--radius-sm); }
.confirm-actions { display: flex; gap: var(--space-sm); justify-content: flex-end; }
.btn-cancel, .btn-execute { padding: 8px 20px; border: none; border-radius: var(--radius-md); font-size: 14px; cursor: pointer; }
.btn-cancel  { background: var(--color-bg-soft); color: var(--color-text); }
.btn-execute { background: var(--color-primary); color: #fff; }
.btn-execute:hover { opacity: 0.9; }

.log-panel { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-md); padding: var(--space-lg); font-size: 13px; max-height: 200px; overflow-y: auto; }
.log-line { padding: 3px 0; color: var(--color-text-secondary); white-space: pre-wrap; }
.loading-line { padding: 8px 0; }

.status-bar {
  padding: 6px var(--space-xl); font-size: 12px; color: var(--color-text-secondary);
  background: var(--color-bg-soft); border-top: 1px solid var(--color-border);
  flex-shrink: 0;
}
.status-bar.running { background: #fef3c7; color: #92400e; border-top-color: #f59e0b; }
</style>
