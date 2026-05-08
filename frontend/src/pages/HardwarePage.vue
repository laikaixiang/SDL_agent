<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { useHardwareStore } from '@/stores/hardware'
import { useLayoutStore } from '@/stores/layout'
import InputBar from '@/components/chat/InputBar.vue'
import ResultCard from '@/components/cards/ResultCard.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import Badge from '@/components/common/Badge.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import ConfirmDialog from '@/components/modals/ConfirmDialog.vue'
import { Cpu, Wrench } from 'lucide-vue-next'

const store = useHardwareStore()
const layout = useLayoutStore()
const inputText = ref('')

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

async function onSend(text: string) {
  inputText.value = ''
  await store.sendCommand(text)
}
</script>

<template>
  <div class="hardware-page">
    <div class="page-header">
      <h2>硬件控制</h2>
      <Badge v-if="store.isRunning" variant="warning">运行中</Badge>
    </div>

    <div class="page-body">
      <!-- Tools panel -->
      <div class="tools-panel" v-if="store.tools.length">
        <h3><Wrench :size="14" /> 可用工具 ({{ store.tools.length }})</h3>
        <div class="tool-list">
          <div v-for="t in store.tools" :key="t.name" class="tool-item">
            <Cpu :size="14" class="tool-icon" />
            <div>
              <div class="tool-name">{{ t.name }}</div>
              <div class="tool-desc">{{ t.description }}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Command input -->
      <div class="command-area">
        <InputBar
          v-model="inputText"
          :disabled="store.isRunning"
          placeholder="硬件控制：设置加热台温度为150度"
          @send="onSend"
        />
      </div>

      <!-- Confirm card -->
      <div v-if="store.confirmMessage" class="confirm-area">
        <ResultCard title="确认操作">
          <pre class="confirm-text">{{ store.confirmMessage }}</pre>
          <div class="confirm-actions">
            <button class="btn-cancel" @click="store.dismissConfirm()">取消</button>
            <button class="btn-execute" @click="store.execute()">确认执行</button>
          </div>
        </ResultCard>
      </div>

      <!-- Logs -->
      <div v-if="store.logMessages.length" class="log-panel">
        <div v-for="(log, i) in store.logMessages" :key="i" class="log-line">{{ log }}</div>
        <div v-if="store.isRunning" class="loading-line">
          <LoadingSpinner :size="16" label="执行中..." />
        </div>
      </div>

      <div v-if="!store.logMessages.length && !store.confirmMessage" class="body-center">
        <EmptyState title="硬件控制" description='输入"硬件控制：<指令>" 操控实验设备' />
      </div>
    </div>
  </div>
</template>

<style scoped>
.hardware-page { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
.page-header { display: flex; align-items: center; gap: var(--space-md); padding: var(--space-lg) var(--space-xl) 0; }
.page-header h2 { font-size: 18px; }
.page-body { flex: 1; overflow-y: auto; padding: var(--space-lg) var(--space-xl); display: flex; flex-direction: column; gap: var(--space-lg); }
.body-center { flex: 1; display: flex; align-items: center; justify-content: center; }
.tools-panel h3 { font-size: 13px; color: var(--color-text-secondary); margin-bottom: var(--space-md); display: flex; align-items: center; gap: 6px; }
.tool-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: var(--space-sm); }
.tool-item { display: flex; gap: var(--space-sm); padding: var(--space-md); background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-md); }
.tool-icon { color: var(--color-primary); margin-top: 2px; flex-shrink: 0; }
.tool-name { font-size: 14px; font-weight: 600; }
.tool-desc { font-size: 12px; color: var(--color-text-secondary); margin-top: 2px; }
.command-area { flex-shrink: 0; }
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
</style>
