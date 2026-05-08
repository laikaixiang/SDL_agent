<script setup lang="ts">
import { ref, watch } from 'vue'
import { useExtractionStore } from '@/stores/extraction'
import { useLayoutStore } from '@/stores/layout'
import InputBar from '@/components/chat/InputBar.vue'
import ResultCard from '@/components/cards/ResultCard.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import Badge from '@/components/common/Badge.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import ConfirmDialog from '@/components/modals/ConfirmDialog.vue'
import SummaryModal from '@/components/modals/SummaryModal.vue'
import { FileText, Check, X } from 'lucide-vue-next'

const store = useExtractionStore()
const layout = useLayoutStore()
const inputText = ref('')

watch(() => store.isRunning, (val) => {
  if (val) {
    layout.updateTaskStatus('extraction', 'running', 5)
  } else {
    layout.updateTaskStatus('extraction', 'completed')
  }
})

watch(() => store.logMessages.length, (len) => {
  if (store.isRunning && len > 0) {
    layout.updateTaskStatus('extraction', 'running', Math.min(90, 5 + len * 3))
  }
})
const showFieldConfirm = ref(false)
const showSummary = ref(false)

async function onSend(text: string) {
  inputText.value = ''

  if (text.startsWith('帮我搜寻：')) {
    const desc = text.replace('帮我搜寻：', '').trim()
    if (desc) {
      await store.requestFields(desc)
      showFieldConfirm.value = true
    }
  }
}

function confirmAndStart() {
  showFieldConfirm.value = false
  store.start(store.taskDesc, store.fields)
  store.connectSSE()
}

function onComplete() {
  showSummary.value = true
}
</script>

<template>
  <div class="extraction-page">
    <!-- Header -->
    <div class="page-header">
      <h2>文献提取</h2>
      <Badge v-if="store.isRunning" variant="warning">提取中</Badge>
    </div>

    <!-- Input via chat -->
    <div class="extraction-input">
      <InputBar
        v-model="inputText"
        :disabled="store.isRunning"
        placeholder="输入 帮我搜寻：FAPbI3钝化剂参数"
        @send="onSend"
      />
    </div>

    <!-- Progress / Logs -->
    <div class="extraction-body">
      <div v-if="!store.logMessages.length && !store.isRunning" class="body-center">
        <EmptyState title="文献提取" description='输入"帮我搜寻：<关键词>" 开始提取实验参数' />
      </div>

      <div v-else class="log-panel">
        <div v-for="(log, i) in store.logMessages" :key="i" class="log-line" :class="{ error: log.startsWith('⚠️') }">
          {{ log }}
        </div>

        <div v-if="store.currentPage" class="page-reading">
          <FileText :size="14" />
          <span>{{ store.currentPage.pdf_name }} — 第 {{ store.currentPage.page_num }} 页</span>
        </div>

        <div v-if="store.readingActive && store.llmStream" class="llm-stream">
          {{ store.llmStream }}
        </div>

        <div v-if="store.isRunning" class="loading-line">
          <LoadingSpinner :size="16" label="提取中..." />
        </div>
      </div>

      <!-- Findings list -->
      <div v-if="store.findings.length" class="findings-panel">
        <h3>提取结果 ({{ store.findings.length }} 条)</h3>
        <div class="findings-list">
          <ResultCard
            v-for="(f, i) in store.findings"
            :key="i"
            :title="f.value"
            :subtitle="f.tag"
          />
        </div>
      </div>
    </div>

    <!-- Field confirm dialog -->
    <ConfirmDialog
      :open="showFieldConfirm"
      title="确认提取字段"
      :message="'LLM 推断的提取字段: ' + store.fields.join(', ')"
      confirmText="开始提取"
      @confirm="confirmAndStart"
      @cancel="showFieldConfirm = false"
      @update:open="showFieldConfirm = $event"
    />

    <!-- Summary modal -->
    <SummaryModal
      :open="showSummary"
      :summary="store.summary"
      @update:open="showSummary = $event"
    />
  </div>
</template>

<style scoped>
.extraction-page { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
.page-header { display: flex; align-items: center; gap: var(--space-md); padding: var(--space-lg) var(--space-xl) 0; }
.page-header h2 { font-size: 18px; }
.extraction-input { padding: var(--space-md) var(--space-xl); flex-shrink: 0; }
.extraction-body { flex: 1; overflow-y: auto; padding: var(--space-md) var(--space-xl); display: flex; flex-direction: column; gap: var(--space-lg); }
.body-center { flex: 1; display: flex; align-items: center; justify-content: center; }
.log-panel { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-md); padding: var(--space-lg); font-size: 13px; max-height: 200px; overflow-y: auto; }
.log-line { padding: 3px 0; color: var(--color-text-secondary); }
.log-line.error { color: var(--color-error); }
.page-reading { display: flex; align-items: center; gap: var(--space-sm); padding: 8px 0; color: var(--color-primary); font-weight: 500; }
.llm-stream { max-height: 120px; overflow-y: auto; padding: var(--space-sm); margin-top: var(--space-sm); background: var(--color-bg-soft); border-radius: var(--radius-sm); font-size: 13px; color: var(--color-text-secondary); white-space: pre-wrap; line-height: 1.6; }
.loading-line { padding: 8px 0; }
.findings-panel { flex: 1; }
.findings-panel h3 { font-size: 14px; margin-bottom: var(--space-md); color: var(--color-text-secondary); }
.findings-list { display: flex; flex-direction: column; gap: var(--space-sm); }
</style>
