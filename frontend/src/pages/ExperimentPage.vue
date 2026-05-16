<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useExperimentStore } from '@/stores/experiment'
import { useLayoutStore } from '@/stores/layout'
import ElementPanel from '@/components/experiment/ElementPanel.vue'
import StepCanvas from '@/components/experiment/StepCanvas.vue'
import VariableBar from '@/components/experiment/VariableBar.vue'
import CodeArea from '@/components/experiment/CodeArea.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import ConfirmDialog from '@/components/modals/ConfirmDialog.vue'
import { Brain, FlaskConical, Save, Play, Upload, Sparkles, Trash2 } from 'lucide-vue-next'

const store = useExperimentStore()
const layout = useLayoutStore()

const showClearConfirm = ref(false)
const showAIPrompt = ref(false)
const aiPrompt = ref('')
const thinkingCollapsed = ref(false)

const thinkingTimerLabel = computed(() => {
  const d = store.thinkingDuration
  return d > 0 ? `思考中，用时${d.toFixed(1)}秒` : '思考中...'
})

watch(() => store.loading, (val) => {
  if (val) {
    layout.updateTaskStatus('experiment', 'running', 10)
  } else {
    layout.updateTaskStatus('experiment', 'completed')
  }
})

onMounted(() => {
  store.loadHardwareTools()
  store.loadAlgorithms()
})

function onImport() {
  const input = document.createElement('input')
  input.type = 'file'
  input.accept = '.json'
  input.onchange = () => {
    const file = input.files?.[0]
    if (file) store.importFile(file)
  }
  input.click()
}

function onAIGenerate() {
  if (!aiPrompt.value.trim()) return
  showAIPrompt.value = false
  store.generateFromAI(aiPrompt.value)
  aiPrompt.value = ''
}
</script>

<template>
  <div class="experiment-page">
    <!-- Header -->
    <div class="exp-header">
      <div class="exp-title">
        <FlaskConical :size="18" />
        <span>{{ store.experimentName }}</span>
      </div>
      <div class="exp-status">
        <span v-if="store.loading" class="status-loading">AI 设计中...</span>
        <span v-if="store.running" class="status-running">执行中...</span>
      </div>
    </div>

    <!-- Main: ElementPanel | Canvas -->
    <div class="exp-main">
      <ElementPanel />
      <StepCanvas />

      <!-- Thinking / Loading overlay -->
      <div v-if="store.loading" class="exp-loading">
        <div v-if="store.thinking" class="thinking-card">
          <button class="thinking-toggle" @click="thinkingCollapsed = !thinkingCollapsed">
            <Brain :size="14" />
            <span>{{ thinkingTimerLabel }}</span>
            <span class="chevron">{{ thinkingCollapsed ? '▶' : '▼' }}</span>
          </button>
          <div class="thinking-content" v-show="!thinkingCollapsed">{{ store.thinking }}</div>
        </div>
        <LoadingSpinner v-if="!store.thinking" :size="24" label="AI 设计实验中..." />
      </div>

      <!-- Error -->
      <div v-if="store.error && !store.loading" class="exp-error">{{ store.error }}</div>
    </div>

    <!-- Variable bar -->
    <VariableBar />

    <!-- Code area -->
    <CodeArea />

    <!-- Toolbar -->
    <div class="exp-toolbar">
      <button class="tb-btn" @click="store.save()">
        <Save :size="14" /> 保存
      </button>
      <button class="tb-btn" @click="store.execute()" :disabled="!store.steps.length || store.running">
        <Play :size="14" /> 执行
      </button>
      <button class="tb-btn" @click="onImport">
        <Upload :size="14" /> 导入
      </button>
      <button class="tb-btn" @click="showAIPrompt = true">
        <Sparkles :size="14" /> AI 生成
      </button>
      <button class="tb-btn tb-btn-secondary" @click="showClearConfirm = true">
        <Trash2 :size="14" /> 清空
      </button>
    </div>

    <!-- Clear confirm -->
    <ConfirmDialog
      :open="showClearConfirm"
      title="清空实验设计"
      message="确认清空所有步骤？此操作不可撤销。"
      confirmText="清空"
      @confirm="store.clear(); showClearConfirm = false"
      @cancel="showClearConfirm = false"
    />

    <!-- AI prompt modal -->
    <div v-if="showAIPrompt" class="ai-prompt-overlay" @click.self="showAIPrompt = false">
      <div class="ai-prompt-card">
        <h3>AI 生成实验</h3>
        <textarea
          v-model="aiPrompt"
          class="ai-prompt-input"
          placeholder="描述你的实验，例如：设计一个旋涂实验，转速3000rpm，退火150度30分钟"
          rows="4"
          autofocus
        />
        <div class="ai-prompt-actions">
          <button class="btn-cancel" @click="showAIPrompt = false">取消</button>
          <button class="btn-save" @click="onAIGenerate">生成</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.experiment-page {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

/* Header */
.exp-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-md) var(--space-lg);
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}

.exp-title {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  font-size: 15px;
  font-weight: 600;
  color: var(--color-text);
}

.status-loading { font-size: 12px; color: var(--color-warning); }
.status-running { font-size: 12px; color: var(--color-primary); }

/* Main */
.exp-main {
  flex: 1;
  display: flex;
  overflow: hidden;
  position: relative;
}

.exp-loading {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--space-md);
  background: rgba(255,255,255,0.85);
  backdrop-filter: blur(4px);
  z-index: 10;
  padding: var(--space-lg);
}

.thinking-card {
  max-width: 560px;
  width: 100%;
  border-radius: var(--radius-md);
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  box-shadow: var(--shadow-md);
  overflow: hidden;
}

.thinking-toggle {
  display: flex;
  align-items: center;
  gap: var(--space-xs);
  width: 100%;
  padding: 10px 14px;
  border: none;
  background: none;
  color: var(--color-text-secondary);
  font-size: 13px;
  cursor: pointer;
}

.thinking-toggle:hover {
  background: var(--color-bg-soft);
}

.chevron {
  font-size: 10px;
  margin-left: auto;
}

.thinking-content {
  padding: 0 14px 12px 32px;
  font-size: 13px;
  color: var(--color-text-secondary);
  white-space: pre-wrap;
  line-height: 1.7;
  max-height: 260px;
  overflow-y: auto;
}

.exp-error {
  position: absolute;
  bottom: var(--space-md);
  left: 200px;
  right: var(--space-md);
  padding: var(--space-md) var(--space-lg);
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: var(--radius-md);
  color: var(--color-error);
  font-size: 13px;
  z-index: 10;
}

/* Toolbar */
.exp-toolbar {
  display: flex;
  gap: var(--space-sm);
  padding: var(--space-sm) var(--space-lg);
  border-top: 1px solid var(--color-border);
  background: var(--color-surface);
  flex-shrink: 0;
}

.tb-btn {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 7px 14px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-surface);
  color: var(--color-text);
  font-size: 13px;
  cursor: pointer;
  transition: background var(--transition-fast);
}

.tb-btn:hover:not(:disabled) {
  background: var(--color-bg-soft);
}

.tb-btn:disabled {
  opacity: 0.4;
  cursor: default;
}

.tb-btn-secondary {
  margin-left: auto;
}

/* AI prompt */
.ai-prompt-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10001;
}

.ai-prompt-card {
  background: var(--color-surface);
  border-radius: var(--radius-lg);
  padding: var(--space-xl);
  width: 500px;
  max-width: 90vw;
  box-shadow: var(--shadow-lg);
}

.ai-prompt-card h3 {
  font-size: 16px;
  margin-bottom: var(--space-lg);
}

.ai-prompt-input {
  width: 100%;
  padding: var(--space-md);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  font-size: 14px;
  font-family: inherit;
  resize: vertical;
}

.ai-prompt-input:focus {
  outline: none;
  border-color: var(--color-primary);
}

.ai-prompt-actions {
  display: flex;
  gap: var(--space-sm);
  justify-content: flex-end;
  margin-top: var(--space-lg);
}

.btn-cancel, .btn-save {
  padding: 8px 20px;
  border: none;
  border-radius: var(--radius-sm);
  font-size: 14px;
  cursor: pointer;
}

.btn-cancel { background: var(--color-bg-mute); color: var(--color-text); }
.btn-save { background: var(--color-primary); color: #fff; }
.btn-save:hover { opacity: 0.9; }
</style>
