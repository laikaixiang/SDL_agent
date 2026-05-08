<script setup lang="ts">
import { ref, watch } from 'vue'
import { useExperimentStore } from '@/stores/experiment'
import { useLayoutStore } from '@/stores/layout'
import ResultCard from '@/components/cards/ResultCard.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import Badge from '@/components/common/Badge.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import InputBar from '@/components/chat/InputBar.vue'
import { FlaskConical, Code, Play, Terminal } from 'lucide-vue-next'

const store = useExperimentStore()
const layout = useLayoutStore()
const inputText = ref('')

watch(() => store.loading, (val) => {
  if (val) {
    layout.updateTaskStatus('experiment', 'running', 10)
  } else {
    layout.updateTaskStatus('experiment', 'completed')
  }
})

async function onSend(text: string) {
  inputText.value = ''
  await store.sendDesignRequest(text)
}
</script>

<template>
  <div class="experiment-page">
    <div class="page-header">
      <h2><FlaskConical :size="20" /> 实验设计</h2>
    </div>

    <div class="page-body">
      <!-- Input -->
      <div class="command-area">
        <InputBar
          v-model="inputText"
          :disabled="store.loading"
          placeholder='输入"实验设计：<描述>" ...'
          @send="onSend"
        />
      </div>

      <!-- Loading -->
      <div v-if="store.loading" class="loading-area">
        <LoadingSpinner :size="24" label="AI 设计实验中..." />
      </div>

      <!-- Error -->
      <div v-if="store.error" class="error-area">{{ store.error }}</div>

      <!-- Plan -->
      <div v-if="store.plan" class="plan-area">
        <div class="plan-header">
          <h3>{{ store.plan.experiment_name }}</h3>
          <div class="plan-actions">
            <button class="tb-btn" :class="{ active: store.codeViewMode === 'json' }" @click="store.codeViewMode = 'json'">
              <Code :size="14" /> JSON
            </button>
            <button class="tb-btn" @click="store.compile()">
              <Terminal :size="14" /> 编译
            </button>
            <button class="tb-btn" :class="{ active: store.codeViewMode === 'python' }" @click="store.codeViewMode = 'python'">
              <Play :size="14" /> Python
            </button>
          </div>
        </div>

        <!-- Step list view -->
        <div v-if="store.codeViewMode === 'json'" class="step-list">
          <div v-for="(s, i) in store.plan.steps" :key="i" class="step-item">
            <div class="step-num">{{ i + 1 }}</div>
            <div class="step-body">
              <div class="step-head">
                <Badge :variant="s.type === 'tool' ? 'success' : s.type === 'helper' ? 'warning' : 'default'">{{ s.type }}</Badge>
                <span class="step-name">{{ s.name }}</span>
              </div>
              <div class="step-desc" v-if="s.description">{{ s.description }}</div>
              <pre class="step-params" v-if="s.params && Object.keys(s.params).length">{{ JSON.stringify(s.params, null, 2) }}</pre>
            </div>
          </div>
        </div>

        <!-- Code view -->
        <div v-else class="code-view">
          <pre><code>{{ store.pythonCode || '点击"编译"生成 Python 代码' }}</code></pre>
        </div>
      </div>

      <!-- Empty -->
      <div v-if="!store.plan && !store.loading && !store.error" class="body-center">
        <EmptyState title="实验设计" description='输入"实验设计：<描述>" AI 将自动规划实验流程' />
      </div>
    </div>
  </div>
</template>

<style scoped>
.experiment-page { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
.page-header { padding: var(--space-lg) var(--space-xl) 0; }
.page-header h2 { font-size: 18px; display: flex; align-items: center; gap: var(--space-sm); }
.page-body { flex: 1; overflow-y: auto; padding: var(--space-lg) var(--space-xl); display: flex; flex-direction: column; gap: var(--space-lg); }
.body-center { flex: 1; display: flex; align-items: center; justify-content: center; }
.command-area { flex-shrink: 0; }
.loading-area { display: flex; justify-content: center; }
.error-area { font-size: 14px; color: var(--color-error); text-align: center; padding: var(--space-xl); }
.plan-area { flex: 1; }
.plan-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: var(--space-lg); }
.plan-header h3 { font-size: 16px; }
.plan-actions { display: flex; gap: 4px; }
.tb-btn { display: flex; align-items: center; gap: 4px; padding: 6px 12px; border: 1px solid var(--color-border); border-radius: var(--radius-sm); background: var(--color-surface); color: var(--color-text-secondary); font-size: 13px; cursor: pointer; }
.tb-btn:hover { background: var(--color-bg-soft); }
.tb-btn.active { background: var(--color-primary-soft); color: var(--color-primary); border-color: var(--color-primary); }
.step-list { display: flex; flex-direction: column; gap: var(--space-sm); }
.step-item { display: flex; gap: var(--space-md); padding: var(--space-md); background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-md); }
.step-num { width: 28px; height: 28px; border-radius: 50%; background: var(--color-primary-soft); color: var(--color-primary); display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 600; flex-shrink: 0; }
.step-body { flex: 1; min-width: 0; }
.step-head { display: flex; align-items: center; gap: var(--space-sm); margin-bottom: 6px; }
.step-name { font-weight: 600; font-size: 14px; }
.step-desc { font-size: 13px; color: var(--color-text-secondary); margin-bottom: 6px; }
.step-params { font-size: 12px; color: var(--color-text-tertiary); background: var(--color-bg-soft); padding: var(--space-sm); border-radius: var(--radius-sm); overflow-x: auto; }
.code-view { background: var(--color-bg-soft); border: 1px solid var(--color-border); border-radius: var(--radius-md); padding: var(--space-lg); overflow: auto; }
.code-view pre { font-size: 13px; line-height: 1.6; white-space: pre-wrap; }
</style>
