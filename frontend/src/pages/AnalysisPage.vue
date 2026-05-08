<script setup lang="ts">
import { onMounted } from 'vue'
import { useAnalysisStore } from '@/stores/analysis'
import ResultCard from '@/components/cards/ResultCard.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import { BarChart3, FileText, Play } from 'lucide-vue-next'

const store = useAnalysisStore()

onMounted(() => {
  store.loadAlgorithms()
  store.loadFiles()
})
</script>

<template>
  <div class="analysis-page">
    <div class="page-header">
      <h2><BarChart3 :size="20" /> 数据分析</h2>
    </div>

    <div class="page-body">
      <!-- Algorithms -->
      <section>
        <h3>可用算法</h3>
        <div class="algo-grid">
          <button
            v-for="a in store.algorithms"
            :key="a.name"
            class="algo-card"
            :class="{ active: store.selectedAlgo?.name === a.name }"
            @click="store.selectedAlgo = a"
          >
            <div class="algo-name">{{ a.chinese_name || a.name }}</div>
            <div class="algo-desc">{{ a.description }}</div>
          </button>
        </div>
      </section>

      <!-- Files -->
      <section>
        <h3>CSV 文件</h3>
        <div class="file-list">
          <button
            v-for="f in store.csvFiles"
            :key="f"
            class="file-item"
            :class="{ active: store.selectedFile === f }"
            @click="store.selectedFile = f"
          >
            <FileText :size="14" /><span>{{ f }}</span>
          </button>
        </div>
      </section>

      <!-- Run -->
      <div class="run-bar">
        <button
          class="run-btn"
          :disabled="!store.selectedAlgo || !store.selectedFile || store.loading"
          @click="store.run()"
        >
          <Play :size="16" />
          <span>运行 {{ store.selectedAlgo?.chinese_name || '分析' }}</span>
        </button>
      </div>

      <!-- Progress -->
      <div v-if="store.loading" class="loading-area">
        <LoadingSpinner :size="24" label="分析中..." />
      </div>

      <!-- Error -->
      <div v-if="store.error" class="error-msg">{{ store.error }}</div>

      <!-- Result -->
      <div v-if="store.result">
        <ResultCard title="分析完成" :subtitle="'输出: ' + store.result.output_path">
          <pre class="result-msg">{{ store.result.message }}</pre>
        </ResultCard>
      </div>

      <div v-if="!store.algorithms.length" class="body-center">
        <EmptyState title="数据分析" description="选择算法和 CSV 文件，运行分析" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.analysis-page { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
.page-header { padding: var(--space-lg) var(--space-xl) 0; }
.page-header h2 { font-size: 18px; display: flex; align-items: center; gap: var(--space-sm); }
.page-body { flex: 1; overflow-y: auto; padding: var(--space-lg) var(--space-xl); display: flex; flex-direction: column; gap: var(--space-lg); }
.body-center { flex: 1; display: flex; align-items: center; justify-content: center; }
section h3 { font-size: 14px; color: var(--color-text-secondary); margin-bottom: var(--space-md); }
.algo-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: var(--space-sm); }
.algo-card {
  text-align: left; padding: var(--space-md); border: 1px solid var(--color-border);
  border-radius: var(--radius-md); background: var(--color-surface); cursor: pointer;
  transition: border var(--transition-fast), box-shadow var(--transition-fast);
}
.algo-card:hover { border-color: var(--color-primary-soft); }
.algo-card.active { border-color: var(--color-primary); box-shadow: 0 0 0 3px var(--color-primary-mute); }
.algo-name { font-size: 14px; font-weight: 600; }
.algo-desc { font-size: 12px; color: var(--color-text-secondary); margin-top: 4px; }
.file-list { display: flex; flex-direction: column; gap: 4px; }
.file-item {
  display: flex; align-items: center; gap: var(--space-sm); padding: 8px 12px;
  border: none; border-radius: var(--radius-sm); background: var(--color-surface);
  color: var(--color-text); font-size: 13px; cursor: pointer; text-align: left;
  transition: background var(--transition-fast);
}
.file-item:hover { background: var(--color-bg-soft); }
.file-item.active { background: var(--color-primary-soft); color: var(--color-primary); }
.run-bar { display: flex; justify-content: center; }
.run-btn {
  display: flex; align-items: center; gap: var(--space-sm); padding: 10px 32px;
  border: none; border-radius: var(--radius-md); background: var(--color-primary);
  color: #fff; font-size: 15px; cursor: pointer; transition: opacity var(--transition-fast);
}
.run-btn:disabled { opacity: 0.4; cursor: default; }
.run-btn:not(:disabled):hover { opacity: 0.9; }
.loading-area { display: flex; justify-content: center; }
.error-msg { font-size: 13px; color: var(--color-error); text-align: center; }
.result-msg { font-size: 13px; color: var(--color-text-secondary); white-space: pre-wrap; max-height: 200px; overflow-y: auto; margin-top: var(--space-sm); }
</style>
