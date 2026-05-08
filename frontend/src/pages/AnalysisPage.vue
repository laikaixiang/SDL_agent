<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { useAnalysisStore } from '@/stores/analysis'
import { useLayoutStore } from '@/stores/layout'
import FileSelectorModal from '@/components/modals/FileSelectorModal.vue'
import ResultCard from '@/components/cards/ResultCard.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import Badge from '@/components/common/Badge.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import { BarChart3, FileText, Play, Plus, Sparkles, ChevronDown, ChevronRight } from 'lucide-vue-next'

const store = useAnalysisStore()
const layout = useLayoutStore()

const showFileSelector = ref(false)
const showDirSelector = ref(false)
const pendingFileAlgo = ref('')
const pendingDirAlgo = ref('')
const showGenerator = ref(false)
const genDesc = ref('')

onMounted(() => {
  store.loadAlgorithms()
  store.loadFiles()
})

watch(() => store.loading, (val) => {
  if (val) {
    layout.updateTaskStatus('analysis', 'running', 10)
  } else if (store.result) {
    layout.updateTaskStatus('analysis', 'completed')
  }
})

function onFileSelected(path: string, _name: string) {
  store.setInputFile(pendingFileAlgo.value, path)
  showFileSelector.value = false
}

function onDirSelected(path: string, _name: string) {
  store.setOutputDir(pendingDirAlgo.value, path)
  showDirSelector.value = false
}

function openFilePicker(algoName: string) {
  pendingFileAlgo.value = algoName
  showFileSelector.value = true
}

function openDirPicker(algoName: string) {
  pendingDirAlgo.value = algoName
  showDirSelector.value = true
}

async function onGenerate() {
  if (!genDesc.value.trim()) return
  await store.generateAlgorithm(genDesc.value.trim())
  showGenerator.value = false
  genDesc.value = ''
}
</script>

<template>
  <div class="analysis-page">
    <div class="page-header">
      <h2><BarChart3 :size="18" /> 数据分析</h2>
      <button class="gen-btn" @click="showGenerator = true">
        <Sparkles :size="14" /> 生成新算法
      </button>
    </div>

    <div class="page-body">
      <!-- Algorithms -->
      <section>
        <h3>可用算法 ({{ store.algorithms.length }})</h3>
        <div class="algo-list">
          <div
            v-for="a in store.algorithms"
            :key="a.name"
            class="algo-item"
            :class="{
              expanded: store.expandedAlgo === a.name,
              selected: store.selectedAlgo?.name === a.name,
            }"
          >
            <div class="algo-main" @click="store.toggleDetail(a.name)">
              <component :is="store.expandedAlgo === a.name ? ChevronDown : ChevronRight"
                :size="14" class="algo-arrow" />
              <div class="algo-info">
                <span class="algo-name">{{ a.chinese_name || a.name }}</span>
                <span class="algo-desc">{{ a.description }}</span>
              </div>
              <div class="algo-actions">
                <button
                  class="algo-act-btn"
                  title="添加到实验设计"
                  @click.stop="store.addToExperiment(a)"
                >
                  <Plus :size="14" />
                </button>
                <button
                  class="algo-run-btn"
                  :class="{ active: store.selectedAlgo?.name === a.name }"
                  title="选中此算法"
                  @click.stop="store.selectedAlgo = a"
                >
                  <Play :size="12" />
                </button>
              </div>
            </div>

            <div v-if="store.expandedAlgo === a.name" class="algo-detail">
              <p v-if="a.description" class="algo-detail-desc">{{ a.description }}</p>
              <div v-if="a.params_schema && Object.keys(a.params_schema).length" class="algo-params-schema">
                <span v-for="(v, k) in a.params_schema" :key="k" class="algo-tag">{{ k }}: {{ v }}</span>
              </div>
              <div class="algo-pickers">
                <div class="picker-row">
                  <span class="picker-label">输入文件</span>
                  <span class="picker-value" :class="{ set: store.algoInputFiles[a.name] }">
                    {{ store.algoInputFiles[a.name] ? store.algoInputFiles[a.name].split(/[/\\]/).pop() : '未选择' }}
                  </span>
                  <button class="picker-btn" @click="openFilePicker(a.name)">选择</button>
                </div>
                <div class="picker-row">
                  <span class="picker-label">输出目录</span>
                  <span class="picker-value" :class="{ set: store.algoOutputDirs[a.name] }">
                    {{ store.algoOutputDirs[a.name] || '默认' }}
                  </span>
                  <button class="picker-btn" @click="openDirPicker(a.name)">选择</button>
                </div>
              </div>
            </div>
          </div>
          <div v-if="!store.algorithms.length" class="algo-empty">
            <LoadingSpinner v-if="store.generating" :size="16" label="加载算法..." />
            <span v-else>暂无可用算法</span>
          </div>
        </div>
      </section>

      <!-- Files + Run -->
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
      </section>

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

      <div v-if="!store.algorithms.length && !store.result" class="body-center">
        <EmptyState title="数据分析" description="选择算法和 CSV 文件，运行分析" />
      </div>
    </div>

    <!-- File selector modal -->
    <FileSelectorModal
      :open="showFileSelector"
      title="选择输入文件"
      @update:open="showFileSelector = $event"
      @selected="onFileSelected"
    />

    <!-- Dir selector modal -->
    <FileSelectorModal
      :open="showDirSelector"
      title="选择输出目录"
      dir-mode
      @update:open="showDirSelector = $event"
      @selected="onDirSelected"
    />

    <!-- Algorithm generator -->
    <div v-if="showGenerator" class="gen-overlay" @click.self="showGenerator = false">
      <div class="gen-card">
        <h3>生成新算法</h3>
        <textarea
          v-model="genDesc"
          class="gen-input"
          placeholder="描述你需要的算法，例如：对数据进行正态分布拟合并计算置信区间"
          rows="4"
          autofocus
        />
        <div class="gen-actions">
          <button class="btn-cancel" @click="showGenerator = false">取消</button>
          <button class="btn-save" :disabled="store.generating" @click="onGenerate">
            {{ store.generating ? '生成中...' : '生成' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.analysis-page { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
.page-header { display: flex; align-items: center; justify-content: space-between; padding: var(--space-lg) var(--space-xl) 0; flex-shrink: 0; }
.page-header h2 { font-size: 18px; display: flex; align-items: center; gap: var(--space-sm); }
.gen-btn {
  display: flex; align-items: center; gap: 5px;
  padding: 6px 12px; border: 1px solid var(--color-primary);
  border-radius: var(--radius-sm); background: var(--color-primary-soft);
  color: var(--color-primary); font-size: 13px; cursor: pointer;
  transition: opacity var(--transition-fast);
}
.gen-btn:hover { opacity: 0.85; }
.page-body { flex: 1; overflow-y: auto; padding: var(--space-lg) var(--space-xl); display: flex; flex-direction: column; gap: var(--space-lg); }
.body-center { flex: 1; display: flex; align-items: center; justify-content: center; }
section h3 { font-size: 13px; color: var(--color-text-secondary); margin-bottom: var(--space-md); }

/* Algorithm list */
.algo-list { display: flex; flex-direction: column; gap: 2px; }
.algo-item { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-md); overflow: hidden; transition: border-color var(--transition-fast); }
.algo-item:hover { border-color: var(--color-primary-soft); }
.algo-item.expanded { border-color: var(--color-primary); }
.algo-item.selected { border-color: var(--color-primary); box-shadow: 0 0 0 2px var(--color-primary-mute); }

.algo-main {
  display: flex; align-items: center; gap: var(--space-sm);
  padding: var(--space-md); cursor: pointer; user-select: none;
}
.algo-arrow { color: var(--color-text-tertiary); flex-shrink: 0; }
.algo-info { flex: 1; min-width: 0; }
.algo-name { font-size: 14px; font-weight: 600; color: var(--color-text); display: block; }
.algo-desc { font-size: 12px; color: var(--color-text-secondary); margin-top: 2px; display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.algo-actions { display: flex; gap: 4px; flex-shrink: 0; }
.algo-act-btn {
  width: 30px; height: 30px; display: flex; align-items: center; justify-content: center;
  border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  background: var(--color-surface); color: var(--color-text-secondary); cursor: pointer;
}
.algo-act-btn:hover { background: var(--color-primary-soft); color: var(--color-primary); }
.algo-run-btn {
  width: 30px; height: 30px; display: flex; align-items: center; justify-content: center;
  border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  background: var(--color-surface); color: var(--color-text-secondary); cursor: pointer;
}
.algo-run-btn:hover, .algo-run-btn.active { background: var(--color-primary); color: #fff; border-color: var(--color-primary); }

.algo-detail { border-top: 1px solid var(--color-border); padding: var(--space-md); background: var(--color-bg-soft); }
.algo-detail-desc { font-size: 13px; color: var(--color-text-secondary); margin-bottom: var(--space-sm); }
.algo-params-schema { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: var(--space-sm); }
.algo-tag { font-size: 11px; padding: 2px 8px; background: var(--color-bg-mute); border-radius: var(--radius-full); color: var(--color-text-secondary); }
.algo-pickers { display: flex; flex-direction: column; gap: 6px; }
.picker-row { display: flex; align-items: center; gap: var(--space-sm); }
.picker-label { font-size: 12px; color: var(--color-text-secondary); width: 60px; flex-shrink: 0; }
.picker-value { font-size: 12px; color: var(--color-text-tertiary); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.picker-value.set { color: var(--color-text); }
.picker-btn {
  padding: 3px 10px; border: 1px solid var(--color-border);
  border-radius: var(--radius-sm); background: var(--color-surface);
  color: var(--color-text-secondary); font-size: 12px; cursor: pointer;
}
.picker-btn:hover { background: var(--color-primary-soft); color: var(--color-primary); border-color: var(--color-primary); }

.algo-empty { padding: var(--space-lg); text-align: center; color: var(--color-text-tertiary); font-size: 13px; }

/* Files */
.file-list { display: flex; flex-direction: column; gap: 4px; margin-bottom: var(--space-md); }
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

/* Generator modal */
.gen-overlay {
  position: fixed; inset: 0; z-index: 10001;
  background: rgba(0,0,0,0.4);
  display: flex; align-items: center; justify-content: center;
}
.gen-card {
  background: var(--color-surface); border-radius: var(--radius-lg);
  padding: var(--space-xl); width: 500px; max-width: 90vw;
  box-shadow: var(--shadow-lg);
}
.gen-card h3 { font-size: 16px; margin-bottom: var(--space-lg); }
.gen-input {
  width: 100%; padding: var(--space-md);
  border: 1px solid var(--color-border); border-radius: var(--radius-md);
  font-size: 14px; font-family: inherit; resize: vertical;
}
.gen-input:focus { outline: none; border-color: var(--color-primary); }
.gen-actions { display: flex; gap: var(--space-sm); justify-content: flex-end; margin-top: var(--space-lg); }
.btn-cancel, .btn-save { padding: 8px 20px; border: none; border-radius: var(--radius-sm); font-size: 14px; cursor: pointer; }
.btn-cancel { background: var(--color-bg-mute); color: var(--color-text); }
.btn-save { background: var(--color-primary); color: #fff; }
.btn-save:disabled { opacity: 0.5; cursor: default; }
.btn-save:not(:disabled):hover { opacity: 0.9; }
</style>
