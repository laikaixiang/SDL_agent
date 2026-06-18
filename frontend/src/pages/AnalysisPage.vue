<script setup lang="ts">
import { ref, onMounted, watch, computed } from 'vue'
import { useAnalysisStore } from '@/stores/analysis'
import { useLayoutStore } from '@/stores/layout'
import FileSelectorModal from '@/components/modals/FileSelectorModal.vue'
import ResultCard from '@/components/cards/ResultCard.vue'
import ResultRenderer from '@/components/result/ResultRenderer.vue'
import ParamForm from '@/components/result/ParamForm.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import Badge from '@/components/common/Badge.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import { BarChart3, FileText, Play, Plus, Sparkles, ChevronDown, ChevronRight } from 'lucide-vue-next'
import type { Algorithm, PreviewData } from '@/types/analysis'

const store = useAnalysisStore()
const layout = useLayoutStore()

const showFileSelector = ref(false)
const showDirSelector = ref(false)
const pendingFileAlgo = ref('')
const pendingDirAlgo = ref('')

// 每个算法的 params 状态: algo.name -> params dict
const algoParams = ref<Record<string, Record<string, unknown>>>({})

// 当前选中文件对应的列名 (ParamForm type=columns 用)
const selectedFileColumns = ref<string[]>([])

onMounted(async () => {
  await store.loadAlgorithms()
  await store.loadFiles()
})

watch(() => store.loading, (val) => {
  if (val) {
    layout.updateTaskStatus('analysis', 'running', 10)
  } else if (store.result) {
    layout.updateTaskStatus('analysis', 'completed')
  }
})

// 选中文件变化时, 加载列名 (供 ParamForm type=columns 用)
watch(() => store.selectedFile, async (path) => {
  if (!path) {
    selectedFileColumns.value = []
    return
  }
  // 从 previewCache 取, 没有就拉一次
  const cacheKey = `${path}::n=20`
  let preview: PreviewData | undefined = store.previewCache[cacheKey]
  if (!preview) {
    preview = await store.loadPreview(path, 20) || undefined
  }
  if (preview) {
    selectedFileColumns.value = preview.columns.map(c => c.name)
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

// 切换算法时初始化 params 默认值
function onAlgoSelected(algo: Algorithm) {
  store.selectedAlgo = algo
  if (!algoParams.value[algo.name] && algo.params_schema) {
    const initial: Record<string, unknown> = {}
    for (const [k, def] of Object.entries(algo.params_schema)) {
      if (def.default !== undefined) initial[k] = def.default
    }
    algoParams.value[algo.name] = initial
  }
}

const currentAlgoResult = computed(() => {
  if (!store.result) return null
  // 兼容 result 的 message 字段可能含 raw result JSON 字符串
  return {
    success: true,
    result: (store.result as { result?: unknown }).result ?? parseMessage(store.result.message),
    message: store.result.message,
  }
})

function parseMessage(msg: string): unknown {
  // 后端返回的 result.message 可能是 JSON 字符串 (来自 run_algorithm_with_file 的 SSE complete)
  try {
    return JSON.parse(msg)
  } catch {
    return msg
  }
}

const currentSchema = computed(() => {
  return store.selectedAlgo?.result_schema
})

const runnable = computed(() => {
  return store.selectedAlgo && store.selectedFile && !store.loading
})
</script>

<template>
  <div class="analysis-page">
    <div class="page-header">
      <h2><BarChart3 :size="18" /> {{ $t('modes.dataAnalysis') }}</h2>
      <button class="gen-btn" @click="store.startGuide()" :disabled="store.generating">
        <Sparkles :size="14" /> {{ store.generating && store.showGuide ? $t('analysis.guidedMode') : $t('analysis.generateNewAlgorithm') }}
      </button>
    </div>

    <div class="page-body">
      <!-- Algorithms -->
      <section>
        <h3>{{ $t('analysis.availableAlgorithms') }} ({{ store.algorithms.length }})</h3>
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
                  :title="$t('analysis.addToExperiment')"
                  @click.stop="store.addToExperiment(a)"
                >
                  <Plus :size="14" />
                </button>
                <button
                  class="algo-run-btn"
                  :class="{ active: store.selectedAlgo?.name === a.name }"
                  :title="$t('analysis.selectAlgorithm')"
                  @click.stop="onAlgoSelected(a)"
                >
                  <Play :size="12" />
                </button>
              </div>
            </div>

            <div v-if="store.expandedAlgo === a.name" class="algo-detail">
              <p v-if="a.description" class="algo-detail-desc">{{ a.description }}</p>
              <div v-if="a.params_schema && Object.keys(a.params_schema).length" class="algo-params">
                <h4 class="algo-sub-title">参数</h4>
                <ParamForm
                  v-model="algoParams[a.name]"
                  :schema="a.params_schema"
                  :column-list="selectedFileColumns"
                />
              </div>
              <div class="algo-pickers">
                <div class="picker-row">
                  <span class="picker-label">{{ $t('analysis.inputFile') }}</span>
                  <span class="picker-value" :class="{ set: store.algoInputFiles[a.name] }">
                    {{ store.algoInputFiles[a.name] ? store.algoInputFiles[a.name].split(/[/\\]/).pop() : $t('analysis.notSelected') }}
                  </span>
                  <button class="picker-btn" @click="openFilePicker(a.name)">{{ $t('analysis.select') }}</button>
                </div>
                <div class="picker-row">
                  <span class="picker-label">{{ $t('analysis.outputDir') }}</span>
                  <span class="picker-value" :class="{ set: store.algoOutputDirs[a.name] }">
                    {{ store.algoOutputDirs[a.name] || $t('analysis.defaultOutput') }}
                  </span>
                  <button class="picker-btn" @click="openDirPicker(a.name)">{{ $t('analysis.select') }}</button>
                </div>
              </div>
            </div>
          </div>
          <div v-if="!store.algorithms.length" class="algo-empty">
            <LoadingSpinner v-if="store.generating" :size="16" :label="$t('analysis.loadingAlgorithms')" />
            <span v-else>{{ $t('analysis.noAlgorithmsAvailable') }}</span>
          </div>
        </div>
      </section>

      <!-- Files + Run -->
      <section>
        <h3>{{ $t('analysis.csvFiles') }}</h3>
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
            :disabled="!runnable"
            @click="store.run()"
          >
            <Play :size="16" />
            <span>{{ store.selectedAlgo?.chinese_name ? $t('analysis.runNamed', { name: store.selectedAlgo.chinese_name }) : $t('analysis.runAnalysis') }}</span>
          </button>
        </div>
      </section>

      <!-- Progress -->
      <div v-if="store.loading" class="loading-area">
        <LoadingSpinner :size="24" :label="$t('analysis.analyzing')" />
      </div>

      <!-- Error -->
      <div v-if="store.error" class="error-msg">{{ store.error }}</div>

      <!-- Result: 有 schema → ResultRenderer 直接渲染 -->
      <div v-if="currentAlgoResult && currentSchema && Object.keys(currentSchema).length > 0">
        <ResultRenderer
          :algo-result="currentAlgoResult"
          :schema="currentSchema"
        />
      </div>
      <!-- Result: 无 schema → fallback to ResultCard -->
      <div v-else-if="currentAlgoResult">
        <ResultCard
          :title="$t('analysis.analysisComplete')"
          :subtitle="$t('analysis.outputPrefix') + (store.result?.output_path || '')"
        >
          <pre class="result-raw">{{ JSON.stringify(currentAlgoResult.result ?? currentAlgoResult.message, null, 2) }}</pre>
        </ResultCard>
      </div>

      <div v-if="!store.algorithms.length && !store.result" class="body-center">
        <EmptyState :title="$t('modes.dataAnalysis')" :description="$t('analysis.analysisHint')" />
      </div>
    </div>

    <!-- File selector modal -->
    <FileSelectorModal
      :open="showFileSelector"
      :title="$t('analysis.selectInputFile')"
      @update:open="showFileSelector = $event"
      @selected="onFileSelected"
    />

    <!-- Dir selector modal -->
    <FileSelectorModal
      :open="showDirSelector"
      :title="$t('analysis.selectOutputDir')"
      dir-mode
      @update:open="showDirSelector = $event"
      @selected="onDirSelected"
    />

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
.algo-sub-title { font-size: 12px; font-weight: 600; color: var(--color-text-secondary); margin: 0 0 6px 0; }
.algo-params { margin-bottom: var(--space-md); }
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
.result-raw {
  background: var(--color-bg-soft);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  padding: var(--space-md);
  font-size: 12px;
  overflow-x: auto;
  margin: 0;
  max-height: 400px;
  overflow-y: auto;
}
</style>
