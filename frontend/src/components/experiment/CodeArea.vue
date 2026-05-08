<script setup lang="ts">
import { useExperimentStore } from '@/stores/experiment'
import { Code, Play, Terminal, Maximize2, Minimize2, RefreshCw } from 'lucide-vue-next'

const store = useExperimentStore()
</script>

<template>
  <div class="code-area" :class="{ minimized: store.codeAreaMinimized, fullscreen: store.codeAreaFullscreen }">
    <div class="code-header">
      <div class="code-tabs">
        <button class="code-tab" :class="{ active: store.codeViewMode === 'json' }" @click="store.codeViewMode = 'json'">
          <Code :size="13" /> JSON
        </button>
        <button class="code-tab" :class="{ active: store.codeViewMode === 'python' }" @click="store.codeViewMode = 'python'">
          <Play :size="13" /> Python
        </button>
      </div>
      <div class="code-actions">
        <button class="code-act-btn" title="编译为 Python" @click="store.compile()"><Terminal :size="13" /></button>
        <button class="code-act-btn" title="从代码同步" @click="store.syncFromCode()"><RefreshCw :size="13" /></button>
        <button class="code-act-btn" title="全屏" @click="store.codeAreaFullscreen = !store.codeAreaFullscreen">
          <component :is="store.codeAreaFullscreen ? Minimize2 : Maximize2" :size="13" />
        </button>
        <button class="code-act-btn" title="最小化" @click="store.codeAreaMinimized = !store.codeAreaMinimized">
          {{ store.codeAreaMinimized ? '+' : '−' }}
        </button>
      </div>
    </div>

    <div v-if="!store.codeAreaMinimized" class="code-body">
      <!-- JSON view -->
      <pre v-if="store.codeViewMode === 'json'" class="code-content"><code>{{ store.jsonCode }}</code></pre>

      <!-- Python view -->
      <pre v-else class="code-content"><code>{{ store.pythonCode || '点击 ⚡ 编译生成 Python 代码' }}</code></pre>
    </div>
  </div>
</template>

<style scoped>
.code-area {
  border-top: 1px solid var(--color-border);
  background: #1e1e2e;
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  height: 220px;
  transition: height var(--transition-normal);
}

.code-area.minimized {
  height: 36px;
}

.code-area.fullscreen {
  position: fixed;
  inset: 0;
  z-index: 10000;
  height: auto !important;
}

.code-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 8px;
  height: 36px;
  border-bottom: 1px solid rgba(255,255,255,0.06);
  flex-shrink: 0;
}

.code-tabs { display: flex; gap: 0; }

.code-tab {
  display: flex; align-items: center; gap: 4px;
  padding: 6px 12px; border: none; background: transparent;
  color: rgba(255,255,255,0.45); font-size: 12px; cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: color var(--transition-fast), border-color var(--transition-fast);
}

.code-tab:hover { color: rgba(255,255,255,0.7); }
.code-tab.active { color: #7c3aed; border-bottom-color: #7c3aed; }

.code-actions { display: flex; gap: 2px; }

.code-act-btn {
  width: 28px; height: 28px;
  display: flex; align-items: center; justify-content: center;
  border: none; border-radius: var(--radius-sm); background: transparent;
  color: rgba(255,255,255,0.35); font-size: 14px; cursor: pointer;
  transition: background var(--transition-fast), color var(--transition-fast);
}

.code-act-btn:hover { background: rgba(255,255,255,0.08); color: rgba(255,255,255,0.8); }

.code-body {
  flex: 1; overflow: auto; padding: var(--space-md);
}

.code-content {
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 12px; line-height: 1.6;
  color: rgba(255,255,255,0.8);
  white-space: pre-wrap;
  margin: 0;
}
</style>
