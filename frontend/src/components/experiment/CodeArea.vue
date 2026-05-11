<script setup lang="ts">
import { useExperimentStore } from '@/stores/experiment'
import { Code, Play, Terminal, Maximize2, Minimize2, RefreshCw } from 'lucide-vue-next'

const store = useExperimentStore()
</script>

<template>
  <div class="code-area" :class="{ minimized: store.codeAreaMinimized, fullscreen: store.codeAreaFullscreen }">
    <div class="code-header">
      <div class="code-left">
        <div class="code-tabs">
          <button class="code-tab" :class="{ active: store.codeViewMode === 'json' }" @click="store.codeViewMode = 'json'">
            <Code :size="13" /> JSON
          </button>
          <button class="code-tab" :class="{ active: store.codeViewMode === 'python' }" @click="store.codeViewMode = 'python'">
            <Play :size="13" /> Python
          </button>
        </div>
        <span v-if="store.compileStatus === 'compiling'" class="compile-dot compiling" title="编译中...">●</span>
        <span v-else-if="store.compileStatus === 'error'" class="compile-dot error" title="编译失败">●</span>
      </div>
      <div class="code-actions">
        <button class="code-act-btn" title="手动编译为 Python" @click="store.compile()"><Terminal :size="13" /></button>
        <button class="code-act-btn" title="从 JSON 同步到实验步骤" @click="store.syncFromCode()"><RefreshCw :size="13" /></button>
        <button class="code-act-btn" title="全屏" @click="store.codeAreaFullscreen = !store.codeAreaFullscreen">
          <component :is="store.codeAreaFullscreen ? Minimize2 : Maximize2" :size="13" />
        </button>
        <button class="code-act-btn" title="最小化" @click="store.codeAreaMinimized = !store.codeAreaMinimized">
          {{ store.codeAreaMinimized ? '+' : '−' }}
        </button>
      </div>
    </div>

    <div v-if="!store.codeAreaMinimized" class="code-body">
      <!-- JSON editor -->
      <textarea
        v-if="store.codeViewMode === 'json'"
        v-model="store.editableJsonCode"
        class="code-editor"
        spellcheck="false"
        @focus="store.onJsonFocus()"
        @blur="store.onJsonBlur()"
      />

      <!-- Python editor -->
      <textarea
        v-else
        v-model="store.editablePythonCode"
        class="code-editor"
        spellcheck="false"
        @focus="store.onPyFocus()"
        @blur="store.onPyBlur()"
      />
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

.code-left { display: flex; align-items: center; gap: 8px; }

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

.compile-dot { font-size: 8px; line-height: 1; }
.compile-dot.compiling { color: #f59e0b; animation: blink 0.8s ease-in-out infinite; }
.compile-dot.error { color: #ef4444; }

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.2; }
}

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
  flex: 1; overflow: hidden; padding: var(--space-md);
}

.code-editor {
  width: 100%; height: 100%;
  background: transparent; border: none; outline: none; resize: none;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 12px; line-height: 1.6;
  color: rgba(255,255,255,0.8);
  tab-size: 2; white-space: pre-wrap; word-wrap: break-word;
}

.code-editor:focus {
  background: rgba(0,0,0,0.15);
}
</style>
