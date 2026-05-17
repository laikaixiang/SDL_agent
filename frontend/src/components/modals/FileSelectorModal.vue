<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { ref, watch } from 'vue'
import { X, FileText, Folder, Upload } from 'lucide-vue-next'

const { t } = useI18n()

const props = defineProps<{
  open: boolean
  title?: string
  dirMode?: boolean
}>()

const emit = defineEmits<{
  (e: 'update:open', val: boolean): void
  (e: 'selected', path: string, name: string): void
}>()

const activeTab = ref<'recent' | 'browse' | 'custom'>('recent')
const recentFiles = ref<{ path: string; name: string; mtime: string }[]>([])
const dirs = ref<{ path: string; name: string }[]>([])
const customPath = ref('')
const loading = ref(false)

watch(() => props.open, (val) => {
  if (val) {
    if (props.dirMode) {
      loadDirs()
    } else {
      loadRecent()
    }
  }
})

async function loadRecent() {
  loading.value = true
  try {
    const resp = await fetch('/api/recent_files')
    const data = await resp.json()
    recentFiles.value = (data.files || []).slice(0, 20)
  } catch {
    recentFiles.value = []
  } finally {
    loading.value = false
  }
}

async function loadDirs() {
  loading.value = true
  try {
    const resp = await fetch('/api/browse_output_dirs')
    const data = await resp.json()
    dirs.value = data.dirs || []
  } catch {
    dirs.value = []
  } finally {
    loading.value = false
  }
}

function selectItem(path: string, name: string) {
  emit('selected', path, name)
  close()
}

function confirmCustom() {
  if (!customPath.value.trim()) return
  const name = customPath.value.split(/[/\\]/).pop() || customPath.value
  selectItem(customPath.value.trim(), name)
}

function close() {
  emit('update:open', false)
}

async function onUpload(e: Event) {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (!file) return
  try {
    const formData = new FormData()
    formData.append('file', file)
    const resp = await fetch('/api/upload', { method: 'POST', body: formData })
    const data = await resp.json()
    if (data.filename) {
      selectItem(data.filename, file.name)
    }
  } catch {
    // silent
  }
}
</script>

<template>
  <div v-if="open" class="fs-overlay" @click.self="close">
    <div class="fs-modal">
      <div class="fs-header">
        <h3>{{ title || $t(dirMode ? 'modals.selectOutputDir' : 'modals.selectDataFile') }}</h3>
        <button class="fs-close" @click="close"><X :size="16" /></button>
      </div>

      <div v-if="!dirMode" class="fs-tabs">
        <button class="fs-tab" :class="{ active: activeTab === 'recent' }" @click="activeTab = 'recent'; loadRecent()">{{ $t('modals.recentlyUsed') }}</button>
        <button class="fs-tab" :class="{ active: activeTab === 'browse' }" @click="activeTab = 'browse'">{{ $t('modals.uploadFile') }}</button>
        <button class="fs-tab" :class="{ active: activeTab === 'custom' }" @click="activeTab = 'custom'">{{ $t('modals.customPath') }}</button>
      </div>

      <div class="fs-body">
        <!-- Recent files -->
        <div v-if="activeTab === 'recent' && !dirMode" class="fs-list">
          <div v-if="loading" class="fs-loading">{{ $t('common.loading') }}</div>
          <button
            v-for="f in recentFiles"
            :key="f.path"
            class="fs-item"
            @click="selectItem(f.path, f.name)"
          >
            <FileText :size="14" class="fs-icon" />
            <div class="fs-info">
              <span class="fs-name">{{ f.name }}</span>
              <span class="fs-path">{{ f.path }}</span>
            </div>
          </button>
          <div v-if="!loading && !recentFiles.length" class="fs-empty">{{ $t('modals.noRecentFiles') }}</div>
        </div>

        <!-- Upload -->
        <div v-if="activeTab === 'browse' && !dirMode" class="fs-upload">
          <label class="fs-upload-area">
            <Upload :size="24" />
            <span>{{ $t('modals.clickToUploadCsv') }}</span>
            <input type="file" accept=".csv,.xlsx,.xls" hidden @change="onUpload" />
          </label>
        </div>

        <!-- Custom path -->
        <div v-if="activeTab === 'custom' && !dirMode" class="fs-custom">
          <input v-model="customPath" class="fs-input" :placeholder="$t('modals.enterFilePath')" @keyup.enter="confirmCustom" />
          <button class="fs-confirm-btn" @click="confirmCustom">{{ $t('common.confirm') }}</button>
        </div>

        <!-- Dir mode -->
        <div v-if="dirMode" class="fs-list">
          <div v-if="loading" class="fs-loading">{{ $t('common.loading') }}</div>
          <button
            v-for="d in dirs"
            :key="d.path"
            class="fs-item"
            @click="selectItem(d.path, d.name)"
          >
            <Folder :size="14" class="fs-icon" />
            <div class="fs-info">
              <span class="fs-name">{{ d.name }}</span>
              <span class="fs-path">{{ d.path }}</span>
            </div>
          </button>
          <div v-if="!loading && !dirs.length" class="fs-empty">{{ $t('modals.noAvailableDirs') }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.fs-overlay {
  position: fixed; inset: 0; z-index: 10000;
  background: rgba(0,0,0,0.35);
  display: flex; align-items: center; justify-content: center;
}
.fs-modal {
  background: var(--color-surface); border-radius: var(--radius-lg);
  width: 520px; max-width: 90vw; max-height: 70vh;
  display: flex; flex-direction: column;
  box-shadow: var(--shadow-lg);
}
.fs-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: var(--space-lg) var(--space-xl);
  border-bottom: 1px solid var(--color-border);
}
.fs-header h3 { font-size: 16px; }
.fs-close {
  width: 28px; height: 28px; display: flex; align-items: center; justify-content: center;
  border: none; border-radius: var(--radius-sm); background: transparent;
  color: var(--color-text-tertiary); cursor: pointer;
}
.fs-close:hover { background: var(--color-bg-soft); }

.fs-tabs { display: flex; border-bottom: 1px solid var(--color-border); padding: 0 var(--space-lg); }
.fs-tab {
  padding: 10px 16px; border: none; background: transparent;
  color: var(--color-text-secondary); font-size: 13px; cursor: pointer;
  border-bottom: 2px solid transparent; transition: color var(--transition-fast), border-color var(--transition-fast);
}
.fs-tab:hover { color: var(--color-text); }
.fs-tab.active { color: var(--color-primary); border-bottom-color: var(--color-primary); }

.fs-body { flex: 1; overflow-y: auto; padding: var(--space-md); }
.fs-list { display: flex; flex-direction: column; gap: 2px; }
.fs-loading, .fs-empty { padding: var(--space-xl); text-align: center; color: var(--color-text-tertiary); font-size: 13px; }

.fs-item {
  display: flex; align-items: center; gap: var(--space-sm);
  padding: 10px 12px; border: none; border-radius: var(--radius-sm);
  background: transparent; cursor: pointer; text-align: left; width: 100%;
  transition: background var(--transition-fast);
}
.fs-item:hover { background: var(--color-bg-soft); }

.fs-icon { color: var(--color-primary); flex-shrink: 0; }
.fs-info { flex: 1; min-width: 0; }
.fs-name { font-size: 14px; color: var(--color-text); display: block; }
.fs-path { font-size: 11px; color: var(--color-text-tertiary); display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.fs-upload { padding: var(--space-2xl); }
.fs-upload-area {
  display: flex; flex-direction: column; align-items: center; gap: var(--space-md);
  padding: var(--space-2xl); border: 2px dashed var(--color-border);
  border-radius: var(--radius-md); cursor: pointer; color: var(--color-text-secondary);
  transition: border-color var(--transition-fast), color var(--transition-fast);
}
.fs-upload-area:hover { border-color: var(--color-primary); color: var(--color-primary); }

.fs-custom { display: flex; flex-direction: column; gap: var(--space-md); padding: var(--space-md) 0; }
.fs-input {
  padding: 10px 12px; border: 1px solid var(--color-border);
  border-radius: var(--radius-sm); font-size: 14px; width: 100%;
}
.fs-input:focus { outline: none; border-color: var(--color-primary); }
.fs-confirm-btn {
  padding: 8px 20px; border: none; border-radius: var(--radius-sm);
  background: var(--color-primary); color: #fff; font-size: 14px; cursor: pointer;
  align-self: flex-end;
}
.fs-confirm-btn:hover { opacity: 0.9; }
</style>
