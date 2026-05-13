<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import { storeToRefs } from 'pinia'
import { useChatStore } from '@/stores/chat'
import { useAnalysisStore } from '@/stores/analysis'
import { useSearchStore } from '@/stores/search'
import { uploadPDF } from '@/api/chat'
import MessageBubble from './MessageBubble.vue'
import InputBar from './InputBar.vue'
import PdfViewer from './PdfViewer.vue'
import { Plus, X } from 'lucide-vue-next'

const store = useChatStore()
const analysisStore = useAnalysisStore()
const searchStore = useSearchStore()
const { messages, isStreaming, fieldConfirm } = storeToRefs(store)
const { showGuide, guideReply, guideProgress, guideDone, generating } = storeToRefs(analysisStore)
const { pdfPanelOpen, viewPdfFile } = storeToRefs(searchStore)

// DEBUG: track PDF panel state
watch(pdfPanelOpen, (val) => {
  console.log('[ChatContainer] pdfPanelOpen changed:', val, 'viewPdfFile:', JSON.stringify(viewPdfFile.value))
})
const inputText = ref('')
const chatEl = ref<HTMLDivElement>()
const newFieldName = ref('')
const editingFieldIndex = ref<number | null>(null)
const editFieldValue = ref('')

function startEditField(index: number, current: string) {
  editingFieldIndex.value = index
  editFieldValue.value = current
}

function saveEditField() {
  if (editingFieldIndex.value !== null && editFieldValue.value.trim()) {
    store.updateConfirmField(editingFieldIndex.value, editFieldValue.value.trim())
  }
  editingFieldIndex.value = null
  editFieldValue.value = ''
}

function addField() {
  if (newFieldName.value.trim()) {
    store.addConfirmField(newFieldName.value.trim())
    newFieldName.value = ''
  }
}

function scrollToBottom() {
  nextTick(() => {
    if (chatEl.value) chatEl.value.scrollTop = chatEl.value.scrollHeight
  })
}

watch(() => store.messages.length, scrollToBottom)

async function onSend(text: string) {
  inputText.value = ''
  // 引导模式：路由到算法引导流程
  if (showGuide.value && !guideDone.value) {
    store.addMessage('user', text || '（跳过）')
    await analysisStore.submitGuideAnswer(text)
    scrollToBottom()
    return
  }
  await store.send(text)
  scrollToBottom()
}

async function onFileSelected(file: File) {
  try {
    const result = await uploadPDF(file)
    if (result.success) {
      inputText.value = `帮我搜寻：${result.filename}`
    }
  } catch (err) {
    console.error('文件上传失败:', err)
  }
}

async function onCancelExtraction() {
  await store.cancelExtractionTask()
}
</script>

<template>
  <div class="chat-container" :class="{ 'has-pdf-panel': pdfPanelOpen }">
    <div class="chat-body">
      <div class="chat-main">
        <div class="chat-messages" ref="chatEl">
          <MessageBubble
            v-for="(msg, i) in messages"
            :key="i"
            :role="msg.role"
            :content="msg.content"
            :timestamp="msg.timestamp"
          />

          <!-- Algorithm guide card -->
          <div v-if="showGuide" class="guide-card">
            <div class="guide-progress-bar">
              <div class="guide-progress-fill"
                :style="{ width: guideProgress === 'complete' ? '100%' : (parseInt(guideProgress) / 4 * 100) + '%' }">
              </div>
            </div>
            <div class="guide-progress-label">{{ guideProgress === 'complete' ? '完成' : guideProgress }}</div>
            <div class="guide-reply">{{ guideReply }}</div>
            <div v-if="!guideDone" class="guide-actions">
              <button class="btn-guide-cancel" @click="analysisStore.cancelGuide()">取消</button>
              <button class="btn-guide-back" @click="analysisStore.guideGoBack()">返回</button>
              <button class="btn-guide-submit" :disabled="generating" @click="onSend(inputText)">提交</button>
            </div>
          </div>

          <!-- Inline field confirm card -->
          <div v-if="fieldConfirm" class="confirm-card">
            <div class="confirm-label">LLM 推断的提取字段，可编辑后确认：</div>
            <div class="confirm-fields">
              <div v-for="(f, i) in fieldConfirm.fields" :key="i" class="field-tag-row">
                <!-- Display mode -->
                <template v-if="editingFieldIndex !== i">
                  <span class="field-tag" @dblclick="startEditField(i, f)">{{ f }}</span>
                  <button class="field-del" title="删除" @click="store.removeConfirmField(i)"><X :size="12" /></button>
                </template>
                <!-- Edit mode -->
                <template v-else>
                  <input
                    v-model="editFieldValue"
                    class="field-edit-input"
                    @keydown.enter="saveEditField()"
                    @keydown.escape="editingFieldIndex = null"
                    @blur="saveEditField()"
                  />
                </template>
              </div>
            </div>
            <div class="confirm-add-row">
              <input
                v-model="newFieldName"
                class="field-add-input"
                placeholder="添加新字段..."
                @keydown.enter="addField()"
              />
              <button class="field-add-btn" :disabled="!newFieldName.trim()" @click="addField()"><Plus :size="14" /></button>
            </div>
            <div class="confirm-actions">
              <button class="confirm-btn-yes" @click="store.confirmExtraction()">✅ 确认提取</button>
              <button class="confirm-btn-no" @click="store.cancelExtraction()">❌ 修改要求</button>
            </div>
          </div>
        </div>
        <div class="chat-input-area">
          <InputBar
            v-model="inputText"
            :disabled="isStreaming"
            :placeholder="isStreaming ? 'AI 回复中...' : '输入消息... (Enter 发送)'"
            @send="onSend"
            @file-selected="onFileSelected"
            @cancel-extraction="onCancelExtraction"
          />
        </div>
      </div>

      <!-- PDF side panel -->
      <div v-if="pdfPanelOpen && viewPdfFile" class="pdf-side-panel">
        <div class="pdf-panel-header">
          <button class="pdf-panel-close" title="关闭PDF预览" @click="searchStore.closePdfViewer()">×</button>
        </div>
        <PdfViewer
          :pdf-path="viewPdfFile.pdfPath"
          :filename="viewPdfFile.filename"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
.chat-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  align-items: center;
}

.chat-container.has-pdf-panel {
  align-items: stretch;
}

.chat-body {
  display: flex;
  flex: 1;
  overflow: hidden;
  width: 100%;
}

.chat-main {
  flex: 1;
  min-width: 400px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.chat-container:not(.has-pdf-panel) .chat-main {
  align-items: center;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding-bottom: var(--space-md);
}

.chat-container:not(.has-pdf-panel) .chat-messages {
  width: 50%;
  min-width: 400px;
}

.chat-container.has-pdf-panel .chat-messages {
  width: 100%;
  padding-left: var(--space-xl);
  padding-right: var(--space-xl);
}

.chat-input-area {
  flex-shrink: 0;
}

.chat-container:not(.has-pdf-panel) .chat-input-area {
  width: 50%;
  min-width: 400px;
}

.chat-container.has-pdf-panel .chat-input-area {
  width: 100%;
  padding: 0 var(--space-xl);
}

/* PDF side panel */
.pdf-side-panel {
  width: 520px;
  flex-shrink: 0;
  border-left: 2px solid #404040;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.pdf-panel-header {
  display: flex;
  justify-content: flex-end;
  padding: 4px 8px;
  background: #2d2d2d;
  flex-shrink: 0;
}
.pdf-panel-close {
  background: none;
  border: none;
  color: #999;
  font-size: 20px;
  cursor: pointer;
  padding: 2px 6px;
  line-height: 1;
  border-radius: 4px;
}
.pdf-panel-close:hover {
  background: #444;
  color: #fff;
}

/* Inline field confirm card */
.confirm-card {
  margin: var(--space-sm) 0;
  padding: var(--space-md);
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
}
.confirm-label { font-size: 12px; color: var(--color-text-tertiary); margin-bottom: 8px; }
.confirm-fields {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-bottom: var(--space-sm);
}
.field-tag-row {
  display: flex;
  align-items: center;
  gap: 1px;
  background: var(--color-primary-soft);
  border-radius: var(--radius-full);
  padding-left: 10px;
}
.field-tag {
  font-size: 12px;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  color: var(--color-primary);
  cursor: pointer;
}
.field-tag:hover { text-decoration: underline; }
.field-del {
  width: 20px; height: 20px; display: flex; align-items: center; justify-content: center;
  border: none; background: transparent; color: var(--color-primary); cursor: pointer;
  border-radius: 50%; opacity: 0.6; padding: 0;
}
.field-del:hover { opacity: 1; background: rgba(0,0,0,0.05); }
.field-edit-input {
  width: 120px; padding: 2px 8px; border: 1px solid var(--color-primary);
  border-radius: var(--radius-full); font-size: 12px; outline: none;
  background: var(--color-surface);
}
.confirm-add-row {
  display: flex; gap: 4px; margin-bottom: var(--space-md); align-items: center;
}
.field-add-input {
  flex: 1; padding: 4px 10px; border: 1px dashed var(--color-border);
  border-radius: var(--radius-sm); font-size: 12px; outline: none;
  background: transparent; color: var(--color-text);
}
.field-add-input:focus { border-color: var(--color-primary); border-style: solid; }
.field-add-btn {
  width: 26px; height: 26px; display: flex; align-items: center; justify-content: center;
  border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  background: var(--color-surface); color: var(--color-text-secondary); cursor: pointer;
}
.field-add-btn:disabled { opacity: 0.4; cursor: default; }
.field-add-btn:not(:disabled):hover { background: var(--color-bg-soft); color: var(--color-text); }
.confirm-actions {
  display: flex;
  gap: var(--space-sm);
}
.confirm-btn-yes,
.confirm-btn-no {
  padding: 6px 16px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-size: 13px;
  cursor: pointer;
  transition: all var(--transition-fast);
}
.confirm-btn-yes {
  background: var(--color-primary);
  color: #fff;
  border-color: var(--color-primary);
}
.confirm-btn-yes:hover { opacity: 0.85; }
.confirm-btn-no {
  background: var(--color-surface);
  color: var(--color-text-secondary);
}
.confirm-btn-no:hover { background: var(--color-bg-soft); color: var(--color-text); }

/* Algorithm guide card */
.guide-card {
  margin: var(--space-sm) 0;
  padding: var(--space-lg);
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
  border-radius: var(--radius-lg);
}
.guide-progress-bar {
  width: 100%;
  height: 6px;
  background: #e5e7eb;
  border-radius: 3px;
  margin-bottom: 8px;
  overflow: hidden;
}
.guide-progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #10b981, #34d399);
  border-radius: 3px;
  transition: width 0.4s ease;
}
.guide-progress-label {
  font-size: 12px;
  color: #059669;
  font-weight: 600;
  margin-bottom: 12px;
  text-align: right;
}
.guide-reply {
  font-size: 14px;
  color: var(--color-text);
  line-height: 1.7;
  white-space: pre-wrap;
}
.guide-actions {
  display: flex;
  gap: var(--space-sm);
  justify-content: flex-end;
  margin-top: var(--space-md);
}
.btn-guide-cancel, .btn-guide-back, .btn-guide-submit {
  padding: 6px 16px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-size: 13px;
  cursor: pointer;
  transition: all var(--transition-fast);
}
.btn-guide-cancel { background: var(--color-bg-mute); color: var(--color-text-secondary); }
.btn-guide-cancel:hover { background: #e5e7eb; color: var(--color-text); }
.btn-guide-back { background: #fef3c7; color: #92400e; border-color: #fcd34d; }
.btn-guide-back:hover { background: #fde68a; }
.btn-guide-submit { background: #10b981; color: #fff; border-color: #10b981; font-weight: 600; }
.btn-guide-submit:hover { background: #059669; }
.btn-guide-submit:disabled { opacity: 0.5; cursor: default; }

</style>
