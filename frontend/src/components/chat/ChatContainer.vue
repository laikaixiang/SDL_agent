<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import { storeToRefs } from 'pinia'
import { useChatStore } from '@/stores/chat'
import { uploadPDF } from '@/api/extraction'
import MessageBubble from './MessageBubble.vue'
import InputBar from './InputBar.vue'

const store = useChatStore()
const { messages, isStreaming } = storeToRefs(store)
const inputText = ref('')
const chatEl = ref<HTMLDivElement>()

function scrollToBottom() {
  nextTick(() => {
    if (chatEl.value) chatEl.value.scrollTop = chatEl.value.scrollHeight
  })
}

watch(() => store.messages.length, scrollToBottom)

async function onSend(text: string) {
  inputText.value = ''
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
</script>

<template>
  <div class="chat-container">
    <div class="chat-messages" ref="chatEl">
      <MessageBubble
        v-for="(msg, i) in messages"
        :key="i"
        :role="msg.role"
        :content="msg.content"
        :timestamp="msg.timestamp"
      />
    </div>
    <div class="chat-input-area">
      <InputBar
        v-model="inputText"
        :disabled="isStreaming"
        :placeholder="isStreaming ? 'AI 回复中...' : '输入消息... (Enter 发送)'"
        @send="onSend"
        @file-selected="onFileSelected"
      />
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

.chat-messages {
  flex: 1;
  overflow-y: auto;
  width: 50%;
  min-width: 400px;
  padding-bottom: var(--space-md);
}

.chat-input-area {
  width: 50%;
  min-width: 400px;
  flex-shrink: 0;
}
</style>
