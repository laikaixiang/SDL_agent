<script setup lang="ts">
import { computed } from 'vue'
import { markdownToHtml } from '@/utils/markdown'
import ThinkingBlock from './ThinkingBlock.vue'
import type { ToolCallInfo, TeamAgentInfo, AgentQuestion } from '@/types/chat'

const props = defineProps<{
  role: 'user' | 'ai'
  content: string
  thinking?: string
  thinkingDuration?: number
  timestamp?: string
  // Agent attachments (kept visible after the AI's final reply)
  toolCalls?: ToolCallInfo[]
  pendingQuestion?: AgentQuestion | null
  // 系统消息 (压缩 / 超时总结 / info)
  systemNote?: {
    kind: 'compaction' | 'timeout_summary' | 'info'
    text: string
  }
  teamAgents?: TeamAgentInfo[]
}>()

const emit = defineEmits<{
  'question-select': [answer: string]
}>()

const renderedContent = computed(() => {
  if (props.role === 'ai' && props.content) {
    return markdownToHtml(props.content)
  }
  return ''
})

// 解析 options 字符串为数组
const parsedOptions = computed<string[]>(() => {
  if (!props.pendingQuestion?.options) return []
  try {
    const parsed = JSON.parse(props.pendingQuestion.options)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
})

// 工具调用 — 紧凑的对话内联注释(不再使用独立卡片)
function summarizeArgs(args: Record<string, unknown>): string {
  if (!args) return ''
  const entries = Object.entries(args).slice(0, 2)
  return entries.map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`).join(', ')
}

const toolCallLines = computed(() => {
  if (!props.toolCalls) return []
  return props.toolCalls
    .filter(tc => tc.status === 'running' || tc.result !== undefined)
    .map(tc => {
      const icon = tc.status === 'running' ? '⏳'
                 : tc.status === 'error' ? '✗'
                 : '✓'
      return {
        key: `tc-${tc.index}-${tc.name}`,
        text: `${icon} ${tc.name}`,
        status: tc.status,
        args: summarizeArgs(tc.arguments),
        result: tc.result && tc.result.length > 200
          ? tc.result.slice(0, 200) + '...'
          : tc.result,
      }
    })
})
</script>

<template>
  <div class="msg-row" :class="role">
    <ThinkingBlock
      v-if="role === 'ai' && thinking"
      :content="thinking"
      :duration="thinkingDuration"
      class="msg-thinking"
    />
    <div v-if="systemNote" class="system-note" :class="`system-note--${systemNote.kind}`">
      <span class="system-note__icon">
        {{ systemNote.kind === 'compaction' ? '🔄' : systemNote.kind === 'timeout_summary' ? '📋' : 'ℹ️' }}
      </span>
      <span class="system-note__text">{{ systemNote.text }}</span>
    </div>
    <div v-if="content || (role === 'ai' && (pendingQuestion || toolCallLines.length > 0))" class="msg-bubble" :class="role">
      <div v-if="content && role === 'ai'" class="msg-text markdown-body" v-html="renderedContent"></div>
      <div v-else-if="content" class="msg-text">{{ content }}</div>
      <!-- 紧凑的工具调用注释: 紧跟在正文后面, 像对话里的一条小字 -->
      <div v-if="role === 'ai' && toolCallLines.length > 0" class="tool-calls">
        <div v-for="line in toolCallLines" :key="line.key" class="tool-calls__line" :class="`tool-calls__line--${line.status}`">
          <span class="tool-calls__name">{{ line.text }}</span>
          <span v-if="line.args" class="tool-calls__args">{{ line.args }}</span>
          <details v-if="line.result" class="tool-calls__details">
            <summary>查看结果</summary>
            <pre>{{ line.result }}</pre>
          </details>
        </div>
      </div>
      <!-- 未回答的 ask_user 选项: 内联在 AI 气泡底部, 像聊天里点选 -->
      <div v-if="role === 'ai' && pendingQuestion && parsedOptions.length > 0" class="question-options">
        <button
          v-for="(opt, oi) in parsedOptions"
          :key="oi"
          class="question-options__btn"
          @click="emit('question-select', opt)"
        >{{ opt }}</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.msg-row {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  padding: var(--space-sm) var(--space-xl);
  animation: messageIn 0.3s ease-out;
}
.msg-row.user { align-items: flex-end; }
.msg-row.ai   { align-items: flex-start; }

.msg-thinking {
  max-width: 75%;
  margin-bottom: 6px;
}

.msg-bubble {
  max-width: 75%;
  padding: 12px 18px;
  border-radius: 20px;
  font-size: 14px;
  line-height: 1.65;
  word-break: break-word;
}
.msg-bubble.user {
  background: var(--color-primary);
  color: #fff;
  border-bottom-right-radius: 6px;
  white-space: pre-wrap;
}
.msg-bubble.ai {
  background: var(--color-surface);
  color: var(--color-text);
  border: 1px solid var(--color-border);
  border-bottom-left-radius: 6px;
}
.msg-text { }

/* 工具调用注释: 紧跟在 AI 正文下方, 紧凑像对话 */
.tool-calls {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed var(--color-border);
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.tool-calls__line {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--color-text-tertiary);
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
}
.tool-calls__line--running { color: var(--color-primary); }
.tool-calls__line--error { color: var(--color-error); }
.tool-calls__name { font-weight: 600; }
.tool-calls__args {
  opacity: 0.7;
  font-size: 11px;
}
.tool-calls__details {
  width: 100%;
  margin-top: 4px;
}
.tool-calls__details summary {
  cursor: pointer;
  font-size: 11px;
  opacity: 0.6;
  user-select: none;
}
.tool-calls__details summary:hover { opacity: 1; }
.tool-calls__details pre {
  margin: 4px 0 0;
  padding: 8px;
  background: var(--color-bg-soft);
  border-radius: 6px;
  font-size: 11px;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 200px;
  overflow-y: auto;
}

/* 内联选项按钮: 紧跟在 AI 气泡底部, 不再是独立卡片 */
.question-options {
  margin-top: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.question-options__btn {
  padding: 5px 12px;
  background: var(--color-bg-soft);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-full);
  font-size: 13px;
  color: var(--color-text);
  cursor: pointer;
  transition: all var(--transition-fast);
}
.question-options__btn:hover {
  background: var(--color-primary-soft);
  border-color: var(--color-primary);
  color: var(--color-primary);
}

/* 系统消息: 压缩 / 超时总结 / info */
.system-note {
  max-width: 75%;
  margin: 4px 0;
  padding: 6px 12px;
  font-size: 12px;
  display: flex;
  align-items: center;
  gap: 6px;
  border-radius: var(--radius-md);
}
.system-note--compaction {
  background: #f0f9ff;
  color: #0369a1;
  border: 1px solid #bae6fd;
}
.system-note--timeout_summary {
  background: #fffbeb;
  color: #92400e;
  border: 1px solid #fde68a;
}
.system-note--info {
  background: var(--color-bg-soft);
  color: var(--color-text-secondary);
  border: 1px dashed var(--color-border);
}
.system-note__icon { font-size: 13px; }
.system-note__text { line-height: 1.4; }

/* Markdown rendered content */
.markdown-body :deep(h1) { font-size: 1.4em; font-weight: 700; margin: 0.6em 0 0.3em; }
.markdown-body :deep(h2) { font-size: 1.2em; font-weight: 700; margin: 0.5em 0 0.25em; }
.markdown-body :deep(h3) { font-size: 1.05em; font-weight: 600; margin: 0.4em 0 0.2em; }
.markdown-body :deep(p) { margin: 0.3em 0; }
.markdown-body :deep(ul), .markdown-body :deep(ol) { padding-left: 1.5em; margin: 0.3em 0; }
.markdown-body :deep(li) { margin: 0.15em 0; }
.markdown-body :deep(code) {
  background: rgba(0,0,0,0.08);
  padding: 1px 5px;
  border-radius: 4px;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 0.9em;
}
.markdown-body :deep(pre) {
  background: rgba(0,0,0,0.06);
  padding: 10px 14px;
  border-radius: 8px;
  overflow-x: auto;
  margin: 0.4em 0;
}
.markdown-body :deep(pre code) {
  background: none;
  padding: 0;
}
.markdown-body :deep(blockquote) {
  border-left: 3px solid var(--color-primary);
  padding-left: 12px;
  color: var(--color-text-secondary);
  margin: 0.4em 0;
}
.markdown-body :deep(table) { border-collapse: collapse; margin: 0.4em 0; width: 100%; }
.markdown-body :deep(th), .markdown-body :deep(td) {
  border: 1px solid var(--color-border);
  padding: 4px 10px;
  text-align: left;
  font-size: 0.9em;
}
.markdown-body :deep(th) { background: rgba(0,0,0,0.04); font-weight: 600; }
.markdown-body :deep(a) { color: var(--color-primary); }
.markdown-body :deep(hr) { border: none; border-top: 1px solid var(--color-border); margin: 0.6em 0; }
.markdown-body :deep(strong) { font-weight: 700; }
.markdown-body :deep(em) { font-style: italic; }
</style>
