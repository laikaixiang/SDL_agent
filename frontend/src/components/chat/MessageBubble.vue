<script setup lang="ts">
import { computed } from 'vue'
import { markdownToHtml } from '@/utils/markdown'
import ThinkingBlock from './ThinkingBlock.vue'
import AgentToolCard from './AgentToolCard.vue'
import AgentTeamCard from './AgentTeamCard.vue'
import AgentQuestionCard from './AgentQuestionCard.vue'
import type { ToolCallInfo, TeamAgentInfo, AgentQuestionWithAnswer } from '@/types/chat'

const props = defineProps<{
  role: 'user' | 'ai'
  content: string
  thinking?: string
  thinkingDuration?: number
  timestamp?: string
  // Agent attachments (kept visible after the AI's final reply)
  toolCalls?: ToolCallInfo[]
  questions?: AgentQuestionWithAnswer[]
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

// Filter out tool calls that have no result and aren't running —
// they're transient noise after the agent finishes.
const visibleToolCalls = computed(() => {
  if (!props.toolCalls) return []
  return props.toolCalls.filter(
    tc => tc.status === 'running' || tc.result !== undefined
  )
})

const hasAgentData = computed(() =>
  props.role === 'ai' && (
    visibleToolCalls.value.length > 0 ||
    (props.questions && props.questions.length > 0) ||
    (props.teamAgents && props.teamAgents.length > 0)
  )
)
</script>

<template>
  <div class="msg-row" :class="role">
    <ThinkingBlock
      v-if="role === 'ai' && thinking"
      :content="thinking"
      :duration="thinkingDuration"
      class="msg-thinking"
    />
    <div class="msg-bubble" :class="role">
      <div v-if="role === 'ai'" class="msg-text markdown-body" v-html="renderedContent"></div>
      <div v-else class="msg-text">{{ content }}</div>
    </div>
    <!-- Agent attachments stay attached to this message even after the agent
         finishes its turn, so the chat log looks like a normal conversation. -->
    <template v-if="hasAgentData">
      <div v-if="visibleToolCalls.length > 0" class="msg-attachments">
        <AgentToolCard
          v-for="tc in visibleToolCalls"
          :key="`tc-${tc.index}-${tc.name}`"
          :index="tc.index"
          :name="tc.name"
          :args="tc.arguments"
          :result="tc.result"
          :status="tc.status"
        />
      </div>
      <div v-if="teamAgents && teamAgents.length > 0" class="msg-attachments">
        <AgentTeamCard
          mode="parallel"
          :agents="teamAgents"
          :collapsed="true"
        />
      </div>
      <div v-if="questions && questions.length > 0" class="msg-attachments">
        <AgentQuestionCard
          v-for="(q, qi) in questions"
          :key="`q-${qi}-${q.question.slice(0, 20)}`"
          :question="q.question"
          :options="q.options"
          :answer="q.answer"
          @select="emit('question-select', $event)"
        />
      </div>
    </template>
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

/* Agent attachments (tool calls, team card, questions) sit below the
   message bubble but above the next message — they belong to THIS turn. */
.msg-attachments {
  max-width: 75%;
  margin-top: 4px;
  display: flex;
  flex-direction: column;
  align-items: stretch;
}

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
