<script setup lang="ts">
import { computed } from 'vue'
import { useExperimentStore } from '@/stores/experiment'
import StepCard from './StepCard.vue'
import { AlertTriangle } from 'lucide-vue-next'

const INDENT = 20

const store = useExperimentStore()

const draggedIndex = computed({
  get: () => store.draggedStepIndex,
  set: (v) => { store.draggedStepIndex = v },
})

function onDragStart(index: number, event: DragEvent) {
  draggedIndex.value = index
  if (event.dataTransfer) {
    event.dataTransfer.effectAllowed = 'move'
  }
}

function onDragOver(index: number, event: DragEvent) {
  event.preventDefault()
  if (event.dataTransfer) {
    event.dataTransfer.dropEffect = 'move'
  }
}

function onDrop(index: number) {
  if (draggedIndex.value !== null && draggedIndex.value !== index) {
    store.moveStep(draggedIndex.value, index)
  }
  draggedIndex.value = null
}

function onDragEnd() {
  draggedIndex.value = null
}

function getHelperLabel(name: string): string {
  const labels: Record<string, string> = {
    LOOP: '循环', GROUP: '分组', CONDITION: '条件',
    WAIT: '等待', END: '结束', USER_INPUT: '输入',
  }
  return labels[name] || name
}

function isCollapsed(index: number): boolean {
  return store.collapsedBlocks.has(index)
}
</script>

<template>
  <div class="step-canvas">
    <!-- Empty state -->
    <div v-if="!store.steps.length" class="canvas-empty">
      <div class="empty-icon">🧪</div>
      <p>从左侧双击工具/算法添加步骤</p>
      <p class="empty-hint">或点击工具栏 [AI 生成] 自动设计</p>
    </div>

    <!-- Step rows with gutter -->
    <div v-else class="canvas-body">
      <template v-for="(step, i) in store.steps" :key="i">
        <div
          v-if="!store.hiddenStepIndices.has(i)"
          class="step-row"
          :class="{
            dragging: store.draggedStepIndex === i,
            'has-error': store.blockErrors.lastIndex === i,
            collapsed: isCollapsed(i),
          }"
          draggable="true"
          @dragstart="onDragStart(i, $event)"
          @dragover="onDragOver(i, $event)"
          @drop="onDrop(i)"
          @dragend="onDragEnd"
        >
          <!-- Gutter -->
          <div
            class="step-gutter"
            :class="{
              'gutter-error': store.blockErrors.lastIndex === i,
              'gutter-orphan': store.blockErrors.orphanedEnd === i,
            }"
          >
            <!-- Block marker button (clickable for block-openers) -->
            <button
              v-if="store.nestingInfo[i].isBlockStart"
              class="block-marker-btn"
              :title="isCollapsed(i) ? '展开' + getHelperLabel(step.name) : '收起' + getHelperLabel(step.name)"
              @click="store.toggleCollapse(i)"
            >
              <span v-if="isCollapsed(i)">▶</span>
              <span v-else>▼</span>
            </button>
            <span
              v-else-if="store.nestingInfo[i].isBlockEnd"
              class="block-marker block-end"
              title="结束"
            >▲</span>

            <!-- Step number -->
            <span class="step-num">{{ i + 1 }}</span>
          </div>

          <!-- Step content with indentation -->
          <div
            class="step-content"
            :style="{ paddingLeft: store.nestingInfo[i].level * INDENT + 'px' }"
          >
            <!-- Indentation guide lines -->
            <span
              v-for="lv in store.nestingInfo[i].guideLines"
              :key="'gl-' + lv"
              class="indent-guide"
              :style="{ left: lv * INDENT + INDENT / 2 + 'px' }"
            />

            <StepCard
              :step="step"
              :index="i"
              :total="store.steps.length"
              :level="store.nestingInfo[i].level"
              :is-block-start="store.nestingInfo[i].isBlockStart"
              :is-block-end="store.nestingInfo[i].isBlockEnd"
              :editing="store.editingStepIndex === i"
              @edit="store.toggleEdit(i)"
              @remove="store.removeStep(i)"
              @move-up="store.moveStepUp(i)"
              @move-down="store.moveStepDown(i)"
            />
          </div>
        </div>
      </template>

      <!-- Warning for unclosed blocks -->
      <div v-if="store.blockErrors.unclosed" class="block-warning">
        <AlertTriangle :size="14" />
        <span>{{ getHelperLabel(store.blockErrors.unclosed) }} 缺少对应的 END</span>
      </div>

      <!-- Warning for orphaned END -->
      <div v-if="store.blockErrors.orphanedEnd !== null && !store.blockErrors.unclosed" class="block-warning">
        <AlertTriangle :size="14" />
        <span>多余的 END，前面没有对应的 循环/分组/条件</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.step-canvas {
  flex: 1;
  overflow-y: auto;
  min-width: 0;
  background: var(--color-bg-soft);
}

/* Empty */
.canvas-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--color-text-tertiary);
  font-size: 14px;
  text-align: center;
  gap: var(--space-sm);
}
.empty-icon { font-size: 36px; opacity: 0.5; }
.empty-hint { font-size: 12px; }

/* Body */
.canvas-body {
  display: flex;
  flex-direction: column;
}

/* Row */
.step-row {
  display: flex;
  min-height: 48px;
  border-bottom: 1px solid var(--color-border);
  transition: background var(--transition-fast);
  position: relative;
}

.step-row:hover {
  background: var(--color-bg-mute);
}

.step-row.dragging {
  opacity: 0.4;
  background: var(--color-primary-mute);
}

.step-row.has-error {
  background: #fef2f2;
}

.step-row.collapsed {
  background: #fafafa;
}

/* =================== GUTTER =================== */

.step-gutter {
  width: 42px;
  min-width: 42px;
  background: var(--color-bg-soft);
  border-right: 1px solid var(--color-border);
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: 8px;
  position: relative;
  user-select: none;
  gap: 0;
}

.step-gutter.gutter-error {
  background: #fef2f2;
  border-right-color: #fecaca;
}

.step-gutter.gutter-error .step-num {
  color: #ef4444;
  font-weight: 700;
}

.step-num {
  font-size: 11px;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  color: var(--color-text-tertiary);
  line-height: 1;
}

/* Block marker button (clickable expand/collapse) */
.block-marker-btn {
  position: absolute;
  left: 5px;
  top: 50%;
  transform: translateY(-50%);
  border: none;
  background: transparent;
  padding: 3px;
  cursor: pointer;
  color: #f59e0b;
  font-size: 10px;
  line-height: 1;
  border-radius: 3px;
  transition: background var(--transition-fast);
}

.block-marker-btn:hover {
  background: #fef3c7;
}

/* Non-clickable block-end marker */
.block-marker {
  position: absolute;
  left: 6px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 10px;
  line-height: 1;
}

.block-end {
  color: var(--color-text-tertiary);
  opacity: 0.6;
}

/* =================== CONTENT =================== */

.step-content {
  flex: 1;
  min-width: 0;
  position: relative;
  padding: var(--space-sm) 0;
}

/* Indentation guide lines */
.indent-guide {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 0;
  border-left: 1px solid var(--color-border);
  opacity: 0.4;
  pointer-events: none;
}

/* =================== WARNING =================== */

.block-warning {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px 8px 50px;
  background: #fffbeb;
  border-bottom: 1px solid #fde68a;
  color: #92400e;
  font-size: 12px;
}
</style>
