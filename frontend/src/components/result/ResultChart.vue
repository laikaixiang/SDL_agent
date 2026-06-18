<script setup lang="ts">
/**
 * ResultChart.vue — uPlot 包装的图表组件
 *
 * 简单折线 (line) / 柱状 (bar) 图
 * 数据格式: { x: number[], y: number[] }
 *
 * uPlot (~40KB) 优势: 0 依赖, 适合科学数据可视化
 */
import { ref, onMounted, watch, onBeforeUnmount } from 'vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'

interface Props {
  x: number[]
  y: number[]
  chartType?: 'line' | 'bar'
  title?: string
  xLabel?: string
  yLabel?: string
  height?: number
}

const props = withDefaults(defineProps<Props>(), {
  chartType: 'line',
  height: 280,
})

const containerRef = ref<HTMLDivElement | null>(null)
let plotInstance: uPlot | null = null

function buildData(): uPlot.AlignedData {
  return [props.x, props.y] as uPlot.AlignedData
}

function getOpts(): uPlot.Options {
  return {
    width: containerRef.value?.clientWidth || 600,
    height: props.height,
    title: props.title,
    series: [
      {
        label: props.xLabel || 'X',
      },
      {
        label: props.yLabel || 'Y',
        stroke: 'var(--color-primary)',
        width: 2,
        ...(props.chartType === 'bar' ? { type: 'bars' as const } : {}),
        points: { show: false },
      },
    ],
    scales: {
      x: { time: false },
    },
    axes: [
      { stroke: 'var(--color-text-secondary)' },
      { stroke: 'var(--color-text-secondary)' },
    ],
  }
}

onMounted(() => {
  if (containerRef.value && props.x.length && props.y.length) {
    plotInstance = new uPlot(getOpts(), buildData(), containerRef.value)
  }
})

watch(
  () => [props.x, props.y, props.chartType],
  () => {
    if (plotInstance) {
      plotInstance.destroy()
      plotInstance = null
    }
    if (containerRef.value && props.x.length && props.y.length) {
      plotInstance = new uPlot(getOpts(), buildData(), containerRef.value)
    }
  },
  { deep: true }
)

onBeforeUnmount(() => {
  if (plotInstance) {
    plotInstance.destroy()
    plotInstance = null
  }
})
</script>

<template>
  <div class="chart-wrap">
    <div v-if="!x.length || !y.length" class="chart-empty">无数据可绘制</div>
    <div v-else ref="containerRef" class="chart-container"></div>
  </div>
</template>

<style scoped>
.chart-wrap { display: flex; flex-direction: column; }
.chart-container { width: 100%; min-height: 200px; }
.chart-empty {
  padding: var(--space-lg);
  text-align: center;
  color: var(--color-text-tertiary);
  font-size: 13px;
  border: 1px dashed var(--color-border);
  border-radius: var(--radius-sm);
}
</style>

<style>
/* uPlot 全局样式微调 (与本组件主题适配) */
.uplot, .u-wrap {
  font-family: inherit;
  color: var(--color-text-secondary);
}
.u-legend {
  color: var(--color-text-tertiary);
  font-size: 12px;
}
</style>
