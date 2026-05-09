import { ref } from 'vue'
import { defineStore } from 'pinia'

export type TaskPanelType = 'search' | 'extraction' | 'hardware' | 'analysis' | 'experiment' | null
export type TaskStatus = 'idle' | 'running' | 'completed'

export interface TaskEntry {
  type: Exclude<TaskPanelType, null>
  status: TaskStatus
  progress: number  // 0-100
}

export const useLayoutStore = defineStore('layout', () => {
  const sidebarCollapsed = ref(false)
  const rightPanelCollapsed = ref(false)
  const activeTaskPanel = ref<TaskPanelType>(null)
  const taskList = ref<TaskEntry[]>([])

  function toggleSidebar() {
    sidebarCollapsed.value = !sidebarCollapsed.value
  }

  function toggleRightPanel() {
    rightPanelCollapsed.value = !rightPanelCollapsed.value
  }

  function openTaskPanel(type: TaskPanelType) {
    if (!type) return

    if (!taskList.value.find(t => t.type === type)) {
      taskList.value.push({ type, status: 'idle', progress: 0 })
    }

    if (activeTaskPanel.value === type) {
      activeTaskPanel.value = null
    } else {
      activeTaskPanel.value = type
    }
  }

  function closeTask(type: TaskPanelType) {
    if (!type) return
    taskList.value = taskList.value.filter(t => t.type !== type)
    if (activeTaskPanel.value === type) {
      activeTaskPanel.value = null
    }
  }

  function updateTaskStatus(type: Exclude<TaskPanelType, null>, status: TaskStatus, progress?: number) {
    const task = taskList.value.find(t => t.type === type)
    if (task) {
      task.status = status
      if (progress !== undefined) {
        task.progress = Math.max(0, Math.min(100, progress))
      }
      if (status === 'completed') {
        task.progress = 100
      }
    }
  }

  function acknowledgeTask(type: Exclude<TaskPanelType, null>) {
    const task = taskList.value.find(t => t.type === type)
    if (task && task.status === 'completed') {
      task.status = 'idle'
    }
  }

  function closeTaskPanel() {
    activeTaskPanel.value = null
  }

  return {
    sidebarCollapsed, rightPanelCollapsed, activeTaskPanel, taskList,
    toggleSidebar, toggleRightPanel, openTaskPanel, closeTask, updateTaskStatus,
    acknowledgeTask, closeTaskPanel,
  }
})
