import { ref } from 'vue'
import { defineStore } from 'pinia'

export type TaskPanelType = 'search' | 'extraction' | 'hardware' | 'analysis' | 'experiment' | null

export const useLayoutStore = defineStore('layout', () => {
  const sidebarCollapsed = ref(false)
  const rightPanelCollapsed = ref(false)
  const activeTaskPanel = ref<TaskPanelType>(null)

  function toggleSidebar() {
    sidebarCollapsed.value = !sidebarCollapsed.value
  }

  function toggleRightPanel() {
    rightPanelCollapsed.value = !rightPanelCollapsed.value
  }

  function openTaskPanel(type: TaskPanelType) {
    if (activeTaskPanel.value === type) {
      activeTaskPanel.value = null
    } else {
      activeTaskPanel.value = type
    }
  }

  function closeTaskPanel() {
    activeTaskPanel.value = null
  }

  return { sidebarCollapsed, rightPanelCollapsed, activeTaskPanel, toggleSidebar, toggleRightPanel, openTaskPanel, closeTaskPanel }
})
