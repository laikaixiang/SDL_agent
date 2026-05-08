import { ref, watchEffect } from 'vue'
import { defineStore } from 'pinia'

export const useThemeStore = defineStore('theme', () => {
  const theme = ref<'light' | 'dark'>(
    (localStorage.getItem('theme') as 'light' | 'dark') || 'light'
  )

  function toggle() {
    theme.value = theme.value === 'light' ? 'dark' : 'light'
  }

  watchEffect(() => {
    document.documentElement.setAttribute('data-theme', theme.value)
    localStorage.setItem('theme', theme.value)
  })

  return { theme, toggle }
})
