import { defineStore } from 'pinia'
import { ref } from 'vue'
import { getLanguage, setLanguage, type Language } from '@/i18n'

export const useSettingsStore = defineStore('settings', () => {
  const language = ref<Language>(getLanguage())

  function switchLanguage(lang: Language) {
    language.value = lang
    setLanguage(lang)
  }

  return { language, switchLanguage }
})
