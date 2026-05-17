import { createI18n } from 'vue-i18n'
import en from '@/locales/en.json'
import zh from '@/locales/zh.json'

export type Language = 'en' | 'zh'

export function getLanguage(): Language {
  const stored = localStorage.getItem('language')
  if (stored === 'en' || stored === 'zh') return stored
  if (navigator.language?.toLowerCase().startsWith('en')) return 'en'
  return 'zh'
}

export function setLanguage(lang: Language): void {
  localStorage.setItem('language', lang)
  document.cookie = `lang=${lang}; path=/; max-age=${365 * 86400}; SameSite=Lax`
  if (i18n.global.locale.value !== lang) {
    i18n.global.locale.value = lang
  }
}

const i18n = createI18n({
  legacy: false,
  locale: getLanguage(),
  fallbackLocale: 'zh',
  messages: { en, zh }
})

export default i18n
