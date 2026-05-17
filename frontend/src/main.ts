import { createApp } from 'vue'
import { createPinia } from 'pinia'
import i18n from '@/i18n'
import App from './App.vue'
import router from './router'
import 'katex/dist/katex.min.css'
import './styles/main.css'

const app = createApp(App)

app.use(i18n)
app.use(createPinia())
app.use(router)

app.mount('#app')
