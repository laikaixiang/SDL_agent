import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory('/v2'),
  routes: [
    {
      path: '/',
      name: 'chat',
      component: () => import('@/pages/ChatPage.vue'),
    },
  ],
})

export default router
