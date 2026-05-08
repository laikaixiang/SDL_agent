<script setup lang="ts">
import ChatContainer from '@/components/chat/ChatContainer.vue'
</script>

<template>
  <div class="chat-page">
    <ChatContainer />
  </div>
</template>

<style scoped>
.chat-page { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
</style>
