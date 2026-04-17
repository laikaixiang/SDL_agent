/**
 * input_state.js — 输入区域 UI 状态控制
 *
 * 根据当前操作类型（普通加载 / 流式聊天 / 硬件执行 / 任务提取）
 * 切换发送按钮和输入框的禁用状态与显示文字。
 *
 * 依赖：state.js（chatStreaming, hardwareRunning, userInput, sendBtn, inputContainer）
 */

/** 设置普通 API 请求的加载状态（禁用输入框和发送按钮，显示转圈）。不覆盖流式/硬件状态。 */
function setNormalLoadingState(isLoading) {
    if (chatStreaming || hardwareRunning) return;
    userInput.disabled = isLoading;
    sendBtn.disabled = isLoading;
    inputContainer.classList.toggle('disabled', isLoading);
    sendBtn.innerHTML = isLoading ? '<div class="btn-spinner"></div>' : '发送';
    if (!isLoading) userInput.focus();
}

/** 设置流式聊天状态：isStreaming=true 时发送按钮变为"停止生成"并可点击中断，false 时恢复正常。 */
function setChatStreamingState(isStreaming) {
    chatStreaming = isStreaming;
    if (isStreaming) {
        userInput.disabled = true;
        inputContainer.classList.add('disabled');
        sendBtn.innerHTML = '停止生成';
        sendBtn.className = 'send-btn cancel-mode';
        sendBtn.disabled = false;
    } else {
        userInput.disabled = false;
        inputContainer.classList.remove('disabled');
        sendBtn.innerHTML = '发送';
        sendBtn.className = 'send-btn';
        userInput.focus();
    }
}

/** 设置硬件执行状态：isRunning=true 时完全禁用输入和按钮，防止用户误操作。 */
function setHardwareRunningState(isRunning) {
    hardwareRunning = isRunning;
    if (isRunning) {
        userInput.disabled = true;
        inputContainer.classList.add('disabled');
        sendBtn.innerHTML = '<div class="btn-spinner"></div> 硬件执行中';
        sendBtn.className = 'send-btn';
        sendBtn.disabled = true;
    } else {
        userInput.disabled = false;
        inputContainer.classList.remove('disabled');
        sendBtn.innerHTML = '发送';
        sendBtn.className = 'send-btn';
        sendBtn.disabled = false;
        userInput.focus();
    }
}

/** 设置文献提取任务状态：isRunning=true 时发送按钮变为"暂停提取"并绑定取消回调，false 时解绑并恢复。 */
function setTaskRunningState(isRunning) {
    if (isRunning) {
        userInput.disabled = true;
        inputContainer.classList.add('disabled');
        sendBtn.innerHTML = '<div class="btn-spinner"></div> 暂停提取';
        sendBtn.className = 'send-btn cancel-mode';
        sendBtn.onclick = null;
        sendBtn.addEventListener('click', requestCancelTask);
    } else {
        userInput.disabled = false;
        inputContainer.classList.remove('disabled');
        sendBtn.innerHTML = '发送';
        sendBtn.className = 'send-btn';
        sendBtn.removeEventListener('click', requestCancelTask);
        sendBtn.onclick = null;
        userInput.focus();
    }
}

/** 显示提取完成的摘要弹窗，展示提取条数和 CSV 保存路径。 */
function showSummaryModal(csvPath, count) {
    document.getElementById('summary-text').innerHTML =
        `共提取了 <b>${count}</b> 条高价值数据。<br><br>完整的数据表已保存在你的项目目录下：<br>` +
        `<code style="background:#f3f4f6; padding:4px; border-radius:4px; font-size:0.9em; word-break:break-all;">${csvPath}</code>`;
    document.getElementById('summary-modal').style.display = 'flex';
}

/** 创建一个消息气泡行（message-row + message div），追加到聊天框并返回内层 message div。 */
function createMessageDiv(sender) {
    const row = document.createElement('div');
    row.className = `message-row ${sender}`;
    const msg = document.createElement('div');
    msg.className = `message ${sender}`;
    row.appendChild(msg);
    chatBox.appendChild(row);
    return msg;
}

/** 将 HTML 字符串作为一条消息追加到聊天框（sender: 'user' | 'ai'）。 */
function appendMessageHtml(html, sender) {
    const msg = createMessageDiv(sender);
    msg.innerHTML = html;
    scrollToBottom();
}

/** 将纯文本作为一条消息追加到聊天框（sender: 'user' | 'ai'）。 */
function appendMessage(text, sender) {
    const msg = createMessageDiv(sender);
    msg.textContent = text;
    scrollToBottom();
}

/** 将聊天框滚动到最底部。 */
function scrollToBottom() {
    chatBox.scrollTop = chatBox.scrollHeight;
}
