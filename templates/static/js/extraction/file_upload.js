/**
 * file_upload.js — PDF 文件上传与拖拽处理
 *
 * 监听回形针按钮和拖拽事件，将 PDF 文件上传到 /api/upload。
 * 依赖：state.js, ui/input_state.js（appendMessage）
 */

const fileUpload     = document.getElementById('file-upload');
const chatContainer  = document.getElementById('chat-container');

/* 回形针按钮选择文件 */
fileUpload.addEventListener('change', (e) => handleFiles(e.target.files));

/* 拖拽上传 */
chatContainer.addEventListener('dragover',  (e) => { e.preventDefault(); chatContainer.classList.add('drag-over'); });
chatContainer.addEventListener('dragleave', (e) => { e.preventDefault(); chatContainer.classList.remove('drag-over'); });
chatContainer.addEventListener('drop', (e) => {
    e.preventDefault();
    chatContainer.classList.remove('drag-over');
    handleFiles(e.dataTransfer.files);
});

/* 关闭 PDF 面板按钮 */
document.getElementById('close-pdf-btn').addEventListener('click', () => {
    removePanelFromTracking('pdf-panel');
    document.getElementById('app-wrapper').classList.remove('pdf-mode');
});

/**
 * 过滤出 PDF 文件并上传到服务器。
 * @param {FileList} files
 */
async function handleFiles(files) {
    if (!files || files.length === 0) return;

    const formData = new FormData();
    let pdfCount = 0;
    for (const file of files) {
        if (file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')) {
            formData.append('files', file);
            pdfCount++;
        }
    }

    if (pdfCount === 0) {
        appendMessage('仅支持上传 PDF 格式的文献哦！', 'ai');
        return;
    }

    appendMessage(`正在将 ${pdfCount} 份文献归档至本地知识库...`, 'user');
    try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.status === 'success') {
            appendMessage(`✅ 成功归档！可随时对它们下达提取指令。\n\n已保存：\n${data.saved.join('\n')}`, 'ai');
        } else {
            appendMessage(`❌ 上传失败：${data.error}`, 'ai');
        }
    } catch (e) {
        appendMessage('❌ 网络异常，未能连接到本地服务。', 'ai');
    }

    fileUpload.value = ''; // 清空选择器，允许重复上传同一文件
}
