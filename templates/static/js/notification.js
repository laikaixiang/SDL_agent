/**
 * notification.js — 右上角浮动通知
 *
 * 提供 showNotification() 全局函数，3 秒后自动消失。
 * 依赖：无（纯 DOM 操作）
 */

/**
 * 显示右上角浮动通知。
 * @param {string} message - 通知文字
 * @param {'info'|'success'|'error'} type - 通知类型，决定背景色
 */
function showNotification(message, type = 'info') {
    const colors = { success: '#10b981', error: '#ef4444', info: '#3b82f6' };
    const el = document.createElement('div');
    el.className = `notification notification-${type}`;
    el.textContent = message;
    el.style.cssText = `
        position:fixed; top:20px; right:20px; padding:15px 20px;
        background:${colors[type] || colors.info}; color:white;
        border-radius:8px; box-shadow:0 4px 6px rgba(0,0,0,0.1);
        z-index:10000; animation:slideIn 0.3s ease-out; max-width:400px;
    `;
    document.body.appendChild(el);

    setTimeout(() => {
        el.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => el.remove(), 300);
    }, 3000);
}
