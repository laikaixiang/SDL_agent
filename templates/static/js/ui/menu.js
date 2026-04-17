/**
 * menu.js — 模式菜单与硬件子菜单
 *
 * 管理左下角工具按钮弹出的模式菜单，以及硬件子菜单的显示/隐藏。
 * setMode() 是核心函数，被 chat、extraction、experiment 等模块调用。
 *
 * 依赖：state.js（currentMode, modeMenu, modeBadge, userInput, toolBtn）
 *       ui/panel.js（hideHardwareSubmenu 内部使用 timer）
 */

let hardwareSubmenuTimer = null; // 延迟隐藏子菜单的定时器

/* 工具按钮点击：切换模式菜单显示 */
toolBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    modeMenu.style.display = modeMenu.style.display === 'flex' ? 'none' : 'flex';
});

/* 点击页面其他区域时关闭所有菜单 */
document.addEventListener('click', () => {
    modeMenu.style.display = 'none';
    hideHardwareSubmenu();
});

/** 显示硬件子菜单，清除延迟隐藏定时器防止菜单闪烁。 */
function showHardwareSubmenu(event) {
    event.stopPropagation();
    clearTimeout(hardwareSubmenuTimer);
    document.getElementById('hardware-submenu').style.display = 'flex';
}

/** 延迟 200ms 隐藏硬件子菜单，给用户留出移入子菜单的时间窗口。 */
function hideHardwareSubmenu() {
    hardwareSubmenuTimer = setTimeout(() => {
        document.getElementById('hardware-submenu').style.display = 'none';
    }, 200);
}

/** 鼠标移入子菜单时取消隐藏定时器，保持子菜单可见。 */
function keepHardwareSubmenu() {
    clearTimeout(hardwareSubmenuTimer);
}

/**
 * 切换当前输入模式，更新 currentMode 状态、隐藏菜单、刷新模式徽章和输入框占位符。
 * @param {string} id     - 模式 ID，'normal' 表示普通模式
 * @param {string} prefix - 发送时自动拼接到消息前的前缀字符串
 * @param {string} label  - 显示在输入框左侧徽章中的文字
 */
function setMode(id, prefix, label) {
    currentMode = { id, prefix, label };
    modeMenu.style.display = 'none';
    hideHardwareSubmenu();

    if (id === 'normal') {
        modeBadge.style.display = 'none';
        userInput.placeholder = '输入问题或指令...';
    } else {
        modeBadge.innerHTML = `${label} <span style="cursor:pointer; margin-left:4px" onclick="event.stopPropagation(); setMode('normal','','')">×</span>`;
        modeBadge.style.display = 'flex';
        userInput.placeholder = '输入要求(可留空默认)...';
    }
    userInput.focus();
}
