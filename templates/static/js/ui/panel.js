/**
 * panel.js — 面板 z-index 管理
 *
 * 多个右侧面板（PDF / 单步控制 / 实验设计）可同时存在，
 * 通过递增 z-index 确保后打开的面板始终显示在最上层。
 *
 * 依赖：state.js（panelZIndexCounter, activePanels）
 */

/** 递增全局 z-index 计数器，将指定面板的 zIndex 设为最大值，并加入活跃面板集合。每次打开面板时调用。 */
function bringPanelToFront(panelId) {
    const panel = document.getElementById(panelId);
    if (!panel) return;
    panelZIndexCounter++;
    panel.style.zIndex = panelZIndexCounter;
    activePanels.add(panelId);
}

/** 从活跃面板集合中删除指定面板 ID，关闭面板时调用以避免集合无限增长。 */
function removePanelFromTracking(panelId) {
    activePanels.delete(panelId);
}
