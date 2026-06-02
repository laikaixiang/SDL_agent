# 平台整体布局数字孪生 — 实现计划

## Context

在 `robot_arm_model/templates/index.html` 现有 Three.js 机械臂模型基础上，
扩展平台布局数字孪生功能。平台由 9×9 格点组成（每格 50mm），托盘尺寸 6×9，
集成 4 类模块（匀胶机、载玻片托盘、滴管头盒、瓶子托盘），支持网格吸附拖拽。

## 用户确认

- 在现有 index.html 基础上扩展 ✓
- 网格吸附拖拽（B）✓
- 模块位置先估算，可动态调整 ✓

---

## 实现方案

### 修改文件

**核心修改：** `robot_arm_model/templates/index.html`

- 新增 9×9 网格底板（GRID_SIZE=50mm，总 450×450mm）
- 新增 4 类模块：SubstrateTray / SpinCoater / DropperBox / BottleTray
- 实现 DragControls + 网格吸附（吸附到 50mm 间隔）
- 机械臂保持左下角固定
- 导出 API 供后续集成调用

### 模块建模参数

| 模块 | 估算尺寸 | 初始位置（格） | 说明 |
|------|---------|--------------|------|
| 载玻片托盘 | 300×450mm (6×9格) | (5, 3) | 平台中央偏右 |
| 匀胶机 | 150×150mm (3×3格) | (7, 6) | 平台右上区域 |
| 滴管头盒 | 100×150mm (2×3格) | (4, 6) | 平台右侧中部 |
| 瓶子托盘 | 100×100mm (2×2格) | (2, 6) | 平台右下区域 |

### 关键代码设计

**网格吸附：**
```javascript
const GRID_SIZE = 50;
const PLATFORM_SIZE = GRID_SIZE * 9; // 450mm × 450mm

function snapToGrid(val) {
  return Math.round(val / GRID_SIZE) * GRID_SIZE;
}
```

**拖拽结束吸附：**
```javascript
controls.addEventListener('dragend', e => {
  e.object.position.x = snapToGrid(e.object.position.x);
  e.object.position.z = snapToGrid(e.object.position.z);
  // 边界约束
  const half = PLATFORM_SIZE / 2;
  e.object.position.x = Math.max(-half, Math.min(half, e.object.position.x));
  e.object.position.z = Math.max(-half, Math.min(half, e.object.position.z));
});
```

**模块几何工厂函数：**
```javascript
function makeBox(w, h, d, color, roughness = 0.5, metalness = 0.3) {
  return new THREE.Mesh(
    new THREE.BoxGeometry(w, h, d),
    new THREE.MeshStandardMaterial({color, roughness, metalness})
  );
}
```

---

## 验证方式

1. `python robot_arm_model/digital_twin.py` → http://127.0.0.1:5001
2. 确认 9×9 网格底板可见（450×450mm）
3. 确认机械臂在左下角
4. 确认 4 类模块可见
5. 拖拽模块 → 验证吸附到网格（50mm 间隔）
6. 检查 console 无错误

---

## 融入 frontend 计划（Phase B）

分两阶段：

1. **Phase A（已完成）：** 独立数字孪生页面
   - 修改 `robot_arm_model/templates/index.html`
   - 运行于 http://127.0.0.1:5001
   - 模块化设计，API 接口清晰

2. **Phase B（后续）：** 集成到实验设计界面
   - 将 Three.js 场景封装为独立组件
   - 创建 `frontend/src/components/PlatformTwin.vue`
   - 在实验设计页面嵌入为面板/标签

---

## 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `robot_arm_model/templates/index.html` | 修改 | 扩展 Three.js 场景，添加平台模块 |
| `robot_arm_model/digital_twin.py` | 不变 | Flask 服务端 |
| `robot_arm_model/kinematics.py` | 不变 | 运动学引擎 |