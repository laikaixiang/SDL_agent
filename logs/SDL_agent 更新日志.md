# SDL_agent 更新日志 — 2026/04/17

**修改人员**: lkx

---

## 1. Bug修复

### 单步控制误开PDF面板
- **文件**: `templates/index.html` (`startTaskStream` 函数)
- **问题**: 切换到硬件操控模式 → 单步控制时，`page_reading` 事件会无条件打开PDF预览面板
- **修复**: 添加 `currentMode.id === 'extract'` 守卫，仅在文献提取模式下打开PDF面板；`complete` 事件中仅在PDF面板实际打开时关闭扫描动画

---

## 2. 算法选择UI优化

- **文件**: `templates/index.html` (`displayAlgorithmList` 函数 + CSS)
- **改动**: 算法列表由高卡片改为可折叠行
  - 折叠态：图标 + 名称 + 标签 + 展开箭头（单行）
  - 展开态：算法ID、完整描述、全部标签、"选择并运行"按钮
- **交互**: 复用 `.tool-row` 风格，点击展开/收起，同时只展开一个

---

## 3. 实验设计三栏布局

- **文件**: `templates/index.html` (CSS + HTML + JS)
- **改动**: 选择"实验设计模式"后，页面切换为三栏布局：
  - **左侧栏** (`#experiment-canvas-panel`, 340px): 可视化画布 + JSON代码视图
  - **中间栏** (flex:1): 原有聊天区，显示AI设计结果与审批按钮
  - **右侧栏** (`#experiment-blocks-panel`, 260px): 可拖拽实验步骤积木 + 辅助函数积木
- 进入实验设计模式时自动关闭PDF面板和单步控制面板

---

## 4. 拖拽引擎

- **文件**: `templates/index.html` (JS)
- **实现**: 基于原生HTML5 Drag and Drop API，无外部依赖
  - 右侧积木块拖入左侧画布 → 创建步骤卡片
  - 画布内步骤可拖拽重排序
  - 点击步骤卡片展开内联参数编辑
  - JSON视图实时同步
- **积木类型**:
  - 实验操作: 旋涂(spin_coating)、设温(set_temperature)、移臂(move_robot_arm)、采谱(collect_spectrum)
  - 辅助函数: 循环(LOOP)、组(GROUP)
- **画布工具栏**: 清空、导入JSON、导出JSON、保存到服务器、参数导入(CSV/Excel预留)

---

## 5. AI设计结果同步到画布

- **文件**: `templates/index.html` (`handleDesignResult` 函数)
- **改动**: AI返回实验方案后：
  1. 自动加载到左侧画布（`loadPlanIntoCanvas`）
  2. 自动保存到服务器（`/api/experiment_designs/save`）
  3. 中间栏显示审批按钮组："同意并执行" / "在画布中修改" / "重新设计" / "保存方案"
  4. 执行时优先使用画布最新状态

---

## 6. 后端 — 实验设计JSON存储API

- **文件**: `app.py`
- **新增路由**:
  - `POST /api/experiment_designs/save` — 保存设计到 `experiment_designs/{batch}/{timestamp}_{name}.json`
  - `GET /api/experiment_designs/list` — 列出所有已保存设计
  - `GET /api/experiment_designs/<filename>` — 加载特定设计
  - `DELETE /api/experiment_designs/<filename>` — 删除设计
  - `POST /api/experiment_designs/import_params` — CSV/Excel参数导入（stub，返回501）

---

## 7. 闭环优化预留接口

- **文件**: `templates/index.html` + `app.py`
- **前端**: 实验执行完成后显示"继续优化（闭环）"按钮，点击后将上一轮执行结果发送给AI推荐下一轮实验
- **后端**: `/api/experiment_chat` 新增 `previous_results` 字段支持，自动拼接到用户消息中供AI参考
- **预期闭环流程**: 算法推荐 → 实验执行 → 光谱测试 → 读取数据 → AI推荐下一轮 → 循环

---

## 8. UI视觉优化

- **文件**: `templates/index.html` (CSS)
- 新增CSS变量：`--shadow-sm/md/lg`、`--radius-sm/md/lg`、`--accent-*` 颜色
- Header底部渐变边框（蓝→绿→紫）
- 输入框聚焦时添加蓝色外发光
- Mode badge弹出动画
- 画布步骤卡片之间的连接线
- 积木块左侧颜色编码（蓝=旋涂、橙=温度、紫=机械臂、绿=光谱）
