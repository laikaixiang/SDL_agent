# Dobot M1Pro SCARA 机械臂 — 数字孪生

基于越疆 M1Pro 四轴 SCARA 协作机械臂的物理仿真数字孪生，支持关节空间/笛卡尔空间控制、TCP 协议模拟和实时 3D 可视化。

## 快速启动

```bash
cd digital_twin/robot_arm_model
python digital_twin.py
# 浏览器打开 http://127.0.0.1:5001
```

**零额外依赖** — Flask（项目已有）+ Three.js（CDN）+ numpy（项目已有）。

---

## 目录结构

```
test_digital_twin/
├── PLAN.md                  # 平台布局数字孪生实现计划
├── WORK_SUMMARY.md          # 工作记录与变更总览
└── robot_arm_model/
    ├── README.md            # 本文件
    ├── PLAN.md              # 技术方案文档 (机械臂本体)
    ├── kinematics.py        # Dobot M1Pro 运动学引擎 (FK/IK/Jacobian)
    ├── kinematics_pipette.py # 移液臂运动学引擎 (X/Y/Z1/Z2 棱柱)
    ├── pipette_kinematic_params.json # 移液臂运动学参数 (STL分析)
    ├── extract_pipette_params.py    # 移液臂 STL 参数提取脚本
    ├── digital_twin.py      # Flask 服务端 + REST API + 配置持久化
    ├── platform_config.json # 平台模块配置（可手动编辑或通过UI保存）
    ├── LOG_20260531_lkx.md  # 详细变更日志
    ├── templates/
    │   ├── index.html       # 3D 交互界面 (Three.js, 自包含)
    │   │                    #   - Dobot M1Pro SCARA
    │   │                    #   - 移液臂 XYZZ+双ADP
    │   │                    #   - 800×800 桌面模块 + 托盘
    │   │                    #   - 手指 (Dobot 末端)
    │   │                    #   - 6模块平台布局 + 编辑模式 + 多实例 + 持久化
    │   ├── models/
    │   │   ├── dobot_m1pro/      # Dobot STL 零件
    │   │   ├── pipette_arm/      # 移液臂 STL 零件 + 减面版本
    │   │   ├── finger/           # 末端手指模型
    │   │   └── plates/           # 样品托盘模型
    │   └── viewers/          # 各零件的独立 HTML 预览页
    │       ├── M1_*_viewer.html
    │       ├── pipette_group*_viewer.html
    │       ├── pipette_tip_viewer.html
    │       ├── pipette_assembly_viewer.html
    │       ├── assembly_viewer.html
    │       ├── finger_viewer.html  # 手指
    │       └── plate_viewer.html   # 托盘
    └── dobot_protocol.txt   # 越疆 TCP/IP 协议文档 (参考)
```

---

## 用户编辑指引 (2026-06-02)

### 修改机械臂关节限位

**Dobot M1Pro**：编辑 `data/offsets/dobot_joint_offsets.json`  
**移液臂**：编辑 `data/offsets/pipette_kinematic_params.json`

编辑后刷新页面，前端 slider 的 min/max/value 自动同步。

> `kinematics_M1Pro.py` 和 `kinematics_pipette.py` 在模块加载时读取 JSON，修改 JSON 等效于修改硬编码常量。

### 关节限位配置字段说明

**pipette_kinematic_params.json** 中的 `joints` 节点：

```json
"X": { "mechanical_range": [44.1, 384.9], "reference_mm": 300.3 }
```
- `mechanical_range[0]` → slider **min**
- `mechanical_range[1]` → slider **max**
- `reference_mm` → slider 初始 **value**

同理 `Z` → Y轴滑块，`Y1`/`Y2` → Z1/Z2 滑块。

**dobot_joint_offsets.json** 目前主要用于零件偏移数据，Dobot 关节限位由 `kinematics_M1Pro.py` 中的常量定义（J1/J2/J4 ±85°/±130°/±360°，J3 由 Z_MAX/Z_MIN 决定）。

### 运行时状态持久化

页面加载时自动从 `data/runtime/dobot_state.json` 和 `data/runtime/pipette_state.json` 读取上次位置并恢复。

手动保存/更新运行时状态：

```bash
# 保存 Dobot 状态
curl -X POST http://127.0.0.1:5001/api/runtime/dobot_state \
  -H "Content-Type: application/json" \
  -d '{"j1":30,"j2":-30,"j3_deg":1000,"j4":45,"d3_mm":327.778}'

# 保存移液臂状态
curl -X POST http://127.0.0.1:5001/api/runtime/pipette_state \
  -H "Content-Type: application/json" \
  -d '{"x":310,"y":250,"z1":140,"z2":135}'
```

---

## 新增功能 (2026-06-01)

### STL 真实模型替换

Dobot M1Pro 和移液臂均使用真实 CAD 导出的 STL 零件替换了简化几何体，零件在装配体世界坐标中，通过 Rx(-90°) 旋转使机器人站立于平台。

### 移液机械臂控制面板

面板位于 Dobot 控制面板下方、"模块配置"上方：
- **关节控制**: X / Z / Y1 / Y2 四个滑块
- **笛卡尔控制**: 对应数值输入
- X: 中梁在横梁上滑动 | Z: 横梁沿框架移动 | Y1/Y2: 两个 ADP 独立升降

### 独立零件预览器 (`templates/viewers/`)

每个 STL 零件有独立 HTML 预览页，加载单个零件居中显示。另有两个装配总览页面，加载所有零件并保持装配位置。

### 平台扩展

平台从 450×450mm 扩展到 1600×1600mm，仅保留 X≥0 半区，拖拽范围同步更新。

### 调试坐标轴

原点处显示 AxesHelper(300) + X(红)/Y(绿)/Z(蓝) 文字标签（sprite），始终可见。index.html 和两个装配预览器均包含。

---

## 新增功能 (2026-06-08)

### 离线规划器 — 事件驱动 SSE 推送

数字孪生从「被动 viewer」升级为「事件驱动离线规划器」,浏览器不再轮询,后端主动推送。

**架构对比**:
```
旧 (轮询):
  浏览器每 500ms 问服务器"有新数据吗?"
  4 个请求/秒,即使什么都没发生

新 (SSE 推送):
  浏览器只开一条长连接
  服务器主动推 status / joint / tcp 事件
  空闲时 0 请求
```

**实现**:
- `GET /api/twin/stream` — SSE 长连接端点,服务器推事件
- `new EventSource(...)` — 浏览器原生支持,断线自动重连
- 4 个事件类型:`hello`(连接)/`status`(忙碌状态)/`joint`(关节角度,触发动画)/`tcp`(世界坐标,显示用)

### 字典式 pub/sub

模拟 MQTT 的主题订阅,内存 dict 实现:

```python
_topics = {"joint": [], "done": [], "anomaly": []}
_topic_last = {}   # 兜底 buffer,防止 publish 比 subscribe 早

publish(topic, data)       # 派发给所有订阅者
subscribe(topic, cb)        # 订阅
wait_event(topic, timeout)  # 阻塞等事件(可超时)
```

`/api/twin/publish/<topic>` 端点供前端 POST 消息(模拟 MQTT publish)。

**关键设计**:`_topic_last` buffer 防止 race condition。`publish` 写 buffer,`wait_event` 先查 buffer 再订阅。setTimeout/polling 都无解,只有 buffer 模式干净。

### 事件驱动 call_tool

`move_robot_arm(x,y,z,r)` 不再假等 1 秒,而是等前端动画完成后发回"done"事件:

```
[Python] move_robot_arm
  ↓
[Flask] call_tool 处理
  ↓ notify 'joint' (SSE 推)
[Browser] 收到 joint
  ↓ animateTo (800ms 动画)
  ↓ 动画完成,POST 'done'
[Flask] wait_event("done") 解除阻塞
  ↓ 返回响应 (真实动画时长)
```

**握手时间 = 真实动画时长**,不是猜的 1 秒。超时 10s 自动警告。

### 多行执行器

TCP 协议模拟区支持每行一条指令,串行执行,失败中断:

```python
move_robot_arm(220, -220, 200, 0)   # → ✓
time.sleep(1.0)                       # → ✓ 浏览器侧 sleep
move_robot_arm(220, -220, 0, 0)     # → ✗ 撞桌
move_robot_arm(300, -100, 200, 0)   # → — 跳过
move_robot_arm(400, 0, 200, 0)       # → — 跳过
```

- **运行中** `▶` 绿底闪烁
- **成功** `✓` 浅绿底
- **失败** `✗` 浅红底 + **中断后续行**
- **跳过** `—` 灰底半透明

`time.sleep(N)` / `sleep(N)` 是浏览器侧 `setTimeout`,只控制节奏,不打后端。

### 进程管理

启动时 `kill_port(5001)` 主动清理残留旧进程,退出时 `atexit` + `KeyboardInterrupt` 杀自身子进程。`debug=False` 关闭 watchdog 自动重载(避免反复 reload 卡死)。

### 坐标统一 (Z = 垂直)

整个项目统一约定 `Z = 垂直(J3 升降方向)`,X/Y 是水平面:

```
move_robot_arm(x, y, z, r)
  x = 水平 X(World X)
  y = 水平 Y(World Z)
  z = 高度(World Y, = 调试轴 Z)
  r = 绕垂直轴旋转(yaw)
```

**调试轴**(场景中红线 X、绿线 Y、蓝线 Z)与 TCP 显示的 XYZ 含义一致,刷新可看到。

**平台配置 `data/config/platform_config.json`** 里 `placement.dobot/pipette` 的 yaw 控制机器人朝向,平台所有坐标都 `+800` 平移以保证非负。

### 端点 TCP 显示

后端 `twin_call_tool` 计算出世界 TCP,通过 `tcp` 事件推给浏览器,`pv-x/y/z/r` 直接显示世界值(跟用户输入对应)。不再用前端 FK 算本地然后转换,数值始终一致。

### 自定义坐标轴

场景里的 X/Y/Z 坐标轴从 `AxesHelper` 改用 `THREE.Line` 自定义绘制,线段方向和文字标签方向 100% 对齐。

### 工作空间环

`/api/workspace` 计算的内外圆环挂在 `j1Pivot`(SCARA 真实工作面 = X-Y 水平面),随 placement yaw 旋转。**圆环在 J1 轴心位置,不是桌面**(避免之前"在桌面"导致圈在错误高度的 bug)。

### 新增端点

| 方法 | 路径 | 用途 |
|------|------|------|
| GET  | `/api/twin/health` | 健康检查 |
| GET  | `/api/twin/status` | 单次查忙碌状态 |
| GET  | `/api/twin/stream` | **SSE 长连接**,推 status/joint/tcp 事件 |
| POST | `/api/twin/execute` | raw MQTT `a{x},{y},{z},{r},{grip}` 格式 |
| POST | `/api/twin/call_tool` | Python `move_robot_arm(x,y,z,r)` 格式 |
| POST | `/api/twin/publish/<topic>` | 前端发布事件(模拟 MQTT publish) |
| GET  | `/api/runtime/{dobot,pipette}_state` | 运行时关节状态(已存在) |

### 新建/修改文件

| 文件 | 操作 |
|------|------|
| `hardware/utils/collision.py` | 新建 — 碰撞检测(Z<5mm 撞桌 / reach ∉[100,400]mm) |
| `hardware/tools/move_robot_arm.py` | 修改 — 加 `USE_MQTT` 开关 + 碰撞检查 + MQTT/Twin 双路径 |
| `digital_twin.py` | 大量修改 — SSE/事件驱动/坐标转换/进程管理 |
| `index.html` | 大量修改 — 多行/拒绝中断/time.sleep/SSE 监听/坐标轴 |
| `data/config/platform_config.json` | 全部 position Y 平移 +800,新增 placement |
| `WORK_SUMMARY.md` | Phase 6 — 离线规划器改动与教训记录 |

### 使用示例

```bash
# 1. 启动
python digital_twin/digital_twin.py
# 浏览器 http://127.0.0.1:5001

# 2. 在 TCP 协议模拟区输入多行
move_robot_arm(220, -220, 200, 0)
time.sleep(1.0)
move_robot_arm(300, -100, 250, 0)
time.sleep(1.0)
move_robot_arm(100, 200, 200, 0)

# 3. 点击"发送",看 3D 场景依次动画
#    状态条显示"中断"/"就绪",运行行绿色高亮

# 4. 通过 API 直接调用
curl -X POST http://127.0.0.1:5001/api/twin/call_tool \
  -H 'Content-Type: application/json' \
  -d '{"name":"move_robot_arm","args":[220,-220,200,0]}'
# → {"result":"... 孪生 (动画 808ms)","status":"ok"}
```

---

## 机械臂构型

### 物理结构

```
         J3 升降柱 (同轴于J1, z=0)
         ◄── 固定底座 (85mm) ──►◄──── 升降行程 300mm ────►
         ┌─────┐                ┌───────────────────────┐
  桌面 ──┤     │                │                       │
  (z=0)  │ J3  │═══════════════╪═══════════════════════╪══ 臂平面
         │电机 │  升降柱 (d₃)   │                       │  (z = 85+d₃)
         └─────┘                └───────────────────────┘
                                           │
                          J1(旋转) ────────┼──────── J2(旋转,z=-36,独立面)
                                           │
                              link1(250mm)─┴─link2(150mm)── J4(旋转)── 夹爪── TCP
```

- **J3 升降柱**与 **J1 旋转轴**同轴（z=0 垂直面），驱动丝杆 → 整个臂平面同步升降
- **J2 电机**在独立垂直面（z=-36），驱动肘关节
- **J4 电机**在腕部末端，直接驱动工具旋转

### 关节参数

| 关节 | 类型 | 范围 | 单位 | 说明 |
|------|------|------|------|------|
| J1 | 旋转 | ±85° | deg | 底座旋转，驱动大臂 |
| J2 | 旋转 | ±130° | deg | 肘关节，驱动小臂 |
| J3 | 直线 (升降柱) | ±5400° (电机) / d₃∈[150, 450] | deg·mm | Z轴升降，同轴于J1 |
| J4 | 旋转 | ±360° | deg | 腕部，工具旋转 |

### DH 参数

| 连杆 i | θᵢ | dᵢ | aᵢ | αᵢ |
|--------|-----|-----|-----|-----|
| 1 (J3) | 0 | D1+d₃ | 0 | 0° |
| 2 (J1) | θ₁* | 0 | A1=250 | 0° |
| 3 (J2) | θ₂* | 0 | A2=150 | 180° |
| 4 (J4) | θ₄* | D4=80 | 0 | 0° |

### 杆件参数

| 参数 | 值 | 说明 |
|------|-----|------|
| A1 | 250 mm | 大臂长度 (J1→J2) |
| A2 | 150 mm | 小臂长度 (J2→J4) |
| D1 | 85 mm | 固定底座高度 |
| D4 | 80 mm | 工具法兰至 TCP |
| 最大reach | 400 mm | A1+A2 |
| 最小reach | 100 mm | |A1-A2| |

### TCP 工作范围

| 轴 | 范围 | 单位 |
|-----|------|------|
| X | [-400, 400] | mm |
| Y | [-400, 400] | mm |
| Z | [155, 455] | mm |
| R | [-360, 360] | deg |

---

## 技术实现

### 技术栈

| 层 | 技术 | 用途 |
|----|------|------|
| 运动学引擎 | Python 3 + numpy | DH 参数模型、FK/IK/Jacobian/工作空间 |
| 后端 | Flask 3.x | REST API，服务静态页面 |
| 3D 渲染 | Three.js 0.160 (ES module, CDN) | WebGL 实时 3D 可视化 |
| 前端逻辑 | 原生 JavaScript (ES6+) | 60fps FK 计算、UI 交互、指令解析 |

### 运动学原理

**正向运动学 (FK)** — 关节空间 → 笛卡尔空间：

```
x  = A1·cos(θ₁) + A2·cos(θ₁+θ₂)
y  = A1·sin(θ₁) + A2·sin(θ₁+θ₂)
z  = D1 + d₃ - D4           ← 臂平面高度 - 工具偏移
R  = θ₁ + θ₂ + θ₄            ← Z轴总旋转角
```

**逆向运动学 (IK)** — 笛卡尔空间 → 关节空间，解析解：

```
d₃ = z - D1 + D4                                (升降位移)
cos(θ₂) = (x²+y²-A1²-A2²) / (2·A1·A2)           (肘角)
θ₂ = ±acos(cosθ₂)                                (肘上/肘下两支)
θ₁ = atan2(y,x) - atan2(A2·sinθ₂, A1+A2·cosθ₂)  (肩角)
θ₄ = R - θ₁ - θ₂                                 (腕角, wrap to ±180°)
```

**雅可比矩阵** — 关节速度 → 末端速度：

```
J = [ -A1·s1-A2·s12   -A2·s12   0   0   ]   ∂[vx,vy,vz,ωz]/∂[θ̇₁,θ̇₂,ḋ₃,θ̇₄]
    [  A1·c1+A2·c12    A2·c12   0   0   ]
    [      0              0     +1   0   ]   ← ∂z/∂d₃ = +1 (臂平面上升)
    [      1              1      0   1   ]
```

**J3 丝杆映射** — 协议电机角 ↔ 升降位移：

```
d₃(mm) = 300 + J3°/360 × 10
J3°    = (d₃ - 300) × 360 / 10
```

丝杆导程 10mm/rev，±5400° = 15 转 → 行程 300mm。

### 3D 场景图层次 (STL 模型)

```
scene
├── table 1600×1600 (y=0, X≥0) + grid + debug axes(XYZ labels)
├── robotRoot @ (100,0,0)
│   └── dobotPivot ← rotateX(-90°) [CAD Z→World Y]
│       ├── baseGroup
│       │   └── [AXIS1 STL] 基座，固定
│       └── liftCarriage ← translateZ(d₃)
│           ├── [AXIS2 STL] 随J3升降
│           └── j1Pivot ← rotateZ(J1)
│               ├── [AXIS3 STL] 随J3+J1
│               └── j2Pivot ← rotateZ(J2), offset(199,0,0)
│                   ├── [AXIS4 STL] 随J3+J1+J2
│                   └── j4Pivot ← rotateZ(J4), offset(196,0,0)
│                       └── [ROTATE STL] 随全部关节, 自转
└── pipetteRoot @ (1000,0,0)
    └── pipettePivot ← rotateX(-90°) [同Dobot坐标系]
        ├── pipetteBase
        │   └── [pipette_group1] 框架，固定
        └── pipetteZ ← translateY(-Z) [横梁Z向移动]
            ├── [pipette_group2] 横梁
            └── pipetteX ← translateX(X) [中梁X向滑动]
                ├── [pipette_group3] 中梁
                ├── pipetteY1 ← translateZ(Y1) [ADP1升降]
                │   ├── [pipette_group4] ADP1
                │   └── [pipette_tip] 移液头
                └── pipetteY2 ← translateZ(Y2) [ADP2升降]
                    └── [pipette_group5] ADP2
```

### 关节位置 (从 STL 重叠区提取)

| 关节 | CAD坐标 (XY) | 来源 |
|------|-------------|------|
| J1 | (209, 116) | AXIS2∩AXIS3 重叠区中心 X[164,254] |
| J2 | (408, 116) | AXIS3∩AXIS4 重叠区中心 X[372,443] |
| J4 | (604, 116) | ROTATE 自身中心 |

J4 偏移量 (0,0) → ROTATE 绕自身中心自转。

### 开发方法

**运动学调试** — 独立运行 kinematics.py 验证 FK/IK：

```bash
python digital_twin/robot_arm_model/kinematics_M1Pro.py
```

**API 调试** — 通过 Flask test client 或 curl 测试：

```bash
curl -X POST http://127.0.0.1:5001/api/fk \
  -H "Content-Type: application/json" \
  -d '{"j1":0,"j2":0,"j3_deg":0,"j4":0}'

curl -X POST http://127.0.0.1:5001/api/ik \
  -H "Content-Type: application/json" \
  -d '{"x":200,"y":100,"z":305,"r":0}'
```

**3D 调试** — 浏览器 F12 Console 可访问所有核心函数：

```javascript
updateRobot(30, -60, 0, 90)  // 直接设置关节角
fk(30, -60, 300, 90)          // FK 计算
ik(200, 100, 305, 0, true)    // IK 计算
```

**参数修改** — 需同步 3 处：
1. `kinematics.py` — 常量定义
2. `app.py` — import 声明 + print 信息
3. `templates/index.html` — JS 常量 `const A1=...`

### 坐标系约定

| | 机器人空间 | Three.js 空间 |
|--|-----------|--------------|
| X | 前方 (手臂伸展方向) | +X (右) |
| Y | 左方 | +Z (前) |
| Z | 上方 (垂直) | +Y (上) |

Three.js 使用 Y-up 坐标，机器人 Z 轴映射到 Three.js Y 轴。

---

## API 参考

所有端点返回 JSON。Dobot 运动学由 `kinematics.py` 完成，移液臂运动学由 `kinematics_pipette.py` 完成，前端 3D 渲染使用同逻辑的 JS 实现（60fps，不依赖后端）。

| 方法 | 路由 | 输入 | 输出 |
|------|------|------|------|
| GET | `/` | — | 3D 可视化页面 |
| POST | `/api/fk` | `{j1,j2,j3_deg,j4}` | `{x,y,z,r,d3_mm,joint_limits_ok}` |
| POST | `/api/ik` | `{x,y,z,r,elbow_up?}` | `{solutions:[{j1,j2,d3,j4,elbow_up,valid,verify_error_mm}]}` |
| GET | `/api/workspace` | — | `{outer:{x[],y[]}, inner:{x[],y[]}, params:{a1,a2,reach_max,reach_min}}` |
| GET | `/api/joint_limits` | — | `{joints:{j1/j2/j3/j4:{min,max,unit,type}}, dh_params, screw}` |
| POST | `/api/jacobian` | `{j1,j2,j3_deg,j4}` | `{jacobian:4x4, determinant}` |
| GET | `/api/pose` | — | 服务端缓存的当前状态 |
| **移液臂运动学** | | | |
| GET | `/api/pipette/limits` | — | `{x, y, z1, z2, adp_spacing_x_mm}` |
| POST | `/api/pipette/fk` | `{x, y, z1, z2}` | `{tip1, tip2} CAD+World坐标` |
| POST | `/api/pipette/ik` | `{x, y, z1, z2}` | `{x,y,z1,z2, valid, limit_violations}` |
| GET | `/api/pipette/pose` | — | 服务端缓存的当前状态 |
| **平台配置** | | | |
| GET | `/api/platform_config` | — | 读取平台模块配置 |
| POST | `/api/platform_config` | JSON body | 保存平台模块配置到 `platform_config.json` |
| GET | `/api/layout/placement` | — | `{placement:{dobot:{x,z,yaw}, pipette:{x,z,yaw}}}` |
| POST | `/api/layout/placement` | JSON body | 只更新 placement,保留其他配置 |
| **离线规划器 (SSE + pub/sub)** | | | |
| GET | `/api/twin/health` | — | `{status:"ok"}` |
| GET | `/api/twin/status` | — | `{status:"idle"\|"busy"\|"error", current_task, reason}` |
| GET | `/api/twin/stream` | — | **SSE 长连接**,推 `hello`/`status`/`joint`/`tcp` 事件 |
| POST | `/api/twin/execute` | `{msg:"a{x},{y},{z},{r},{grip}"}` | raw MQTT 格式,事件驱动,等前端 `done` |
| POST | `/api/twin/call_tool` | `{name:"move_robot_arm", args:[x,y,z,r]}` | Python 风格,事件驱动,返回 `{result, status}` |
| POST | `/api/twin/publish/<topic>` | 任意 JSON | 前端发布消息(模拟 MQTT publish) |

### IK 肘部选择

SCARA 的 IK 有两个解：肘部在上 (elbow up) 和肘部在下 (elbow down)。`/api/ik` 默认返回两支解。前端可通过下拉框选择。

---

## TCP 协议模拟

前端内建指令解释器，模拟越疆 TCP/IP 远程控制接口（4轴，30003 端口）。

| 指令 | 参数 | 说明 |
|------|------|------|
| `MovJ(X,Y,Z,R)` | 笛卡尔坐标 | 关节运动到目标点 |
| `MovL(X,Y,Z,R)` | 笛卡尔坐标 | 直线运动到目标点 |
| `JointMovJ(J1,J2,J3,J4)` | 关节坐标 | 关节空间运动 |
| `MovJExt(distance)` | 距离(mm) | Z 轴升降柱运动 |

响应格式遵循协议规范：`ErrorID,{返回值},指令名(参数);`

---

## 后续拓展方法

### 1. 接入物理硬件

项目中已有 `hardware/platform_csharp/` 目录（Dobot C# TCP SDK）。将数字孪生的指令输出对接真实机械臂：

```python
# 在 digital_twin.py 添加 WebSocket 端点，将前端指令转发到真实机械臂
# 同步数字孪生状态到真实硬件反馈
```

### 2. 动力学仿真

当前仅包含运动学。添加动力学需补充：
- 各连杆质量、惯量、质心位置（需实测或 CAD 数据）
- 牛顿-欧拉逆动力学递推
- 重力补偿、摩擦力模型
- 轨迹规划（梯形速度/S曲线）

### 3. 碰撞检测

- 使用 Three.js Raycaster 或集成专用库（如 Oimo.js/Ammo.js）
- 在机器人工作空间添加障碍物
- 路径规划时自动避障

### 4. 轨迹规划与回放

- 录制关节轨迹到 JSON
- 时间轴回放控制（播放/暂停/速度调节）
- 多点插值（直线/圆弧/样条）

### 5. 与项目其他模块集成

- **实验设计模块** (`experiment/`)：将机械臂作为硬件工具纳入实验步骤
- **硬件控制** (`hardware/tools/`)：注册为 `move_robot_arm` 工具
- **算法模块** (`software/algorithms/`)：添加路径优化/抓取规划算法

### 6. 平台布局数字孪生

3D 场景中 6 模块系统，支持编辑模式、参数化配置、多实例复制、配置持久化。

#### 模块清单

| # | 模块 | 物理 | 说明 |
|---|------|------|------|
| 1 | 机械臂 | — | 只读信息面板（DH参数、关节限位） |
| 2 | 桌面 | 可拖拽 | 平台底板 + 网格 + 边界线 |
| 3 | 载玻片盒 | 可拖拽 | 盒体 + 半透明载玻片，片尺寸W×L可配 |
| 4 | 匀胶机 | 可拖拽 | 底座 + 转盘 + 卡盘，支持多台 |
| 5 | 滴管头盒 | 可拖拽 | 盒体 + 滴管孔位(8×12=96)，行列间隔可分别定义 |
| 6 | 溶液瓶托盘 | 可拖拽 | 实心长方体 + 圆形凹槽(4×5=20)，瓶直径可配 |

#### 交互

| 操作 | 触发方式 |
|------|---------|
| 拖拽模块 | 左键拖拽 3D 对象（50mm 网格吸附，范围 ±500mm） |
| 添加实例 | 编辑模式 OFF → 双击右侧面板模块名 |
| 删除实例 | 编辑模式 OFF → 右键 3D 模块 / 点击面板 × 按钮 |
| 编辑参数 | 勾选「编辑模式」→ 切换 tab → 修改参数（即时重建） |
| 保存/加载 | 「保存配置」按钮 → `platform_config.json`；页面启动自动加载 |

#### 拖动防散架

自定义 `setupModuleDrag()` 替代 Three.js `DragControls`：射线命中子对象（载玻片、滴管孔）时沿 `.parent` 链向上查找，始终拖动顶层模块 Group。

#### 配置持久化

- 文件 `platform_config.json`：存储所有物理模块的模板参数 + `positions` 数组
- API: `GET /api/platform_config` / `POST /api/platform_config`
- 前端启动自动加载，点击「保存配置」写入文件

#### 面板布局

```
面板:
├── 末端位姿 (TCP) 显示
├── 关节控制 / 笛卡尔控制 (折叠tab)
├── TCP 协议模拟
├── 模块配置
│   ├── [保存配置] [编辑模式 □]
│   ├── 编辑OFF: 6模块摘要列表（双击添加，×删除）
│   └── 编辑ON:  6个tab → 各模块参数面板
└── 参数信息
```

### 7. 前端增强

- 末端拖拽交互（鼠标拖动 TCP → 实时 IK → 更新关节）
- VR/AR 预览（WebXR API）
- 多机械臂协同场景
