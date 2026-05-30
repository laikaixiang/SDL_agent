# Dobot M1Pro SCARA 机械臂 — 数字孪生

基于越疆 M1Pro 四轴 SCARA 协作机械臂的物理仿真数字孪生，支持关节空间/笛卡尔空间控制、TCP 协议模拟和实时 3D 可视化。

## 快速启动

```bash
cd test_digital_twin/robot_arm_model
python digital_twin.py
# 浏览器打开 http://127.0.0.1:5001
```

**零额外依赖** — Flask（项目已有）+ Three.js（CDN）+ numpy（项目已有）。

---

## 目录结构

```
test_digital_twin/robot_arm_model/
├── README.md              # 本文件
├── PLAN.md                # 技术方案文档
├── kinematics.py          # 运动学引擎
├── app.py                 # Flask 服务端
├── templates/
│   └── index.html         # 3D 交互界面 (Three.js, 自包含)
└── dobot_protocol.txt     # 越疆 TCP/IP 协议文档 (参考)
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

### 3D 场景图层次

```
scene
├── table (y=0) + grid + workspace rings
└── robotRoot
    └── baseGroup (固定)
        ├── 底座板 + 固定立柱 (0→85mm)
        ├── J3 电机 (y≈85-119, z=0)
        └── liftCarriage ← translateY(d₃)
            ├── 升降柱杆 (y=0→d₃, 动态伸缩)
            └── shoulderPlat (臂平面)
                ├── J1电机 (z=0) + J2电机 (z=-36)
                └── j1Pivot ← rotateY(θ₁)
                    ├── link1 (大臂 250mm) + 同步带罩
                    └── j2Pivot ← rotateY(θ₂)
                        ├── link2 (小臂 150mm) + 同步带罩
                        └── j4Pivot ← rotateY(θ₄)
                            ├── J4电机 + 法兰
                            ├── 平行爪夹持器
                            └── TCP 球 + 十字准星 (y=-D4=-80)
```

### 开发方法

**运动学调试** — 独立运行 kinematics.py 验证 FK/IK：

```bash
python test_digital_twin/robot_arm_model/kinematics.py
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

所有端点返回 JSON。运动学计算由 `kinematics.py` 完成，前端 3D 渲染使用同逻辑的 JS 实现（60fps，不依赖后端）。

| 方法 | 路由 | 输入 | 输出 |
|------|------|------|------|
| GET | `/` | — | 3D 可视化页面 |
| POST | `/api/fk` | `{j1,j2,j3_deg,j4}` | `{x,y,z,r,d3_mm,joint_limits_ok}` |
| POST | `/api/ik` | `{x,y,z,r,elbow_up?}` | `{solutions:[{j1,j2,d3,j4,elbow_up,valid,verify_error_mm}]}` |
| GET | `/api/workspace` | — | `{outer:{x[],y[]}, inner:{x[],y[]}, params:{a1,a2,reach_max,reach_min}}` |
| GET | `/api/joint_limits` | — | `{joints:{j1/j2/j3/j4:{min,max,unit,type}}, dh_params, screw}` |
| POST | `/api/jacobian` | `{j1,j2,j3_deg,j4}` | `{jacobian:4x4, determinant}` |
| GET | `/api/pose` | — | 服务端缓存的当前状态 |

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

### 6. 前端增强

- 末端拖拽交互（鼠标拖动 TCP → 实时 IK → 更新关节）
- VR/AR 预览（WebXR API）
- 多机械臂协同场景
