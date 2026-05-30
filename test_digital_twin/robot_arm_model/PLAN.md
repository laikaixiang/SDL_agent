# Dobot M1Pro SCARA 机械臂 — 数字孪生

## 技术路线

### 运动学模型

基于 **Standard DH (Denavit-Hartenberg)** 参数法建立 SCARA 运动学模型。

| 连杆 i | θᵢ (关节变量) | dᵢ (偏距) | aᵢ (杆长) | αᵢ (扭转角) |
|--------|---------------|-----------|-----------|-------------|
| 1 | **θ₁*** | D1=330mm | A1=250mm | 0° |
| 2 | **θ₂*** | 0 | A2=150mm | 180° |
| 3 | 0 | **d₃*** | 0 | 0° |
| 4 | **θ₄*** | D4=80mm | 0 | 0° |

- A1+A2 = 400mm = 公开资料中的最大工作半径 ✓
- Z 行程: d₃∈[5, 245mm] → 末端高度 z=D1-d₃-D4∈[5, 245mm] ✓
- J3 协议角度 ↔ d₃ 转换: 丝杆导程 10mm/rev

### 关节限位（来源：公开资料+协议文档）

| 关节 | 类型 | 范围 | 单位 |
|------|------|------|------|
| J1 | 旋转 | ±85° | deg |
| J2 | 旋转 | ±130° | deg |
| J3 | 直线(Z轴) | 5~245mm | mm |
| J4 | 旋转 | ±360° | deg |

### 两种控制模式

1. **关节空间控制 (电机模式)**: 调节 J1/J2/J3/J4 各关节角度/Z位移，实时正向运动学(FK)计算末端位姿
2. **笛卡尔空间控制 (XYZR模式)**: 输入目标 XYZR，逆向运动学(IK)求解关节角，解析解（分肘部在上/在下两支）

### TCP 协议模拟

按照越疆 TCP/IP 远程控制接口文档（4轴）实现常用运动指令的解释和模拟：
- `MovJ(X,Y,Z,R)` — 关节运动到笛卡尔目标点（30003端口）
- `MovL(X,Y,Z,R)` — 直线运动到笛卡尔目标点
- `JointMovJ(J1,J2,J3,J4)` — 关节运动到关节目标点
- `MovJExt(distance)` — 控制Z轴直线电机（扩展轴）

### 3D 可视化

- Three.js 构建机器人几何模型，层次结构对应运动学链
- 底座→J1旋转→大臂→J2旋转→小臂→Z轴平移→J4旋转→末端工具
- 工作空间圆环叠加（最大400mm / 最小100mm）
- 导轨视角控制（OrbitControls）
- 60fps 实时 FK 计算（JS 端，无需后端）

## 运行方式

```bash
cd test_digital_twin/robot_arm_model
python digital_twin.py
# 浏览器打开 http://127.0.0.1:5001
```

无需额外安装依赖（Flask + Three.js CDN），零配置启动。

## 文件清单

```
test_digital_twin/robot_arm_model/
├── PLAN.md              # 本文件
├── kinematics.py        # 运动学引擎（DH/FK/IK/Jacobian/工作空间）
├── app.py               # Flask 服务端（REST API）
├── templates/
│   └── index.html       # 3D 交互界面（Three.js 自包含）
└── dobot_protocol.txt   # 协议文档提取（参考）
```

## API 端点

| 方法 | 路由 | 说明 |
|------|------|------|
| GET | `/` | 3D 可视化页面 |
| POST | `/api/fk` | 正向运动学 {j1,j2,j3_deg,j4} → {x,y,z,r} |
| POST | `/api/ik` | 逆向运动学 {x,y,z,r} → 两支解 |
| GET | `/api/workspace` | 工作空间边界数据 |
| GET | `/api/joint_limits` | 关节限位和DH参数 |
| POST | `/api/jacobian` | 雅可比矩阵计算 |
| GET | `/api/pose` | 当前机器人状态 |
