# 双机械臂平台数字孪生 — 工作记录

## 完成时间
2026-05-31

## 工作概述

1. 将 Dobot M1Pro STEP 拆分为 5 个 STL 零件，替换 index.html 中简化几何体为真实 STL 装配体
2. 添加 XYZZ+双ADP 移液机械臂（6 个 STL 零件）作为静态装配体到平台对角
3. 移液臂 STL 减面（1.4M → 12K 三角面），适配网页加载
4. 平台从 450×450mm 扩展到 800×800mm

---

## 文件结构

```
test_digital_twin/robot_arm_model/
├── digital_twin.py              # Flask 服务器
├── kinematics.py                 # Dobot M1Pro 运动学引擎 (FK/IK/Jacobian)
├── verify_stl_workflow.py       # STL 验证脚本
├── simplify_pipette_meshes.py   # 移液臂 STL 减面脚本
├── extract_joint_offsets.py     # Dobot 关节偏移提取脚本
├── dobot_joint_offsets.json     # 关节偏移数据
├── platform_config.json         # 平台模块配置持久化
├── templates/
│   ├── index.html               # 主 3D 场景（双机械臂 + 可拖拽模块）
│   ├── models/
│   │   ├── dobot_m1pro/
│   │   │   ├── Dobot_M1Pro.step        # 原始 STEP 文件 (7.1 MB)
│   │   │   ├── Dobot M1Pro.stp          # 原始 STEP 文件 (7.4 MB)
│   │   │   ├── M1_AXIS1_1.stl          # 基座 (2.84 MB, 56,786 tris)
│   │   │   ├── M1_AXIS2_1.stl          # 第二轴 (675 KB, 13,506 tris)
│   │   │   ├── M1_AXIS3_1.stl          # 第三轴 (1.36 MB, 27,884 tris)
│   │   │   ├── M1_AXIS4_1.stl          # 第四轴 (793 KB, 15,852 tris)
│   │   │   └── M1_ROTATE_1.stl         # 旋转关节 (41 KB, 820 tris)
│   │   └── pipette_arm/
│   │       ├── pipette_group1.stl       # 原始 (39.4 MB, 826K tris)
│   │       ├── pipette_group2.stl       # 原始 (21.8 MB, 457K tris)
│   │       ├── pipette_group3.stl       # 原始 (3.2 MB, 68K tris)
│   │       ├── pipette_group4.stl       # 原始 (1.4 MB, 29K tris)
│   │       ├── pipette_group5.stl       # 原始 (1.2 MB, 25K tris)
│   │       ├── pipette_tip.stl          # 原始 (51 KB, 1K tris)
│   │       └── simplified/
│   │           ├── pipette_group1.stl   # 减面后 (237 KB, 4,850 tris)
│   │           ├── pipette_group2.stl   # 减面后 (109 KB, 2,222 tris)
│   │           ├── pipette_group3.stl   # 减面后 (81 KB, 1,646 tris)
│   │           ├── pipette_group4.stl   # 减面后 (94 KB, 1,932 tris)
│   │           ├── pipette_group5.stl   # 减面后 (65 KB, 1,324 tris)
│   │           └── pipette_tip.stl      # 减面后 (17 KB, 340 tris)
│   └── viewers/
│       ├── M1_ROTATE_1_viewer.html
│       ├── M1_AXIS1_1_viewer.html
│       ├── M1_AXIS2_1_viewer.html
│       ├── M1_AXIS3_1_viewer.html
│       ├── M1_AXIS4_1_viewer.html
│       ├── assembly_viewer.html              # Dobot 装配总览
│       ├── pipette_group1_viewer.html        # 移液臂 主框架
│       ├── pipette_group2_viewer.html        # 移液臂 横梁
│       ├── pipette_group3_viewer.html        # 移液臂 中梁
│       ├── pipette_group4_viewer.html        # 移液臂 ADP-1
│       ├── pipette_group5_viewer.html        # 移液臂 ADP-2
│       ├── pipette_tip_viewer.html           # 移液臂 移液头
│       └── pipette_assembly_viewer.html      # 移液臂 装配总览
└── LOG_20260531_lkx.md
```

---

## 启动方式

```bash
cd D:/PycharmProjects/sdl_agent/digital_twin/robot_arm_model
python digital_twin.py
# 打开 http://127.0.0.1:5001
```

---

## Flask 路由

| 路由 | 功能 |
|------|------|
| `GET /` | 主页面 index.html |
| `GET /models/<path:filename>` | 服务 STL/STEP 文件（支持子目录） |
| `GET /viewers/<filename>` | 服务 HTML 预览文件 |
| `GET /api/fk` | 正向运动学 (POST JSON) |
| `GET /api/ik` | 逆向运动学 (POST JSON) |
| `GET /api/workspace` | 工作空间边界 |
| `GET /api/joint_limits` | 关节限位 |
| `GET /api/jacobian` | 雅可比矩阵 |
| `GET /api/pose` | 当前机器人状态 |
| `GET/POST /api/platform_config` | 平台模块配置持久化 |

---

## Dobot M1Pro — STL 替换详情 (Phase 1)

### 运动学层级
```
robotRoot (场景位置)
  └── baseGroup (固定)
        ├── [AXIS1 STL: 基座]
        └── liftCarriage (J3 升降 d₃)
              ├── [AXIS3 STL: 升降部件]
              └── j1Pivot (J1 旋转)
                    ├── j2Pivot (J2 位置, A1=250mm)
                    │     ├── [AXIS2 STL]
                    │     └── j4Pivot (J4 位置, A2=150mm)
                    │           ├── [AXIS4 STL]
                    │           ├── [ROTATE STL]
                    │           └── TCP 球体
```

### 关节参数
- DH: a₁=250mm, a₂=150mm, d₁=85mm, d₄=80mm
- J1: ±85° | J2: ±130° | J3: d₃=150-450mm | J4: ±360°
- 臂展: 100-400mm | Z 行程: 155-455mm

### 零件 → 关节映射
| STL 零件 | 父关节 | 说明 |
|----------|--------|------|
| M1_AXIS1_1 | baseGroup | 基座，固定不动 |
| M1_AXIS3_1 | liftCarriage | 随 J3 升降 |
| M1_AXIS2_1 | j2Pivot | 随 J2 旋转 |
| M1_AXIS4_1 | j4Pivot | 随 J4 旋转 |
| M1_ROTATE_1 | j4Pivot | 末端旋转关节 |

---

## 移液机械臂 — 添加详情 (Phase 2)

### 基本信息
- 型号: XYZZ+双ADP 移液机械臂 (2025.03.13)
- 构型: XYZ 龙门 + 双 ADP 移液模块
- 底座: ~423×782mm | 高度: ~474mm
- 加载方式: 6 个减面 STL 作为静态装配体，顶点已在装配坐标中
- 装配体坐标: X[4.6–427.7] Y[3.1–477.3] Z[14.8–796.4]

### 装配零件
| STL 零件 | 三角面 | X 范围 | Y 范围 | Z 范围 | 说明 |
|----------|--------|--------|--------|--------|------|
| pipette_group1 | 4,850 | 339.5–427.7 | 3.1–423.1 | 14.8–796.4 | 主框架/立柱 |
| pipette_group2 | 2,222 | 4.6–424.4 | 360.1–472.1 | 52.5–197.3 | 横梁 |
| pipette_group3 | 1,646 | 260.8–339.8 | 221.1–477.3 | 59.3–195.4 | 中梁 |
| pipette_group4 | 1,932 | 273.5–295.8 | 196.1–400.1 | 88.3–157.8 | ADP模块1 |
| pipette_group5 | 1,324 | 303.8–325.8 | 197.5–400.1 | 88.3–157.8 | ADP模块2 |
| pipette_tip   |   340 | 281.2–288.4 | 108.1–204.1 | 108.6–116.0 | 移液头 |

### STL 减面
- 算法: 顶点聚类 (numpy + struct，零依赖)
- 脚本: `simplify_pipette_meshes.py`
- 原始: 1,406,106 tris / 67 MB → 减面后: 12,314 tris / ~600 KB

### 当前状态
- 在 index.html 中作为 `pipetteRoot` group 加载
- 加到了 `moduleDefs` / `moduleOrder`，UI 有只读信息卡片
- 位置: 平台右上对角（可在编辑模式下调节）
- **运动学参数未配置，仅作静态展示**

---

## 平台模块系统

| 模块 | 类型 | 说明 |
|------|------|------|
| robotArm | 非物理 | Dobot M1Pro SCARA（可运动学控制） |
| pipetteArm | 非物理 | 移液机械臂（静态展示） |
| desktop | 物理 | 800×800mm 桌面平台，16×16 网格 |
| slideBox | 物理 | 载玻片盒，可拖拽 |
| spinCoater | 物理 | 匀胶机，可拖拽 |
| dropperBox | 物理 | 滴管头盒，可拖拽 |
| solutionTray | 物理 | 溶液瓶托盘，可拖拽 |

拖拽吸附网格: 50mm | 配置持久化: `platform_config.json`

---

## 收到移液臂 STEP 文件后应执行的操作

> 文件: `XYZZ＋双ADP移液机械臂（2025.03.13）.STEP`
> 状态: **已完成 Step 1–3**，Step 4–5 待定

### Step 1 — 存放文件 ✅
文件已在 `templates/models/pipette_arm/XYZZ+双ADP移液机械臂（2025.03.13）.STEP`

### Step 2 — 拆分 STL ✅
原始 STL（6个零件）在 `templates/models/pipette_arm/`，减面版本在 `simplified/` 子目录。
零件在装配体世界坐标中，无需额外拆分。

### Step 3 — 分析装配结构 ✅
装配体坐标范围: X[4.6–427.7] Y[3.1–477.3] Z[14.8–796.4]
底座: ~423×782mm | 高度: ~474mm
6 个零件详细数据见上方表格。
装配预览: `pipette_assembly_viewer.html`

### Step 4 — 确定运动学参数 ✅ (2026-06-01)
从移液臂 STL 装配件包围盒提取机械限位:

| 轴 | 类型 | 机械限位 | 行程 | 参考位 | 方向 |
|------|------|---------|------|--------|------|
| X | 棱柱 | [44, 385] mm | 341mm | 300.3 | 水平, 中梁在横梁上滑动 |
| Y | 棱柱 | [59, 367] mm | 308mm | 213.1 | 水平, 横梁沿框架移动 |
| Z1 | 棱柱 | [94, 161] mm | 67mm | 123.1 | **垂直↑**, ADP1升降 |
| Z2 | 棱柱 | [94, 161] mm | 67mm | 123.1 | **垂直↑**, ADP2升降 |

ADP间距: 30.2mm (X方向)
原Z_ref=416.1超出机械上限49mm，修正为框架中心213.1

### Step 5 — 添加运动学支持 ✅ (2026-06-01)
1. `kinematics_pipette.py` — 移液臂运动学引擎 (FK/IK/工作空间/限位)
2. `digital_twin.py` — 新增4个API路由:
   - `GET /api/pipette/limits` — 四轴限位+ADP间距
   - `POST /api/pipette/fk` — 正向运动学 (轴值→两tip的CAD/World坐标)
   - `POST /api/pipette/ik` — 逆向运动学 (clamp到限位+越限警告)
   - `GET /api/pipette/pose` — 当前状态
3. `index.html` — 变量重命名 + 滑块范围修正为机械限位
4. `pipette_kinematic_params.json` — STL分析结果
5. `extract_pipette_params.py` — STL参数提取脚本

---

## 关键文件清单

| 文件 | 用途 |
|------|------|
| `templates/index.html` | 主场景：双机械臂 + 平台模块 + 关节控制 |
| `templates/models/dobot_m1pro/` | Dobot M1Pro 原始 STL + STEP |
| `templates/models/pipette_arm/` | 移液臂原始 STL（备份） |
| `templates/models/pipette_arm/simplified/` | 移液臂减面 STL（网页加载用） |
| `templates/viewers/assembly_viewer.html` | Dobot 装配预览 |
| `templates/viewers/pipette_assembly_viewer.html` | 移液臂装配预览 |
| `simplify_pipette_meshes.py` | 减面脚本 |
| `extract_joint_offsets.py` | Dobot 关节偏移提取 |
| `dobot_joint_offsets.json` | Dobot 关节偏移数据 |
| `kinematics.py` | Dobot M1Pro 运动学引擎 |
| `digital_twin.py` | Flask 服务器 |
| `platform_config.json` | 平台模块配置 |

---

## Phase 3: STL 集成到 index.html (2026-06-01)

### 完成内容
1. Dobot M1Pro STL 零件加载 + 运动学层级绑定
2. 移液臂 STL 零件加载 + 运动学层级 + 关节控制面板
3. 平台从 450×450 扩展到 1600×1600，X≥0
4. 移除雾效，无限渲染距离
5. 原点调试坐标轴 (AxesHelper + X/Y/Z 文字标签)
6. 移液臂的 X/Z/Y1/Y2 滑块控制面板
7. 两个装配预览器的坐标轴标签

### Dobot 运动学层级 (最终版)

```
baseGroup
  ├── [AXIS1 STL] 固定
  └── liftCarriage (J3 Z升降)
        ├── [AXIS2 STL] 仅随J3 → AXIS2不跟J1旋转
        └── j1Pivot (J1 绕Z旋转)
              ├── [AXIS3 STL] 随J3+J1
              └── j2Pivot (J2, offset 199mm)
                    ├── [AXIS4 STL] 随J3+J1+J2
                    └── j4Pivot (J4, offset 196mm)
                          └── [ROTATE STL] 自转(offset = 0,0)
```

### 关节位置提取方法

从 STL 文件解析三角面顶点，计算各零件的包围盒 (CAD 世界坐标)。关节 pivot 位置通过**零件重叠区中心**确定：

| 关节 | 重叠区 | 中心坐标 |
|------|--------|---------|
| J1 | AXIS2∩AXIS3 X[164, 254] | (209, 116) |
| J2 | AXIS3∩AXIS4 X[372, 443] | (408, 116) |
| J4 | ROTATE 自身 | (604, 116) |

### 移液臂运动学层级

```
pipetteBase (框架固定)
  └── pipetteZ (Z向移动: 横梁+所有沿框架)
        ├── [group2] 横梁
        └── pipetteX (X向移动: 中梁在横梁上滑动)
              ├── [group3] 中梁
              ├── pipetteY1 (Y1升降: ADP1)
              │     ├── [group4] ADP1
              │     └── [tip] 移液头
              └── pipetteY2 (Y2升降: ADP2)
                    └── [group5] ADP2
```

---

## 踩坑记录

### 1. Viewer HTML 路径错误
**症状**: 单零件 viewer 加载 STL 时 404
**原因**: viewer 在 `viewers/` 目录，STL 在 `models/` 目录，路径写成了 `./models/` 而非 `../models/`
**修复**: 全部改为 `../models/dobot_m1pro/`

### 2. JS 双闭合括号 `}}` 导致模块静默失败
**症状**: AXIS1/3/4 viewer 一直显示"加载中..."，AXIS2 正常
**原因**: `MeshStandardMaterial({...}})` 和错误回调 `})}` 中的多余 `}` 导致 `<script type="module">` 解析失败，整个模块不执行
**修复**: 对比 AXIS2 逐行修复多余的 `}`
**教训**: ES module 中语法错误会使整个模块静默失败，浏览器 F12 才能看到

### 3. `THREE.STLLoader` 不存在
**症状**: index.html 白屏，无任何显示
**原因**: `STLLoader` 是命名导入 `import { STLLoader }`，不是 `THREE` 的属性。`new THREE.STLLoader()` → TypeError
**修复**: 改为 `new STLLoader()`

### 4. Rx(-90°) 旋转后的坐标轴映射
**症状**: Dobot 散架，关节运动方向错误
**原因**: `dobotPivot` 有 Rx(-90°) 旋转，内部 local Z = world Y (上)，但 updateRobot 仍用 `.position.y` / `.rotation.y` 操作
**修复**: 全部改为 `.position.z` / `.rotation.z`
**原则**: 两种方案选一 — (A) 旋转组 + Z轴操作 或 (B) 不旋转 + 手动变换 STL。本工程选 A

### 5. 移液臂坐标系反复调整
**过程**: 
- 初版用 Rx(-90°)，但最长轴(CAD Z 782mm)变成 World Y (朝上)，用户要求最长轴在 X
- 改为 Ry(-90°)，CAD Z→World X，但 X/Z 方向与 Dobot 不一致
- 最终统一使用 Rx(-90°) (与 Dobot 相同)，两机械臂共享坐标系
**当前约定**: X=水平臂展方向, Y=垂直朝上, Z=水平垂直方向

### 6. Y↔Z 轴命名交换引发的问题
**症状**: 交换后 index.html 完全不显示机械臂
**原因**: `git checkout` 回滚后发现原始提交版本使用几何体而非 STL 模型。重新应用所有 STL 改动后恢复
**教训**: 大范围重命名前确认当前文件状态，用 git 管理中间版本

### 7. CanvasTexture Sprite 标签初始化
**症状**: 可能在某些环境下抛异常阻断模块
**修复**: 将 sprite 标签创建包装在 try-catch 中

### 8. pipette 零件加载方式演进
- 初版: 居中几何体 + 偏移量 (复杂易错)
- 终版: 直接加载原始 CAD 顶点，mesh.position=(0,0,0)，父节点位置控制运动 (简洁可靠)
**原则**: STL 顶点保留 CAD 世界坐标，运动由父 Group 的 position 驱动，不在 mesh 层做位置变换

---

## 关键总结：STL 传输链路 & 虚拟关节定义

### 一、STL 文件如何从服务端传输到 HTML

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 文件存储                                                  │
│    templates/models/dobot_m1pro/M1_AXIS1_1.stl  (Binary STL) │
│    templates/models/pipette_arm/simplified/pipette_group1.stl │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│ 2. Flask 路由 (digital_twin.py:50-55)                       │
│    @app.route('/models/<path:filename>')                    │
│    def serve_model(filename):                               │
│        return send_from_directory(models_dir, filename)      │
│                                                             │
│    → GET /models/dobot_m1pro/M1_AXIS1_1.stl                │
│    → 读取 templates/models/dobot_m1pro/M1_AXIS1_1.stl      │
│    → 返回 binary/octet-stream + Content-Length              │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP 200 + binary STL data
┌──────────────────────────▼──────────────────────────────────┐
│ 3. 浏览器端加载 (index.html:329-340)                        │
│    import { STLLoader } from 'three/.../STLLoader.js'       │
│    const STL_LOADER = new STLLoader();                      │
│                                                             │
│    STL_LOADER.load('/models/dobot_m1pro/M1_AXIS1_1.stl',   │
│      (geometry) => {   ← 成功回调                           │
│        // geometry 是 THREE.BufferGeometry                  │
│        // 顶点已在 CAD 装配体世界坐标中                       │
│        geometry.computeBoundingBox();                       │
│        geometry.translate(-cx, -cy, -cz); // 居中几何体      │
│        const mesh = new THREE.Mesh(geometry, material);      │
│        parent.add(mesh);  ← 挂载到运动学节点                 │
│      },                                                    │
│      undefined,         ← onProgress (不使用)               │
│      (err) => {...}     ← 失败回调                          │
│    );                                                      │
└─────────────────────────────────────────────────────────────┘
```

**关键点**:
- STL 文件从 STEP 装配体导出时保留世界坐标 → 所有零件在同一个 CAD 坐标系中
- Flask `send_from_directory` 直接发送二进制文件，无转换
- Three.js `STLLoader` 异步 fetch → 解析 binary STL → 生成 BufferGeometry
- 几何体居中 (`geometry.translate`) 后 mesh.position 偏移到运动学节点本地坐标

### 二、Dobot M1Pro 虚拟关节定义

#### 关节层级树

```
robotRoot (场景位置 x=100, z=0)
└── dobotPivot ← rotation.x = -90°  (CAD Z-up → World Y-up)
    │                                 内部所有坐标系为 CAD 坐标 (Z=上)
    ├── baseGroup                     [固定不动]
    │   └── mesh(AXIS1) @ (88, 116, 344)
    │
    └── liftCarriage ← position.z = d₃  ← J3 升降电机
        │   范围: d₃ ∈ [150, 450] mm
        │   丝杆: d₃ = 300 + J3_deg/360 × 10
        │
        ├── mesh(AXIS2) @ (189, 116, 440.67 - d₃)  仅随 J3
        │
        └── j1Pivot ← rotation.z = J1°              ← J1 旋转电机
            │   范围: ±85°
            │   位置: CAD (209, 116, 141)  [AXIS2∩AXIS3 重叠区中心]
            │
            ├── mesh(AXIS3) @ (304-209, 116-116, 418-441)  随 J3+J1
            │
            └── j2Pivot ← rotation.z = J2°          ← J2 旋转电机
                │   范围: ±130°
                │   偏移: (199, 0, 0)  [J2-J1 = 408-209]
                │
                ├── mesh(AXIS4) @ (504-408, 115-116, 381-441)  随 J3+J1+J2
                │
                └── j4Pivot ← rotation.z = J4°      ← J4 旋转电机
                    │   范围: ±360°
                    │   偏移: (196, 0, 0)  [J4-J2 = 604-408]
                    │
                    └── mesh(ROTATE) @ (604-604, 116-116, 309-441)
                         ↑ XY偏移 = (0,0) → 自转!
```

#### 每个关节的定义要素

| 要素 | J3 (升降) | J1 (肩) | J2 (肘) | J4 (腕) |
|------|----------|---------|---------|---------|
| **类型** | 直线 (prismatic) | 旋转 (revolute) | 旋转 | 旋转 |
| **3D节点** | `liftCarriage` | `j1Pivot` | `j2Pivot` | `j4Pivot` |
| **运动操作** | `.position.z = d3` | `.rotation.z = J1°` | `.rotation.z = J2°` | `.rotation.z = J4°` |
| **范围** | d₃∈[150,450] | ±85° | ±130° | ±360° |
| **父节点** | baseGroup | liftCarriage | j1Pivot | j2Pivot |
| **Pivot位置** | (0, 0, d₃) | (209, 116, 141)* | +(199, 0, 0) | +(196, 0, 0) |
| **带动零件** | AXIS2/3/4/ROTATE | AXIS3/4/ROTATE | AXIS4/ROTATE | ROTATE |
| **滑块 ID** | s-j3 | s-j1 | s-j2 | s-j4 |

*J1 pivot 的 Z=141 是因为臂平面 Z≈441mm, liftCarriage 默认 Z=300mm, 差值为 141mm

#### 关节位置提取方法

从 STL 文件读取三角面顶点 → 计算每个零件 CAD 包围盒 → 找相邻零件的重叠区中心:

```
AXIS2 包围盒: X[123, 254], Y[66, 166]
AXIS3 包围盒: X[164, 443], Y[76, 155]
→ 重叠区: X[164, 254], Y[76, 155]
→ J1 = (164+254)/2, (76+155)/2 = (209, 116)

AXIS3 包围盒: X[164, 443]
AXIS4 包围盒: X[372, 635]
→ 重叠区: X[372, 443]
→ J2 = (372+443)/2, (76+155)/2 = (408, 116)

ROTATE 包围盒: X[595, 612], Y[107, 124]
→ J4 = (595+612)/2, (107+124)/2 = (604, 116)
```

### 三、移液臂虚拟关节定义

```
pipetteRoot (场景位置 x=1000, z=0)
└── pipettePivot ← rotation.x = -90° (同 Dobot 坐标系)
    │
    ├── pipetteBase                       [固定框架]
    │   └── mesh(group1)  pos=(0,0,0)
    │
    └── pipetteZ ← position.y = -(Z - 416.1)   ← Z 关节
        │   范围: Z ∈ [300, 500] mm
        │   动作: 横梁及其上所有部件沿框架 Z 向移动
        │
        ├── mesh(group2)  pos=(0,0,0)  横梁
        │
        └── pipetteX ← position.x = X - 300.3   ← X 关节
            │   范围: X ∈ [200, 420] mm
            │   动作: 中梁在横梁上 X 向滑动
            │
            ├── mesh(group3)  pos=(0,0,0)  中梁
            │
            ├── pipetteY1 ← position.z = Y1 - 123.1  ← Y1 关节
            │   │   范围: Y1 ∈ [0, 200] mm
            │   │   动作: ADP1 升降 (World Y方向)
            │   │
            │   ├── mesh(group4)  pos=(0,0,0)  ADP1
            │   └── mesh(tip)     pos=(0,0,0)  移液头
            │
            └── pipetteY2 ← position.z = Y2 - 123.1  ← Y2 关节
                范围: Y2 ∈ [0, 200] mm
                动作: ADP2 独立升降

关节参考值 = 各零件在 CAD 中的中心坐标:
  X_ref = group3.center.x  = 300.3
  Z_ref = group2.center.y  = 416.1
  Y1_ref = group4.center.z = 123.1
  Y2_ref = group5.center.z = 123.1
```

### 四、坐标系转换

```
CAD 坐标系 (STL 顶点)          Three.js 世界坐标系
       Z (上)                       Y (上)
       │                            │
       │    Rx(-90°)                │
       └──────►                     │
      /                             │
     /                              │
    X (前)                          X (前)
   /                               /
  /                               /
 Y (左)                          Z (左)

dobotPivot.rotation.x = -PI/2
pipettePivot.rotation.x = -PI/2

内部 (pivot 内 local 空间):  CAD 坐标, Z=上
  升价: position.z
  旋转: rotation.z
外部 (Three.js 世界空间):  Y=上
  升价→World Y, 旋转→绕 World Y
```

### 五、面板 ↔ 关节绑定

```
HTML 滑块                      JS 函数                   3D 节点操作
────────                      ────────                  ──────────
id="s-j1"  oninput→          onJointSlider()  →    j1Pivot.rotation.z = D2R(j1)
id="s-j2"  oninput→          onJointSlider()  →    j2Pivot.rotation.z = D2R(j2)
id="s-j3"  oninput→          onJointSlider()  →    liftCarriage.position.z = d3
id="s-j4"  oninput→          onJointSlider()  →    j4Pivot.rotation.z = D2R(j4)

id="sp-x"  oninput→          onPipetteJoint() →    pipetteX.position.x = X - 300.3
id="sp-z"  oninput→          onPipetteJoint() →    pipetteZ.position.y = -(Z - 416.1)
id="sp-y1" oninput→          onPipetteJoint() →    pipetteY1.position.z = Y1 - 123.1
id="sp-y2" oninput→          onPipetteJoint() →    pipetteY2.position.z = Y2 - 123.1
```

### 六、零件加载方式差异及原因

| | Dobot | 移液臂 |
|--|-------|--------|
| **几何体** | 居中 (`geometry.translate(-center)`) | 不居中 |
| **mesh.position** | 计算偏移 (CAD中心 - 父节点世界位置) | (0, 0, 0) |
| **原因** | 零件需精确对齐到关节 pivot | 零件顶点在 CAD 世界坐标，父节点控制运动 |

Dobot 需要居中+偏移因为每个零件挂载到不同的旋转节点，必须计算相对位置。
移液臂所有零件共享 CAD 世界坐标，父节点在原点，运动由父节点的 position 驱动。

---

## 备注

- Dobot M1Pro 是 4 轴 SCARA，STL 零件在装配体世界坐标中
- 移液臂 STL 原始文件 67MB，已用顶点聚类减面至 ~600KB 适配网页
- 减面脚本可重复运行: `python simplify_pipette_meshes.py`
- 移液臂运动学已配置 (X/Y/Z1/Z2 四轴棱柱)，可在面板上调节
- 坐标轴已统一: Z=垂直向上 (与Dobot一致), X/Y=水平面
- 平台模块位置通过 `platform_config.json` 持久化，刷新页面后恢复
- 编辑模式（勾选"编辑模式"）可修改模块参数和位置
- 三个页面均含调试坐标轴 (AxesHelper + X/Y/Z sprite 标签)

---

## Phase 4: 末端手指 + 托盘集成 (2026-06-01)

### 完成内容
1. **手指 STL 集成到 Dobot 末端** (`templates/models/finger/手指V1.0.STL`)
   - 8.5×29.6×22.0mm 末端夹爪
   - 通过 `geometry.rotateX(-PI/2)` 朝向下方
   - 挂在 `j4Pivot` 上跟随 J4 旋转
2. **样品托盘 STL 集成到平台** (`templates/models/plates/托盘V3.0(1).STL`)
   - 128×10×86mm 平放式样品托盘
   - CAD Y (10mm 厚度) 自然朝上
   - 位置 (300, 0.005, -300)，`rotation.y = -PI/2` 调整朝向
3. **新增 viewer 页面**
   - `templates/viewers/finger_viewer.html`
   - `templates/viewers/plate_viewer.html`

### 场景布局（最终状态）
| 项目 | 坐标 | 旋转 | 说明 |
|------|------|------|------|
| 底板 | 1600×1600 mm | 0° | 桌面 (X≥0) |
| 桌面模块 | 800×800 mm | 0° | 中心 (550, -250) |
| Dobot 机械臂 | (150, 0, -600) | 0° | SCARA, 左下 |
| 移液臂 | (950, 0, -250) | rotY=-90° | 龙门, 右上对角 |
| 手指 (Dobot 末端) | j4Pivot (-D4) | J4 联动 | 朝下 |
| 样品托盘 | (300, 0.005, -300) | rotY=-90° | 桌面附近 |

### 坐标系统一 (Phase 4 完整版)

| 轴 | 含义 | 调试标签 | Dobot | 移液臂 |
|----|------|---------|-------|--------|
| X | 水平 (World X) | 红色 | 臂展方向 | 中梁滑动 |
| Y | 水平 (World Z) | 绿色 | 横向 | 横梁移动 |
| Z | 垂直 (World Y=UP) | 蓝色 | J3 升降 | Z1/Z2 ADP升降 |

**统一规则**: 所有运动学引擎的 Z 轴 = 垂直向上 = J3 让 d3 变大的方向 = World Y。

---

## Phase 5: JSON 配置驱动 — Slider 限位动态同步 (2026-06-02)

### 目标

前端 slider 的 min/max/value 不再硬编码在 HTML 中，而是由 JSON 配置文件驱动：
- 修改 `pipette_kinematic_params.json` → 移液臂 slider 范围自动更新
- 修改 `dobot_joint_offsets.json` → Dobot slider 范围自动更新
- 运行时状态读写 `data/runtime/*.json`，刷新页面后恢复上次位置

### 数据流

```
data/offsets/pipette_kinematic_params.json
         │
         ▼
kinematics_pipette.py  (Python 模块加载时读取)
         │
         ├─→ X_MIN/X_MAX/Y_MIN/Y_MAX/Z_MIN/Z_MAX 等常量
         │
         └─→ /api/kinematic_params  (Flask)
                   │
                   ▼
         index.html (页面加载时 fetch)
                   │
                   ▼
         slider min/max/value 属性动态设置
```

```
data/offsets/dobot_joint_offsets.json
         │
         ▼
kinematics_M1Pro.py  (Python 模块加载时读取)
         │
         └─→ /api/kinematic_params  (Flask, 同上)
```

```
data/runtime/pipette_state.json   ← load_state() / save_state()
data/runtime/dobot_state.json     ← load_state() / save_state()
         ▲                    │
         └──── 页面刷新后自动恢复上次位置 ──┘
```

### 涉及文件变更

| 文件 | 变更 |
|------|------|
| `kinematics_pipette.py` | 模块加载时读 `pipette_kinematic_params.json`；新增 `load_state()` / `save_state()` |
| `kinematics_M1Pro.py` | 模块加载时读 `dobot_joint_offsets.json`；新增 `load_state()` / `save_state()` |
| `digital_twin.py` | 新增 `/api/kinematic_params` GET 端点 |
| `templates/index.html` | 页面加载时 `fetch('/api/kinematic_params')` 动态设置 slider 属性 |
| `data/runtime/pipette_state.json` | 移液臂运行时状态（轴值 + tip 坐标 + 时间戳） |
| `data/runtime/dobot_state.json` | Dobot 运行时状态（关节值 + TCP 坐标 + 时间戳） |

### API 端点

`GET /api/kinematic_params` 返回：

```json
{
  "dobot": {
    "j1":  { "min": -85.0,  "max": 85.0,   "unit": "deg" },
    "j2":  { "min": -130.0, "max": 130.0,  "unit": "deg" },
    "j3_deg": { "min": 5580.0, "max": -5220.0 },
    "j4":  { "min": -360.0, "max": 360.0,  "unit": "deg" },
    "d3_mm": { "min": 155.0,  "max": 455.0, "unit": "mm" }
  },
  "pipette": {
    "x":  { "min": 44.1, "max": 384.9, "ref": 300.3, "unit": "mm" },
    "y":  { "min": 59.1, "max": 367.1, "ref": 416.1, "ref_stl": 416.1, "unit": "mm" },
    "z1": { "min": 94.0, "max": 160.7, "ref": 123.1, "unit": "mm" },
    "z2": { "min": 94.0, "max": 160.7, "ref": 123.1, "unit": "mm" }
  }
}
```

### 运行时状态文件格式

**data/runtime/pipette_state.json**：
```json
{
  "axis": { "x": 310.0, "y": 250.0, "z1": 140.0, "z2": 135.0 },
  "tip1": { "x": 294.1, "y": 322.2, "z": 129.2 },
  "tip2": { "x": 324.3, "y": 322.2, "z": 124.2 },
  "updated_at": "2026-06-01T12:26:02"
}
```

**data/runtime/dobot_state.json**：
```json
{
  "joint": { "j1": 30.0, "j2": -30.0, "j3_deg": 1000.0, "j4": 45.0, "d3_mm": 327.778 },
  "tcp":   { "x": 366.506, "y": 125.0, "z": 332.778, "r": 45.0 },
  "updated_at": "2026-06-01T12:26:02"
}
```

### 前端初始化流程 (index.html)

```javascript
// 1. fetch /api/kinematic_params
// 2. 设置 Dobot 滑块: s-j1/min/max, s-j2/min/max, s-j3/min/max, s-j4/min/max
// 3. 设置 Pipette 滑块: sp-x/min/max/value=ref, sp-y/min/max/value=ref,
//                       sp-z1/min/max/value=ref, sp-z2/min/max/value=ref
// 4. updateRobot(...values) + updatePipette(...values)
// 5. fallback: 若 fetch 失败, 使用 HTML 中硬编码的默认值
```

### 运行时状态持久化 API

| 方法 | 路由 | 功能 |
|------|------|------|
| GET | `/api/runtime/dobot_state` | 读取 `data/runtime/dobot_state.json` |
| POST | `/api/runtime/dobot_state` | 写入 `data/runtime/dobot_state.json` |
| GET | `/api/runtime/pipette_state` | 读取 `data/runtime/pipette_state.json` |
| POST | `/api/runtime/pipette_state` | 写入 `data/runtime/pipette_state.json` |
