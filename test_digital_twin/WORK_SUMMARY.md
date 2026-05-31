# Dobot M1Pro STL 文件处理工作记录

## 完成时间
2026-05-31

## 工作概述

将用户提供的 STEP 装配体文件 (`Dobot M1Pro.stp`) 拆分为 5 个独立 STL 零件文件，并创建了独立的 3D 预览查看器。

---

## 文件结构

```
test_digital_twin/robot_arm_model/
├── digital_twin.py           # Flask 服务器（已配置静态路由）
├── verify_stl_workflow.py     # STL 验证脚本
├── templates/
│   ├── index.html             # 主 3D 场景
│   ├── Dobot_M1Pro.step       # 原始 CAD 文件 (7.36 MB)
│   ├── models/                # ← STL 文件目录
│   │   ├── M1_ROTATE_1.stl    # (41 KB, 820 triangles)
│   │   ├── M1_AXIS1_1.stl     # (2.84 MB, 56,786 triangles)
│   │   ├── M1_AXIS2_1.stl     # (675 KB, 13,506 triangles)
│   │   ├── M1_AXIS3_1.stl     # (1.36 MB, 27,884 triangles)
│   │   └── M1_AXIS4_1.stl     # (793 KB, 15,852 triangles)
│   └── viewers/               # ← HTML 预览文件目录
│       ├── M1_ROTATE_1_viewer.html
│       ├── M1_AXIS1_1_viewer.html
│       ├── M1_AXIS2_1_viewer.html
│       ├── M1_AXIS3_1_viewer.html
│       ├── M1_AXIS4_1_viewer.html
│       └── assembly_viewer.html    # ← 装配总览（可切换零件）
```

---

## 启动方式

```bash
cd D:/PycharmProjects/sdl_agent/test_digital_twin/robot_arm_model
python digital_twin.py
```

服务器启动后访问：
- `http://127.0.0.1:5001/` - 主数字孪生界面
- `http://127.0.0.1:5001/viewers/assembly_viewer.html` - 装配总览
- `http://127.0.0.1:5001/models/M1_AXIS1_1.stl` - 直接下载 STL

---

## 已实现的 Flask 路由

| 路由 | 功能 |
|------|------|
| `GET /` | 主页面 index.html |
| `GET /models/<filename>` | 服务 STL/STEP 等模型文件 |
| `GET /viewers/<filename>` | 服务 HTML 预览文件 |

---

## 零件几何数据

| 零件 | 三角面 | X 范围 | Y 范围 | Z 范围 | 说明 |
|------|--------|--------|--------|--------|------|
| M1_ROTATE_1 | 820 | 613.5mm | 125.8mm | 354.2mm | 旋转关节 |
| M1_AXIS1_1 | 56,786 | 157.3mm | 215.2mm | 682.9mm | 第一轴 |
| M1_AXIS2_1 | 13,506 | 255.3mm | 167.3mm | 544.2mm | 第二轴 |
| M1_AXIS3_1 | 27,884 | 444.8mm | 156.8mm | 490.2mm | 第三轴 |
| M1_AXIS4_1 | 15,852 | 636.3mm | 148.3mm | 467.2mm | 第四轴 |

**总计：115,848 triangles**

---

## 下一步工作

1. **集成到主场景**：将 5 个 STL 零件集成到 `index.html` 的 Three.js 场景中，替换现有的简化机械臂模型

2. **位置标定**：根据实际装配关系，为每个零件设置正确的相对位置

3. **运动学绑定**：将 Three.js 中的模型关节角度与 `kinematics.py` 的运动学计算绑定

---

## 相关技术

- **Three.js** r160：`STLLoader` 加载 STL 文件
- **Flask**：`send_from_directory` 服务静态文件
- **STL 格式**：Binary STL（非 ASCII）

---

## 验证命令

```bash
# 测试所有路由
python -c "
from digital_twin import app
with app.test_client() as c:
    for name in ['M1_ROTATE_1', 'M1_AXIS1_1', 'M1_AXIS2_1', 'M1_AXIS3_1', 'M1_AXIS4_1']:
        r = c.get(f'/models/{name}.stl')
        print(f'STL {name}: {r.status_code}')
    r = c.get('/viewers/assembly_viewer.html')
    print(f'Assembly viewer: {r.status_code}')
"
```

---

## 备注

- `Dobot M1Pro.stp` 是 5 轴 SCARA 机械臂的完整 CAD 装配体
- 已拆分为 5 个独立零件：ROTATE、AXIS1、AXIS2、AXIS3、AXIS4
- 每个零件都有独立的 HTML 预览页面
- `assembly_viewer.html` 支持切换显示/隐藏单个零件