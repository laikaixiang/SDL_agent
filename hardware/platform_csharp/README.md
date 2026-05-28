# C# 自动化实验平台

基于 WinForms 的实验室自动化控制平台，通过 MQTT 接收 Python AI Agent 发送的实验指令，协调控制机械臂、旋涂机、点胶机和光谱仪等硬件设备执行实验。

## 系统架构

```
┌─────────────────┐         MQTT          ┌──────────────────────┐
│   Python AI     │ ────────────────────── │   C# WinForms        │
│   Agent         │  (EMQX 192.168.120.129)│   Platform           │
└─────────────────┘                        └──────────┬───────────┘
                                                      │
            ┌──────────────────────────────────────────┼──────────────────┐
            │                                          │                  │
     ┌──────▼──────┐    TCP    ┌──────────┐   Modbus RTU   ┌────────────┐
     │  Dobot 机械臂 │ ◄─────── │ ArmForm  │ ◄────────────── │ Spin Coater │
     │  (运动/夹爪) │           │          │                │ (电机控制)  │
     └─────────────┘           └──────────┘                └────────────┘
                                                              │
     ┌──────────────┐    Serial    ┌────────────┐    TCP      ┌────────────┐
     │  DK 点胶机    │ ◄─────────── │ Dispenser  │ ◄─────────── │ Spectrometer │
     │  (移液/滴液)  │              │ Form       │              │ (光谱仪)    │
     └──────────────┘              └────────────┘              └────────────┘
```

## 硬件控制

| 硬件 | 控制方式 | 协议端口 | 说明 |
|------|----------|----------|------|
| Dobot 机械臂 | TCP (Dashboard/Move/Feedback) | 29999/30003/30004 | 三端口分别用于状态、控制、反馈 |
| 旋涂机电机 | Modbus RTU | RS485 串口 (9600-115200) | 通过 NModbus4 控制速度和位置 |
| DK 点胶机 | 串口 RS232 | 115200 波特率 | 控制 X/Y/Z 轴运动和左右注射泵 |
| 光谱仪 | TCP Socket | 127.0.0.1:1701 | 读取波长和计数数据 |
| 供料轨道 | 串口 | 可配置 | 控制进料方向和状态 |

## MQTT 通信

- **Broker**: `192.168.120.129:1883`
- **认证**: username=`platform`, password=`s208ht`
- **主要 Topic**:
  - `do_experiment` - 接收 AI Agent 实验参数 (格式: `spin_speed,spin_acc,spin_dur,reagent_pos,volume`)
  - `control` - 发送给 Python 端的控制命令 (continue/stop/record/quit)
  - `wavelength` / `counts` - 光谱数据
  - `time` - 时间戳

## 项目结构

```
platform_csharp/
├── MainForm.cs / .Designer.cs / .resx    # 主控制面板，集成 AI Agent
├── ArmForm.cs / .Designer.cs / .resx    # Dobot 机械臂控制
├── CoaterForm.cs / .Designer.cs / .resx  # 旋涂机/电机控制
├── DispenserForm.cs / .Designer.cs / .resx # DK 点胶机控制
├── SpecForm.cs / .Designer.cs / .resx    # 光谱仪 + MQTT 通信
├── Program.cs
├── Winform_platform.csproj
├── Winform_platform.sln
├── Properties/
│   └── Resources.Designer.cs
├── Auto/
│   ├── Agent.cs         # 实验步骤缓存 & MQTT 消息队列
│   ├── Data.cs           # 光谱数据读取
│   ├── FeedRail.cs       # 供料轨道控制
│   ├── Mqtt_connection.cs # MQTT 连接管理
│   └── ReadExcel.cs      # Excel 参数读取 (调用 ReadExcelDLL.dll)
├── DK/
│   ├── Axes.cs          # DK 机械轴控制
│   ├── DKPoint.cs       # 点位定义
│   ├── Pipette.cs       # 注射泵控制
│   └── DK(for reference).h # API 参考文档
└── com.dobot.api/
    ├── Dashboard.cs      # 机械臂状态/使能
    ├── DobotClient.cs    # TCP 客户端基类
    ├── DobotMove.cs      # 运动指令 (MovJ/MovL/Grip/Release)
    ├── Feedback.cs       # 实时反馈数据
    ├── JointPoint.cs     # 关节坐标
    ├── DescartesPoint.cs # 笛卡尔坐标
    └── ErrorInfoHelper.cs # 报警信息解析
```

## 依赖

- .NET 8.0 SDK (Windows)
- M2Mqtt 4.3.0
- Newtonsoft.Json 13.0.3
- NModbus4 2.1.0
- DocumentFormat.OpenXml 3.3.0
- System.IO.Ports 9.0.2

## 编译运行

```bash
dotnet build Winform_platform.sln
dotnet run --project Winform_platform
```

## 实验流程

1. AI Agent 通过 MQTT 发送 `pstart` 指令触发实验
2. 平台按顺序接收实验参数到 step_buffer
3. 机械臂从基片盘抓取基片放置到旋涂机
4. 点胶机从试剂瓶吸取液体后滴涂到基片
5. 旋涂机按指定参数旋转基片
6. 光谱仪原位采集旋涂过程中的光谱数据
7. 机械臂将基片放回并更换新基片重复实验

贡献者：https://github.com/Raymondm0
