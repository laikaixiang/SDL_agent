# C# 自动化平台代码

此目录存放 AutonomousPlatform 的 C# WinForms 自动化平台代码。
该平台通过 MQTT 接收 Python 端发送的实验指令，控制机械臂、旋涂机、点胶机等硬件。

## 复制说明

请从 `AutonomousPlatform/` 目录手动复制以下文件和文件夹到本目录：

```
platform_csharp/
├── MainForm.cs / .Designer.cs / .resx
├── ArmForm.cs / .Designer.cs / .resx
├── CoaterForm.cs / .Designer.cs / .resx
├── DispenserForm.cs / .Designer.cs / .resx
├── SpecForm.cs / .Designer.cs / .resx
├── Program.cs
├── Winform_platform.csproj
├── Winform_platform.sln
├── Properties/
│   └── Resources.Designer.cs
├── Auto/
│   ├── Agent.cs
│   ├── Data.cs
│   ├── FeedRail.cs
│   ├── Mqtt_connection.cs
│   └── ReadExcel.cs
├── DK/
│   ├── Axes.cs
│   ├── DK(for reference).h
│   ├── DKPoint.cs
│   └── Pipette.cs
└── com.dobot.api/
    ├── Dashboard.cs
    ├── DescartesPoint.cs
    ├── DobotClient.cs
    ├── DobotMove.cs
    ├── ErrorInfoBean.cs
    ├── ErrorInfoHelper.cs
    ├── Feedback.cs
    ├── FeedbackData.cs
    ├── JointPoint.cs
    └── OffsetPosition.cs
```

## 注意事项

- 这些文件仅供参考和独立编译，Python 端不会直接调用它们
- Python 与 C# 平台之间通过 MQTT 消息通信（EMQX 服务器 192.168.120.129:1883）
- 编译运行需要 .NET 8.0 SDK 和 Windows 环境

贡献者：https://github.com/Raymondm0
