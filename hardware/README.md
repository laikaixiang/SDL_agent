# 硬件模块使用指南

本目录包含自动化实验平台的硬件控制模块，支持通过自然语言或直接调用的方式控制硬件设备。

## 📁 目录结构

```
hardware/
├── README.md           # 本文档
├── tools.py            # 硬件工具函数（核心）
├── agent_client.py     # MQTT 通信客户端
├── spec_client.py      # 光谱仪数据采集客户端
├── visualization.py    # 数据可视化工具
└── __init__.py         # 模块初始化
```

## 🚀 快速开始

### 1. 调用方式

硬件工具有两种调用方式：

#### 方式一：通过 Web 界面（推荐）

1. 打开 Web 界面，点击左下角 `+` 号
2. 选择 "硬件操控模式"
3. 选择子模式：
   - **单步控制**：手动选择工具并填写参数
   - **实验设计**：用自然语言描述需求，AI 自动规划实验流程

#### 方式二：通过代码直接调用

```python
from hardware.tools import execute_spin_coating, execute_start_experiment

# 注册一步旋涂实验
result = execute_spin_coating(
    spin_speed=3000,  # 转速 (rpm)
    spin_acc=1000,    # 加速度 (rpm/s)
    spin_dur=30000,   # 持续时间 (ms)
    reagent="Perovskite",  # 试剂名称
    volume=10         # 体积 (µl)
)
print(result)

# 启动实验序列
result = execute_start_experiment()
print(result)
```

## 🔧 如何添加新硬件工具

### 步骤 1：在 `hardware/tools.py` 中添加底层函数

在文件末尾添加新的执行函数：

```python
def execute_your_hardware(param1: int, param2: str) -> str:
    """
    底层同步函数：你的硬件功能描述
    
    Args:
        param1 : 参数1说明
        param2 : 参数2说明
    
    Returns:
        str: 执行结果消息
    """
    try:
        # 实现你的硬件控制逻辑
        # 例如：通过 MQTT 发送指令
        client = get_mqtt_client()
        payload = {
            "action": "your_action",
            "params": {"param1": param1, "param2": param2}
        }
        client.publish("your_topic", json.dumps(payload))
        return f"硬件操作成功: {param1}, {param2}"
    except Exception as e:
        return f"硬件操作失败: {str(e)}"
```

### 步骤 2：在 `core/hardware_controller.py` 中注册工具

#### 2.1 在 `_load_hardware_tools()` 方法中添加工具定义

找到 `HardwareAgent` 类的 `_load_hardware_tools()` 方法，添加：

```python
HardwareTool(
    name="your_hardware",
    description="你的硬件功能描述（AI 会根据这个描述决定何时调用）",
    params={
        "param1": {
            "type": "int",
            "description": "参数1的说明",
            "required": True,
            "default": None
        },
        "param2": {
            "type": "str",
            "description": "参数2的说明",
            "required": False,
            "default": "default_value"
        },
    },
    function="execute_your_hardware",
),
```

#### 2.2 在 `execute_tool_call()` 方法中添加分派逻辑

找到 `execute_tool_call()` 方法，在导入部分添加：

```python
from hardware.tools import (
    execute_spin_coating,
    execute_set_temperature,
    execute_move_robot_arm,
    execute_start_experiment,
    execute_collect_spectrum,
    execute_your_hardware,  # 添加这一行
)
```

然后在 `if-elif` 链中添加：

```python
elif tool_name == "your_hardware":
    result = execute_your_hardware(
        int(params.get("param1")),
        str(params.get("param2", "default_value")),
    )
```

### 步骤 3：测试新工具

1. 重启 Flask 应用
2. 打开 Web 界面 -> 硬件操控 -> 单步控制
3. 在列表中找到你的新工具
4. 填写参数并点击 ▶ 执行

## 🔄 如何修改现有硬件工具

### 修改参数

在 `core/hardware_controller.py` 的 `_load_hardware_tools()` 中找到对应工具，修改 `params` 字典：

```python
HardwareTool(
    name="set_temperature",
    description="设置设备温度",
    params={
        "target": {
            "type": "float",
            "description": "目标温度值（℃）",
            "required": True,
            "default": None
        },
        # 添加新参数
        "rate": {
            "type": "float",
            "description": "升温速率（℃/min）",
            "required": False,
            "default": 5.0
        },
    },
    function="execute_set_temperature",
),
```

然后在 `hardware/tools.py` 中修改对应的执行函数签名和实现。

### 修改功能描述

修改 `description` 字段会影响 AI 对工具的理解和选择：

```python
HardwareTool(
    name="set_temperature",
    description="设置加热台温度并控制升温速率",  # 更详细的描述
    ...
)
```

## 📡 MQTT 通信协议

### 连接配置

MQTT 客户端配置在 `hardware/agent_client.py` 中：

```python
class MQTTConnector:
    def __init__(self):
        self.broker = "localhost"  # MQTT 服务器地址
        self.port = 1883           # MQTT 端口
        self.client_id = f"python_client_{uuid.uuid4()}"
```

### 消息格式

#### 旋涂实验注册

- **主题**: `do_experiment`
- **格式**: `p{转速},{加速度},{时长},{试剂位置},{体积}`
- **示例**: `p3000,1000,30000,BP01,10`

#### 实验序列启动

- **主题**: `do_experiment`
- **格式**: `pstart`

#### 自定义消息

```python
client = get_mqtt_client()
payload = {
    "action": "your_action",
    "params": {"key": "value"}
}
client.publish("your_topic", json.dumps(payload))
```

## 🧪 试剂配置

试剂位置配置在项目根目录的 `reagent_layout.json` 文件中：

```json
{
    "Points": {
        "BP01": {
            "name": "Perovskite",
            "x": 100,
            "y": 200
        },
        "BP02": {
            "name": "DMF",
            "x": 150,
            "y": 250
        }
    }
}
```

### 添加新试剂

1. 编辑 `reagent_layout.json`
2. 添加新的位置点：

```json
"BP03": {
    "name": "YourReagent",
    "x": 200,
    "y": 300
}
```

3. 保存文件，无需重启应用

## 🐛 调试技巧

### 查看硬件调用日志

硬件函数调用会自动输出到控制台和日志文件：

```
[硬件调用] 工具: set_temperature, 参数: {'target': 150.0}
[硬件调用] 工具 set_temperature 执行完成
```

### 检查 MQTT 连接状态

```python
from hardware.tools import get_mqtt_client

client = get_mqtt_client()
if client.is_connected:
    print("MQTT 已连接")
else:
    print("MQTT 未连接，尝试重连...")
    client.connect(timeout=5)
```

### 测试试剂查找

```python
from hardware.tools import find_reagent

position = find_reagent("Perovskite")
print(f"试剂位置: {position}")  # 输出: BP01
```

## 📝 参数类型说明

| 类型    | Python 类型 | 说明                     | 示例          |
|---------|-------------|--------------------------|---------------|
| `int`   | `int`       | 整数                     | `3000`        |
| `float` | `float`     | 浮点数                   | `150.5`       |
| `str`   | `str`       | 字符串                   | `"Perovskite"`|
| `bool`  | `bool`      | 布尔值                   | `True`        |

## ⚠️ 注意事项

1. **参数验证**：所有必填参数必须提供，否则会返回错误
2. **转速限制**：旋涂转速不能超过 6000 rpm
3. **试剂名称**：必须与 `reagent_layout.json` 中的名称完全一致（区分大小写）
4. **MQTT 连接**：首次调用会自动尝试连接，连接失败会返回错误信息
5. **线程安全**：硬件控制器内置防重复提交保护，2秒内相同指令会被拦截

## 🔗 相关文件

- `core/hardware_controller.py` - 硬件控制器和工具注册
- `hardware/tools.py` - 硬件工具函数实现
- `hardware/agent_client.py` - MQTT 通信客户端
- `reagent_layout.json` - 试剂位置配置
- `templates/index.html` - Web 界面（单步控制面板）

## 📞 技术支持

如有问题，请查看：
1. 控制台日志输出
2. `logs/` 目录下的日志文件
3. MQTT 服务器连接状态

---

**最后更新**: 2026-04-15
