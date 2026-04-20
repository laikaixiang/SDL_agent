# hardware/tools — 硬件工具注册系统

## 目录结构

```
hardware/tools/
├── REGISTRY.json          # 工具元数据（供 LLM/ExperimentDesignAgent 读取）
├── registry.py            # ToolRegistry 类 + @register_tool 装饰器
├── __init__.py            # 运行时导入和 __all__
├── spin_coating.py        # 旋涂工具
├── temperature.py         # 温度控制工具
├── robot_arm.py           # 机械臂工具
├── experiment_control.py  # 实验启动工具
└── spectrum.py            # 光谱采集工具
```

## REGISTRY.json 在哪里被使用

| 使用方 | 文件 | 用途 |
|--------|------|------|
| ExperimentDesignAgent | `core/field_inference.py` | 构建 LLM 系统提示，让模型知道有哪些硬件工具可用 |
| HardwareController | `core/hardware_controller.py` | 工具发现与 `execute_tool_call()` 分发 |

`@register_tool` 装饰器在运行时将工具注册到 `ToolRegistry`（内存单例），供 Python 代码直接调用。两套机制独立但内容保持同步。

## 添加新工具（推荐方式）

使用 `platform_init/update_registry.py` 自动同步三处：

```python
from platform_init.update_registry import add_tool

add_tool(
    name="your_tool",
    description="工具描述（供 LLM 理解）",
    params={
        "param1": {"type": "int", "description": "参数说明", "required": True},
        "param2": {"type": "float", "description": "参数说明", "required": False, "default": 1.0},
    }
)
```

这会自动：
1. 更新 `REGISTRY.json`
2. 生成 `hardware/tools/your_tool.py`（含 `@register_tool` 装饰器和函数骨架）
3. 在 `__init__.py` 中添加 import 和 `__all__` 条目

然后在生成的 `your_tool.py` 中实现具体的 MQTT 逻辑，并在 `core/hardware_controller.py:execute_tool_call()` 中添加分发 case。

## 手动添加新工具(不推荐)

如果不使用脚本，需手动同步以下三处：

**1. 新建 `hardware/tools/your_tool.py`：**

```python
from .registry import register_tool
from ..mqtt import get_mqtt_client

@register_tool(
    name="your_tool",
    description="工具描述",
    params={
        "param1": {"type": "int", "description": "...", "required": True}
    }
)
def execute_your_tool(param1: int) -> str:
    client = get_mqtt_client()
    client.publish("topic", f"payload{param1}")
    return "result"
```

**2. 在 `REGISTRY.json` 中添加条目：**

```json
"your_tool": {
  "name": "your_tool",
  "description": "工具描述",
  "params": {
    "param1": {"type": "int", "description": "...", "required": true}
  }
}
```

**3. 在 `__init__.py` 中添加导入：**

```python
from .your_tool import execute_your_tool
# 并在 __all__ 中添加 'execute_your_tool'
```

**4. 在 `core/hardware_controller.py:execute_tool_call()` 中添加分发 case。**
