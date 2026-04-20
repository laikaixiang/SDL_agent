"""
硬件工具注册系统 - 装饰器模式
"""

from typing import Dict, Any, Callable
import json


class ToolRegistry:
    """
    工具注册表 - 单例模式

    使用装饰器 @register_tool 自动注册工具函数
    """
    _instance = None
    _tools: Dict[str, Dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name: str, description: str, params: Dict[str, Any], func: Callable):
        """注册工具函数"""
        cls._tools[name] = {
            "name": name,
            "description": description,
            "params": params,
            "function": func
        }

    @classmethod
    def get_all(cls) -> Dict[str, Dict[str, Any]]:
        """获取所有已注册的工具"""
        return cls._tools.copy()

    @classmethod
    def get_tool(cls, name: str) -> Dict[str, Any]:
        """获取指定工具"""
        return cls._tools.get(name)

    @classmethod
    def export_to_json(cls, filepath: str):
        """导出注册表到JSON文件"""
        export_data = {}
        for name, info in cls._tools.items():
            export_data[name] = {
                "name": info["name"],
                "description": info["description"],
                "params": info["params"]
            }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)


def register_tool(name: str, description: str, params: Dict[str, Any]):
    """
    工具注册装饰器

    Args:
        name: 工具名称
        description: 工具描述
        params: 参数定义字典

    Example:
        @register_tool(
            name="spin_coating",
            description="执行旋涂实验",
            params={
                "spin_speed": {"type": "int", "description": "转速(rpm)", "required": True}
            }
        )
        def execute_spin_coating(spin_speed: int) -> str:
            return "success"
    """
    def decorator(func: Callable):
        ToolRegistry.register(name, description, params, func)
        return func
    return decorator
