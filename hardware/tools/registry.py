"""
硬件工具注册系统 - 装饰器模式 + 自动发现

两种注册方式：
1. 装饰器注册：@register_tool(name="...", description="...", params={...})
2. 自动发现：discover_tools() 扫描 hardware/tools/ 目录，自动导入所有模块

使用示例:
    from hardware import ToolRegistry, register_tool, discover_tools

    # 手动注册
    def my_func(x: int) -> str:
        return "done"
    ToolRegistry.register("my_tool", "描述", {"x": {"type": "int", "required": True}}, my_func)

    # 自动发现（扫描 tools/ 目录）
    discover_tools()
"""

from typing import Dict, Any, Callable, List
import json
import importlib
import pkgutil
from pathlib import Path


class ToolRegistry:
    """
    工具注册表 - 单例模式

    使用装饰器 @register_tool 自动注册工具函数，或通过 discover_tools() 自动扫描。
    """
    _instance = None
    _tools: Dict[str, Dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name: str, description: str, params: Dict[str, Any], func: Callable):
        """
        注册工具函数（静默，日志由 discover_tools() 统一输出）

        Args:
            name: 工具名称（唯一标识）
            description: 工具描述
            params: 参数定义字典 {"param_name": {"type": "int", "description": "...", "required": True, "default": 0}}
            func: 工具函数
        """
        cls._tools[name] = {
            "name": name,
            "description": description,
            "params": params,
            "function": func,
        }

    @classmethod
    def unregister(cls, name: str) -> bool:
        """取消注册指定工具，返回是否成功"""
        if name in cls._tools:
            del cls._tools[name]
            return True
        return False

    @classmethod
    def get_all(cls) -> Dict[str, Dict[str, Any]]:
        """获取所有已注册的工具（返回副本）"""
        return cls._tools.copy()

    @classmethod
    def get_tool(cls, name: str) -> Dict[str, Any]:
        """获取指定工具，未找到返回 None"""
        return cls._tools.get(name)

    @classmethod
    def list_names(cls) -> List[str]:
        """列出所有已注册工具名称"""
        return sorted(cls._tools.keys())

    @classmethod
    def count(cls) -> int:
        """已注册工具数量"""
        return len(cls._tools)

    @classmethod
    def clear(cls):
        """清空所有注册（主要用于测试）"""
        cls._tools.clear()

    @classmethod
    def export_to_json(cls, filepath: str):
        """导出注册表到 JSON 文件（不含函数引用）"""
        export_data = {}
        for name, info in cls._tools.items():
            export_data[name] = {
                "name": info["name"],
                "description": info["description"],
                "params": info["params"],
            }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)


def register_tool(name: str, description: str, params: Dict[str, Any]):
    """
    工具注册装饰器

    用法:
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


def discover_tools(
    packages: List[str] = None,
    base_path: str = None,
    skip_modules: List[str] = None,
) -> int:
    """
    自动发现并注册 hardware/tools/ 目录下的工具模块

    扫描指定目录，导入所有 .py 文件。已加载的模块会 reload 以重新触发
    @register_tool 装饰器，新模块直接 import。

    类似 software/software_controller.py 的 _discover_algorithms()。

    Args:
        packages: 要扫描的子目录名列表，默认 ['']（tools/ 根目录）
        base_path: 扫描根目录，默认为当前文件所在目录（hardware/tools/）
        skip_modules: 跳过的模块名列表，默认 ['registry']

    Returns:
        int: 已加载的模块数量

    Example:
        from hardware import discover_tools, ToolRegistry

        discover_tools()
        print(ToolRegistry.list_names())
    """
    import sys

    if skip_modules is None:
        skip_modules = ['registry']

    if base_path is None:
        base_path = str(Path(__file__).parent)

    if packages is None:
        packages = ['']

    loaded_count = 0

    for pkg_name in packages:
        pkg_dir = Path(base_path) / pkg_name
        if not pkg_dir.is_dir():
            continue

        for finder, module_name, is_pkg in pkgutil.iter_modules([str(pkg_dir)]):
            if module_name in skip_modules or module_name.startswith('_'):
                continue

            full_module_path = (
                f"hardware.tools.{pkg_name}.{module_name}"
                if pkg_name else
                f"hardware.tools.{module_name}"
            )
            # 记录注册前已有的工具名，用于发现新注册
            before = set(ToolRegistry._tools.keys())

            try:
                if full_module_path in sys.modules:
                    importlib.reload(sys.modules[full_module_path])
                else:
                    importlib.import_module(full_module_path)
                loaded_count += 1
            except Exception as e:
                import traceback
                print(f"[ToolRegistry] [WARN] Failed to load module {full_module_path}: {e}")
                traceback.print_exc()
                continue

            # 输出新注册的工具
            after = set(ToolRegistry._tools.keys())
            new_names = after - before
            for name in new_names:
                print(f"[ToolRegistry] [OK] Registered tool: {name} ({full_module_path})")

    return loaded_count


def reload_tools(base_path: str = None) -> int:
    """
    清空注册表并重新扫描所有工具（热加载新文件 / 更新已有工具）

    先清空所有已注册工具，然后重新扫描每个模块（已加载的 reload，新的 import），
    确保 @register_tool 装饰器全部重新执行。

    Args:
        base_path: 扫描根目录

    Returns:
        int: 重新注册的工具数量
    """
    ToolRegistry.clear()
    return discover_tools(base_path=base_path)


if __name__ == "__main__":
    print("=== Scanning tools/ directory ===")
    discover_tools()
    print(f"\nTotal registered: {ToolRegistry.count()} tools")
    print("Names:", ToolRegistry.list_names())