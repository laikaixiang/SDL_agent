"""
Prompt 集中管理模块

提供 PromptManager（加载/渲染/修改）、PromptOptimizer（LLM 优化辅助）、
Flask Blueprint（API 接口）。

使用方式:
    from prompts import create_prompt_manager
    pm = create_prompt_manager()
    text = pm.get("extraction_system_vision", task_description="...", fields="...")
"""

from .manager import PromptManager, MissingVariableError

_manager: PromptManager | None = None


def create_prompt_manager(
    registry_path: str = "prompts/registry.yaml",
    overrides_dir: str = "prompts/overrides",
) -> PromptManager:
    """获取全局 PromptManager 单例（懒加载）"""
    global _manager
    if _manager is None:
        _manager = PromptManager(registry_path, overrides_dir)
    return _manager


def reset_manager():
    """重置单例（测试用）"""
    global _manager
    _manager = None


__all__ = ["PromptManager", "MissingVariableError", "create_prompt_manager", "reset_manager"]
