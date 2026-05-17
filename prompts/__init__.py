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
    lang: str = 'zh',
) -> PromptManager:
    """获取全局 PromptManager（懒加载，按语言缓存）

    Args:
        registry_path: registry.yaml 文件路径
        overrides_dir: overrides 覆盖文件目录路径
        lang: 语言代码，默认 'zh'，可选 'en'
    """
    global _manager
    # Recreate if language changed (different lang = different prompt files)
    if _manager is None:
        _manager = PromptManager(registry_path, overrides_dir, lang=lang)
    elif getattr(_manager, '_lang', None) != lang:
        _manager = PromptManager(registry_path, overrides_dir, lang=lang)
    return _manager


def reset_manager():
    """重置单例（测试用）"""
    global _manager
    _manager = None


__all__ = ["PromptManager", "MissingVariableError", "create_prompt_manager", "reset_manager"]
