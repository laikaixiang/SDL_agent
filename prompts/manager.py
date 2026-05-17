"""
PromptManager — Prompt 集中管理的核心类

职责:
- 从 registry.yaml 加载所有 prompt 的元信息
- 按索引加载各 YAML 文件中的 prompt 定义
- 合并 overrides/ 中的运行时修改
- 使用 string.Template 渲染 prompt，校验变量完整性
- 支持热更新（写入 overrides/，立即生效，不重启）

使用方式:
    pm = PromptManager("prompts/registry.yaml", "prompts/overrides")
    text = pm.get("extraction_system_vision", task_description="...", fields="...")
    pm.update("extraction_system_vision", template="新的模板...")
"""

import os
import yaml
from string import Template
from typing import Dict, List, Optional, Any


class MissingVariableError(Exception):
    """调用 get() 时缺少必需的变量"""

    def __init__(self, prompt_name: str, missing: List[str]):
        self.prompt_name = prompt_name
        self.missing = missing
        super().__init__(
            f"Prompt '{prompt_name}' 缺少变量: {', '.join(missing)}"
        )


class NoSuchPromptError(Exception):
    """引用了不存在的 prompt"""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Prompt '{name}' 不存在")


class PromptManager:
    """Prompt 集中管理器"""

    def __init__(self, registry_path: str, overrides_dir: str, lang: str = 'zh'):
        """
        Args:
            registry_path: registry.yaml 文件路径
            overrides_dir: overrides 覆盖文件目录路径
            lang: 语言代码，默认 'zh'，可选 'en'
        """
        self._registry_path = registry_path
        self._overrides_dir = overrides_dir
        self._lang = lang
        self._registry: Dict[str, dict] = {}      # name → {file, category, enabled}
        self._prompts: Dict[str, dict] = {}        # name → {name, description, variables, template}
        self._templates: Dict[str, Template] = {}  # name → compiled Template
        self._load_all()

    # ═══════════════════════════════════════════════════════════════
    # 公开 API
    # ═══════════════════════════════════════════════════════════════

    def get(self, key: str, **variables) -> str:
        """获取渲染后的 prompt 文本

        校验所有声明变量是否已提供，缺变量抛出 MissingVariableError。

        Args:
            key: prompt 名称（registry.yaml 中的 key）
            **variables: 模板变量键值对

        Returns:
            渲染后的 prompt 文本
        """
        if key not in self._prompts:
            raise NoSuchPromptError(key)

        declared = set(self._prompts[key]["variables"])
        provided = set(variables.keys())
        missing = declared - provided
        if missing:
            raise MissingVariableError(key, sorted(missing))

        template = self._templates[key]
        return template.substitute(**variables)

    def list_all(self, category: str = None) -> list[dict]:
        """列出所有 prompt 元信息（不含模板内容）

        Args:
            category: 可选，按分类过滤

        Returns:
            [{name, category, description, variables, overridden, source_file}, ...]
        """
        result = []
        for name, meta in self._registry.items():
            if not meta.get("enabled", True):
                continue
            if category and meta.get("category") != category:
                continue
            prompt = self._prompts.get(name, {})
            overridden = self._override_file(name) is not None
            result.append({
                "name": name,
                "category": meta["category"],
                "description": prompt.get("description", ""),
                "variables": prompt.get("variables", []),
                "overridden": overridden,
                "source_file": meta["file"],
            })
        return result

    def get_meta(self, name: str) -> dict:
        """获取单个 prompt 完整信息

        Returns:
            {name, category, description, variables, overridden,
             current_template, original_template, source_file, override_file}
        """
        if name not in self._registry:
            raise NoSuchPromptError(name)

        meta = self._registry[name]
        prompt = self._prompts[name]
        override_path = self._override_file(name)
        original = self._load_original_template(name)

        return {
            "name": name,
            "category": meta["category"],
            "description": prompt.get("description", ""),
            "variables": prompt.get("variables", []),
            "overridden": override_path is not None,
            "current_template": prompt.get("template", ""),
            "original_template": original,
            "source_file": meta["file"],
            "override_file": os.path.relpath(override_path, self._overrides_dir)
                if override_path else None,
        }

    def update(
        self,
        name: str,
        template: Optional[str] = None,
        variables: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> None:
        """运行时修改 prompt，写入 overrides/ 目录，立即生效

        只传要改的字段，未传字段保持不变。
        不修改源文件。

        Args:
            name: prompt 名称
            template: 新模板文本（可选）
            variables: 新变量列表（可选）
            description: 新描述（可选）
        """
        if name not in self._registry:
            raise NoSuchPromptError(name)

        # 构建 override 数据（只存要改的字段）
        override = {}
        if template is not None:
            override["template"] = template
        if variables is not None:
            override["variables"] = variables
        if description is not None:
            override["description"] = description

        if not override:
            return  # nothing to update

        # 确保 overrides 子目录存在
        meta = self._registry[name]
        file_rel = meta["file"]  # e.g. "extraction/_system_vision.yaml"
        override_path = os.path.join(self._overrides_dir, file_rel)
        os.makedirs(os.path.dirname(override_path), exist_ok=True)

        with open(override_path, "w", encoding="utf-8") as f:
            yaml.dump(override, f, allow_unicode=True, default_flow_style=False)

        # 立即更新内存
        merged = self._merge(name, override)
        self._prompts[name] = merged
        self._templates[name] = Template(merged["template"])

    def reset(self, name: str) -> None:
        """删除指定 prompt 的 override 文件，回到原始版本"""
        if name not in self._registry:
            raise NoSuchPromptError(name)

        override_path = self._override_file(name)
        if override_path:
            os.remove(override_path)
            # 清理空目录
            parent = os.path.dirname(override_path)
            if os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)

        # 重新从源文件加载
        prompt = self._load_prompt_file(name)
        self._prompts[name] = prompt
        self._templates[name] = Template(prompt["template"])

    def reload(self) -> None:
        """重新加载所有 prompt（清除所有 overrides 的影响）"""
        self._load_all()

    # ═══════════════════════════════════════════════════════════════
    # 内部方法
    # ═══════════════════════════════════════════════════════════════

    def _resolve_path(self, file_path: str) -> str:
        """根据当前 lang 设置替换语言前缀"""
        parts = file_path.split('/', 1)
        if parts[0] in ('zh', 'en') and len(parts) == 2:
            return os.path.join(
                os.path.dirname(self._registry_path),
                self._lang, parts[1]
            )
        return os.path.join(os.path.dirname(self._registry_path), file_path)

    def _load_all(self) -> None:
        """完整加载流程: registry → 源文件 → overrides 合并 → 编译模板"""
        self._registry = {}
        self._prompts = {}
        self._templates = {}

        # 1. 加载 registry.yaml
        with open(self._registry_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data or "prompts" not in data:
            raise ValueError(f"registry.yaml 格式错误：缺少 'prompts' 字段")

        # 2. 逐个加载 prompt 文件
        for name, meta in data["prompts"].items():
            if not meta.get("enabled", True):
                continue
            self._registry[name] = meta
            prompt = self._load_prompt_file(name)
            self._prompts[name] = prompt
            self._templates[name] = Template(prompt["template"])

    def _load_prompt_file(self, name: str) -> dict:
        """加载单个 prompt 的 YAML 文件，合并 overrides

        Returns:
            {name, description, variables, template}
        """
        meta = self._registry[name]
        prompt_path = self._resolve_path(meta["file"])

        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = yaml.safe_load(f)

        # 验证必填字段
        for field in ["name", "description", "variables", "template"]:
            if field not in prompt:
                raise ValueError(f"Prompt 文件 {meta['file']} 缺少必填字段: {field}")

        # 合并 overrides
        override = self._load_override(name)
        if override:
            prompt = self._merge(name, override)

        return prompt

    def _load_override(self, name: str) -> Optional[dict]:
        """加载 override 文件内容（如有）"""
        path = self._override_file(name)
        if path is None:
            return None
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _override_file(self, name: str) -> Optional[str]:
        """获取 override 文件路径（如存在）"""
        meta = self._registry.get(name)
        if not meta:
            return None
        path = os.path.join(self._overrides_dir, meta["file"])
        return path if os.path.isfile(path) else None

    def _load_original_template(self, name: str) -> str:
        """加载源文件中的原始模板（忽略 overrides）"""
        meta = self._registry[name]
        prompt_path = self._resolve_path(meta["file"])
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = yaml.safe_load(f)
        return prompt.get("template", "")

    def _merge(self, name: str, override: dict) -> dict:
        """源 prompt + override 字段级合并

        只合并 override 中显式出现的字段，其他字段保持源文件原文。
        """
        meta = self._registry[name]
        prompt_path = self._resolve_path(meta["file"])
        with open(prompt_path, "r", encoding="utf-8") as f:
            source = yaml.safe_load(f)

        merged = dict(source)
        for field in ["template", "variables", "description"]:
            if field in override:
                merged[field] = override[field]
        return merged
