"""
算法迁移工具 (software/algorithms/migrate.py)
============================================

将 extra_algorithms_fromProjects/ 下的自定义算法迁移（复制）到 default/ 内置算法目录。

迁移内容：
    1. 算法 .py 源文件（覆盖同名文件）
    2. REGISTRY.json 中对应的算法条目（合并到 default/REGISTRY.json）

使用示例：
    from software.algorithms.migrate import migrate_algorithm, migrate_all

    # 迁移单个算法
    migrate_algorithm("moving_average")

    # 迁移全部自定义算法
    migrate_all()
"""

import json
import os
import shutil
from datetime import datetime
from typing import Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, "extra_algorithms_fromProjects")
_DST_DIR = os.path.join(_THIS_DIR, "default")
_SRC_REGISTRY = os.path.join(_SRC_DIR, "REGISTRY.json")
_DST_REGISTRY = os.path.join(_DST_DIR, "REGISTRY.json")

_SKIP_FILES = {"__init__.py", "prompt_template.py", "README.md"}


def _load_registry(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"algorithms": [], "version": "1.0.0", "last_updated": ""}


def _save_registry(path: str, registry: dict) -> None:
    registry["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def _upsert_entry(algorithms: list, entry: dict) -> str:
    """将 entry 合入 algorithms 列表；按 name 判断是新增还是覆盖。返回 'added' 或 'updated'"""
    for i, algo in enumerate(algorithms):
        if algo.get("name") == entry.get("name"):
            algorithms[i] = entry
            return "updated"
    algorithms.append(entry)
    return "added"


def migrate_algorithm(name: str, overwrite: bool = True) -> dict:
    """
    迁移单个自定义算法到 default/ 目录。

    Args:
        name     : 算法标识（对应 extra_algorithms_fromProjects/{name}.py）
        overwrite: 若 default/ 下已存在同名文件，是否覆盖。默认 True

    Returns:
        {
            "success" : bool,
            "name"    : str,
            "file"    : str | None,   # 目标文件路径
            "registry": str | None,   # 'added' / 'updated' / None
            "message" : str
        }
    """
    src_file = os.path.join(_SRC_DIR, f"{name}.py")
    dst_file = os.path.join(_DST_DIR, f"{name}.py")

    if not os.path.exists(src_file):
        return {
            "success": False,
            "name": name,
            "file": None,
            "registry": None,
            "message": f"源文件不存在: {src_file}",
        }

    if os.path.exists(dst_file) and not overwrite:
        return {
            "success": False,
            "name": name,
            "file": dst_file,
            "registry": None,
            "message": f"目标文件已存在且 overwrite=False: {dst_file}",
        }

    shutil.copy2(src_file, dst_file)

    registry_action: Optional[str] = None
    src_registry = _load_registry(_SRC_REGISTRY)
    entry = next(
        (a for a in src_registry.get("algorithms", []) if a.get("name") == name),
        None,
    )

    if entry is None:
        entry = {
            "name": name,
            "description": "",
            "category": "自定义算法",
            "input_type": "未指定",
            "keywords": [name],
        }

    dst_registry = _load_registry(_DST_REGISTRY)
    algorithms = dst_registry.setdefault("algorithms", [])
    registry_action = _upsert_entry(algorithms, entry)
    _save_registry(_DST_REGISTRY, dst_registry)

    return {
        "success": True,
        "name": name,
        "file": dst_file,
        "registry": registry_action,
        "message": f"已迁移 {name}.py 并 {registry_action} 注册表",
    }


def migrate_all(overwrite: bool = True) -> dict:
    """
    迁移 extra_algorithms_fromProjects/ 下所有自定义算法到 default/。

    自动跳过 __init__.py / prompt_template.py / README.md / __pycache__ / REGISTRY.json。

    Args:
        overwrite: 是否覆盖 default/ 下的同名文件。默认 True

    Returns:
        {
            "success": bool,
            "migrated": [ {迁移结果}, ... ],
            "skipped" : [str, ...],
            "message" : str
        }
    """
    if not os.path.isdir(_SRC_DIR):
        return {
            "success": False,
            "migrated": [],
            "skipped": [],
            "message": f"源目录不存在: {_SRC_DIR}",
        }

    results = []
    skipped = []
    for filename in os.listdir(_SRC_DIR):
        if filename in _SKIP_FILES or filename == "REGISTRY.json":
            skipped.append(filename)
            continue
        full = os.path.join(_SRC_DIR, filename)
        if not os.path.isfile(full) or not filename.endswith(".py"):
            skipped.append(filename)
            continue
        name = filename[:-3]
        results.append(migrate_algorithm(name, overwrite=overwrite))

    ok = sum(1 for r in results if r["success"])
    return {
        "success": ok == len(results),
        "migrated": results,
        "skipped": skipped,
        "message": f"迁移完成：成功 {ok}/{len(results)}，跳过 {len(skipped)} 个文件",
    }


if __name__ == "__main__":
    pass
    # import pprint
    # pprint.pp(migrate_all())
