"""
缓存清理工具 — 删除所有未拟定标题的历史对话文件。

可通过 API 调用或直接运行脚本：
  python utils/cache_cleaner.py           # 直接运行
  POST /api/history/clear_cache           # API 调用
"""

import os
import json
import shutil
from datetime import datetime


def get_history_dir():
    """获取 history 目录路径（兼容 core.config 的 DIALOGUE_DATA_DIR）。"""
    try:
        from core.config import Config
        config = Config()
        return config.DIALOGUE_DATA_DIR
    except Exception:
        return os.path.join("dialogue data", "history")


def clear_untitled_sessions(history_dir: str = None) -> dict:
    """
    删除所有未拟定标题的会话文件夹，并清理 sessions_index.json。

    Args:
        history_dir: history 目录路径，为 None 时自动获取

    Returns:
        {"deleted_folders": [...], "deleted_count": int, "index_cleaned": int}
    """
    if history_dir is None:
        history_dir = get_history_dir()

    if not os.path.isdir(history_dir):
        return {"deleted_folders": [], "deleted_count": 0, "index_cleaned": 0, "error": "history 目录不存在"}

    deleted = []
    for name in sorted(os.listdir(history_dir)):
        folder = os.path.join(history_dir, name)
        # 只处理时间戳格式的文件夹
        if not (len(name) == 15 and name[8] == '_' and name.replace('_', '').isdigit()):
            continue
        chat_file = os.path.join(folder, "chat_history.json")
        title = None
        if os.path.isfile(chat_file):
            try:
                with open(chat_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                title = data.get("title")
            except Exception:
                pass
        # 无标题或未命名 → 删除
        if not title or title == "未命名会话":
            try:
                shutil.rmtree(folder)
                deleted.append(name)
            except Exception:
                pass

    # 清理 sessions_index.json 中已删除的条目
    index_cleaned = 0
    index_path = os.path.join(history_dir, "sessions_index.json")
    if os.path.isfile(index_path) and deleted:
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            before = len(index_data.get("sessions", []))
            index_data["sessions"] = [
                s for s in index_data.get("sessions", [])
                if s.get("timestamp") not in deleted
            ]
            index_cleaned = before - len(index_data["sessions"])
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return {
        "deleted_folders": deleted,
        "deleted_count": len(deleted),
        "index_cleaned": index_cleaned,
    }


# ── 直接运行 ──
if __name__ == "__main__":
    import sys, io
    if sys.platform == 'win32':
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        except Exception:
            pass

    # 确保使用绝对路径：从项目根目录定位 dialogue data/history
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    history_dir = os.path.join(project_root, "dialogue data", "history")
    print(f"[缓存清理] 目标目录: {history_dir}")
    result = clear_untitled_sessions(history_dir)
    print(f"[缓存清理] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  删除文件夹: {result['deleted_count']} 个")
    for name in result.get("deleted_folders", [])[:10]:
        print(f"    - {name}")
    if result["deleted_count"] > 10:
        print(f"    ... 共 {result['deleted_count']} 个")
    print(f"  索引清理: {result['index_cleaned']} 条")
