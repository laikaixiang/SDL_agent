"""Align zh.json key structure with en.json (source of truth).

Missing keys in zh.json are filled with "[待翻译] <en value>".
Extra keys in zh.json (not in en.json) are removed.

Usage: python utils/i18n/sync.py
"""
import json
import sys
import os


def extract_keys(obj, prefix=''):
    """Recursively extract all dot-separated key paths."""
    keys = set()
    for k, v in obj.items():
        path = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            keys.update(extract_keys(v, path))
        else:
            keys.add(path)
    return keys


def get_value(obj, path):
    parts = path.split('.')
    cur = obj
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur if isinstance(cur, str) else None


def set_value(obj, path, value):
    parts = path.split('.')
    cur = obj
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def sync(zh_path, en_path):
    with open(en_path, 'r', encoding='utf-8') as f:
        en = json.load(f)
    with open(zh_path, 'r', encoding='utf-8') as f:
        zh = json.load(f)

    en_keys = extract_keys(en)
    zh_keys = extract_keys(zh)

    added, removed = 0, 0

    # Add missing keys
    for key in sorted(en_keys - zh_keys):
        en_val = get_value(en, key)
        set_value(zh, key, f'[待翻译] {en_val}')
        print(f'  + {key}')
        added += 1

    # Remove extra keys
    for key in sorted(zh_keys - en_keys):
        parts = key.split('.')
        parent = zh
        for p in parts[:-1]:
            parent = parent[p]
        if parts[-1] in parent:
            del parent[parts[-1]]
        print(f'  - {key}')
        removed += 1

    with open(zh_path, 'w', encoding='utf-8') as f:
        json.dump(zh, f, ensure_ascii=False, indent=2)

    print(f'\nDone: +{added} added, -{removed} removed')
    return added, removed


if __name__ == '__main__':
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    LOCALES_DIR = os.path.join(PROJECT_ROOT, 'frontend', 'src', 'locales')
    en = os.path.join(LOCALES_DIR, 'en.json')
    zh = os.path.join(LOCALES_DIR, 'zh.json')
    if not os.path.exists(en):
        print(f'Error: {en} not found')
        sys.exit(1)
    if not os.path.exists(zh):
        print(f'Error: {zh} not found')
        sys.exit(1)
    sync(zh, en)
