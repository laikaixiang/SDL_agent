"""Check that en.json and zh.json have identical key structures.

Exit 0 if consistent, exit 1 if mismatched (for pre-commit / CI).

Usage: python utils/i18n/check.py
"""
import json
import os
import sys


def extract_paths(obj, prefix=''):
    """Recursively extract all dot-separated key paths."""
    paths = set()
    for k, v in obj.items():
        path = f'{prefix}.{k}' if prefix else k
        if isinstance(v, dict):
            paths.update(extract_paths(v, path))
        else:
            paths.add(path)
    return paths


def check(locales_dir=None):
    if locales_dir is None:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        locales_dir = os.path.join(PROJECT_ROOT, 'frontend', 'src', 'locales')

    en_path = os.path.join(locales_dir, 'en.json')
    zh_path = os.path.join(locales_dir, 'zh.json')

    with open(en_path, 'r', encoding='utf-8') as f:
        en = json.load(f)
    with open(zh_path, 'r', encoding='utf-8') as f:
        zh = json.load(f)

    en_keys = extract_paths(en)
    zh_keys = extract_paths(zh)

    missing_in_zh = en_keys - zh_keys
    missing_in_en = zh_keys - en_keys

    if missing_in_zh or missing_in_en:
        if missing_in_zh:
            print(f'Keys in en.json missing from zh.json ({len(missing_in_zh)}):')
            for k in sorted(missing_in_zh):
                print(f'  - {k}')
        if missing_in_en:
            print(f'Keys in zh.json missing from en.json ({len(missing_in_en)}):')
            for k in sorted(missing_in_en):
                print(f'  - {k}')
        sys.exit(1)

    print(f'OK: {len(en_keys)} keys match between en.json and zh.json')
    sys.exit(0)


if __name__ == '__main__':
    check()
