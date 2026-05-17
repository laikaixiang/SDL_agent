"""Auto-translate [待翻译] entries in zh.json using LLM.

Reads en.json as English source, scans zh.json for entries with '[待翻译]'
prefix, sends batches to LLM for translation, updates zh.json in place.

Usage: python utils/i18n/translate.py [--batch-size 15]
"""
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.config import config_obj
from core.llm_client import LLMClient

LOCALES_DIR = os.path.join(PROJECT_ROOT, 'frontend', 'src', 'locales')


def find_pending(zh) -> list:
    """Find all entries with [待翻译] prefix, return [(key_path, en_value), ...]"""
    pending = []

    def walk(obj, prefix=''):
        for k, v in obj.items():
            path = f'{prefix}.{k}' if prefix else k
            if isinstance(v, dict):
                walk(v, path)
            elif isinstance(v, str) and v.startswith('[待翻译]'):
                en_val = v[len('[待翻译] '):]
                pending.append((path, en_val))
    walk(zh)
    return pending


def apply_translations(zh, translations: dict):
    """Apply translated values back to zh dict."""
    for path, value in translations.items():
        parts = path.split('.')
        cur = zh
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = value


def translate_batch(client: LLMClient, items: list) -> dict:
    """Call LLM to translate a batch. Returns {path: chinese_value, ...}"""
    lines = '\n'.join(f'{i+1}. {en_val}' for i, (_, en_val) in enumerate(items))
    prompt = (
        f'Translate the following English UI strings to Simplified Chinese (简体中文). '
        f'Return ONLY a JSON object mapping the number to the Chinese translation. '
        f'Keep it concise and natural for a lab automation software UI.\n\n{lines}'
    )

    resp = client.call_api([
        {'role': 'user', 'content': prompt}
    ], stream=False, temperature=0.1, max_tokens=2000)

    raw = resp.get('content', '') if isinstance(resp, dict) else str(resp)
    try:
        raw = raw.replace('```json', '').replace('```', '').strip()
        idx_to_val = json.loads(raw)
        result = {}
        for idx_str, val in idx_to_val.items():
            i = int(idx_str) - 1
            result[items[i][0]] = val
        return result
    except (json.JSONDecodeError, KeyError, IndexError, ValueError):
        print(f'  Parse error, raw response: {raw[:200]}')
        return {}


def main(batch_size=15):
    en_path = os.path.join(LOCALES_DIR, 'en.json')
    zh_path = os.path.join(LOCALES_DIR, 'zh.json')

    with open(en_path, 'r', encoding='utf-8') as f:
        en = json.load(f)
    with open(zh_path, 'r', encoding='utf-8') as f:
        zh = json.load(f)

    pending = find_pending(zh)
    if not pending:
        print('No pending translations found.')
        return

    print(f'Found {len(pending)} entries to translate')

    client = LLMClient(
        api_key=config_obj.TALK_API_KEY or config_obj.API_KEY,
        api_url=config_obj.TALK_API_URL or config_obj.API_URL,
        model=config_obj.TALK_MODEL_NAME or config_obj.MODEL_NAME_TALK,
        extra_body=config_obj.get_extra_body('TALK'),
    )

    for i in range(0, len(pending), batch_size):
        batch = pending[i:i + batch_size]
        print(f'  Batch {i//batch_size + 1}: {len(batch)} items...')
        translations = translate_batch(client, batch)
        apply_translations(zh, translations)
        for path in batch:
            if path[0] in translations:
                print(f'    OK {path[0]}')
            else:
                print(f'    FAIL {path[0]}')

    with open(zh_path, 'w', encoding='utf-8') as f:
        json.dump(zh, f, ensure_ascii=False, indent=2)

    remaining = len(find_pending(zh))
    print(f'\nDone. {remaining} entries still pending.')


if __name__ == '__main__':
    main()
