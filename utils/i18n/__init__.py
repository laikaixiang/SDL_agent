"""I18nHelper — reads shared locales JSON for Flask responses"""
import json
import os
from typing import Any


class I18nHelper:
    def __init__(self, locales_dir: str = None, default_lang: str = 'zh'):
        if locales_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            locales_dir = os.path.join(project_root, 'frontend', 'src', 'locales')
        self._default_lang = default_lang if default_lang in ('en', 'zh') else 'zh'
        self._translations: dict[str, dict[str, Any]] = {}
        for lang in ('en', 'zh'):
            path = os.path.join(locales_dir, f'{lang}.json')
            with open(path, 'r', encoding='utf-8') as f:
                self._translations[lang] = json.load(f)

    def get(self, key: str, lang: str = 'zh') -> str:
        """Resolve dot-separated path like 'chat.placeholder'"""
        translations = self._translations.get(lang, self._translations[self._default_lang])
        value: Any = translations
        for part in key.split('.'):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return key
        return value if isinstance(value, str) else key

    def get_lang(self, request) -> str:
        lang = request.cookies.get('lang', self._default_lang)
        return lang if lang in ('en', 'zh') else self._default_lang


def init_i18n(default_lang: str = 'zh') -> None:
    """Set the default language for the module-level i18n singleton.
    Call once at startup, before any requests are served."""
    if default_lang in ('en', 'zh'):
        i18n._default_lang = default_lang


i18n = I18nHelper()
