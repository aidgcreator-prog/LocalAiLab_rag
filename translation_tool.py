"""Translation tool using TranslateGemma via Ollama /api/generate.

TranslateGemma does not support tool/function calling, so it must be invoked
via the raw generation endpoint rather than the chat completions endpoint.
This module exposes a single LangChain tool `translate_text` that the main
orchestrator agent can call directly.
"""

from __future__ import annotations

import os

import requests
from langchain_core.tools import tool


_OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_DEFAULT_MODEL = os.getenv("TRANSLATOR_MODEL", "translategemma:27b")

# Language code lookup for common languages so callers can pass plain names
_LANG_MAP: dict[str, tuple[str, str]] = {
    "khmer": ("Central Khmer", "km"),
    "cambodian": ("Central Khmer", "km"),
    "english": ("English", "en"),
    "french": ("French", "fr"),
    "spanish": ("Spanish", "es"),
    "german": ("German", "de"),
    "chinese": ("Chinese", "zh-Hans"),
    "japanese": ("Japanese", "ja"),
    "korean": ("Korean", "ko"),
    "arabic": ("Arabic", "ar"),
    "vietnamese": ("Vietnamese", "vi"),
    "thai": ("Thai", "th"),
    "russian": ("Russian", "ru"),
    "portuguese": ("Portuguese", "pt"),
    "italian": ("Italian", "it"),
    "dutch": ("Dutch", "nl"),
    "hindi": ("Hindi", "hi"),
    "indonesian": ("Indonesian", "id"),
    "turkish": ("Turkish", "tr"),
    "polish": ("Polish", "pl"),
}


def _resolve_lang(lang_input: str) -> tuple[str, str]:
    """Return (LanguageName, code) from a plain name or ISO code."""
    key = lang_input.strip().lower()
    if key in _LANG_MAP:
        return _LANG_MAP[key]
    # Assume the caller passed a raw ISO code — capitalise name from code
    return (lang_input.strip().title(), lang_input.strip().lower())


def _build_translategemma_prompt(
    source_lang: str,
    source_code: str,
    target_lang: str,
    target_code: str,
    text: str,
) -> str:
    """Build the exact prompt format required by TranslateGemma."""
    return (
        f"You are a professional {source_lang} ({source_code}) to "
        f"{target_lang} ({target_code}) translator. Your goal is to accurately "
        f"convey the meaning and nuances of the original {source_lang} text while "
        f"adhering to {target_lang} grammar, vocabulary, and cultural sensitivities.\n"
        f"Produce only the {target_lang} translation, without any additional "
        f"explanations or commentary. Please translate the following {source_lang} "
        f"text into {target_lang}:\n\n\n{text}"
    )


@tool(parse_docstring=True)
def translate_text(
    text: str,
    source_language: str = "auto",
    target_language: str = "english",
) -> str:
    """Translate text between languages using TranslateGemma (55 languages supported).

    Use this tool whenever the user sends a message in a non-English language or
    explicitly requests translation. Supports Khmer (km), French, Spanish, Chinese,
    Japanese, Arabic, Vietnamese, Thai, and 47 more languages.

    For Khmer input: set source_language='khmer', target_language='english'.

    Args:
        text: The text to translate.
        source_language: Source language name or ISO code (e.g. 'khmer', 'km', 'french', 'fr').
                         Use 'auto' to let the model detect it (defaults to English output).
        target_language: Target language name or ISO code (e.g. 'english', 'en'). Defaults to English.

    Returns:
        The translated text, or an error message if the request failed.
    """
    target_name, target_code = _resolve_lang(target_language)

    if source_language.strip().lower() == "auto":
        # Use a generic auto-detect prompt — TranslateGemma handles it
        source_name, source_code = "the source", "auto"
        prompt = (
            f"Detect the language of the following text and translate it into "
            f"{target_name} ({target_code}). "
            f"Produce only the {target_name} translation, without any additional "
            f"explanations or commentary.\n\n\n{text}"
        )
    else:
        source_name, source_code = _resolve_lang(source_language)
        prompt = _build_translategemma_prompt(
            source_lang=source_name,
            source_code=source_code,
            target_lang=target_name,
            target_code=target_code,
            text=text,
        )

    model = _DEFAULT_MODEL
    # Strip "ollama:" prefix if present — /api/generate uses bare model names
    if model.startswith("ollama:"):
        model = model[len("ollama:"):]

    try:
        response = requests.post(
            f"{_OLLAMA_BASE}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        translation = data.get("response", "").strip()
        if not translation:
            return "[Translation returned empty — the model may still be loading.]"
        return translation
    except requests.exceptions.ConnectionError:
        return (
            "[Translation failed: Ollama is not reachable at "
            f"{_OLLAMA_BASE}. Ensure Ollama is running.]"
        )
    except requests.exceptions.HTTPError as exc:
        return f"[Translation failed: HTTP {exc.response.status_code} — {exc.response.text[:200]}]"
    except Exception as exc:
        return f"[Translation failed: {exc}]"
