"""
ramo_translate.fast_bypass
==========================
Instant 0ms conversational bypass dictionary for common affirmations, negations, and greetings.
Harvested from c:/ramo-audio/services/llm/engine.py.
Translates 20-30% of conversational meeting turns without model inference or VRAM overhead.
"""

import re
from typing import Optional, Dict

LANG_ALIASES: Dict[str, str] = {
    "fr": "french",
    "fra": "french",
    "french": "french",
    "es": "spanish",
    "spa": "spanish",
    "spanish": "spanish",
    "de": "german",
    "deu": "german",
    "german": "german",
    "ar": "arabic",
    "ara": "arabic",
    "arabic": "arabic",
    "en": "english",
    "eng": "english",
    "english": "english",
}

FAST_BYPASS_DICT: Dict[str, Dict[str, str]] = {
    "french": {
        "okay": "D'accord.",
        "ok": "D'accord.",
        "yes": "Oui.",
        "yeah": "Oui.",
        "yep": "Oui.",
        "no": "Non.",
        "nope": "Non.",
        "thank you": "Merci !",
        "thanks": "Merci !",
        "all right": "Très bien.",
        "alright": "Très bien.",
        "sure": "Bien sûr.",
        "sounds good": "Ça marche.",
        "bye": "Au revoir.",
        "goodbye": "Au revoir.",
        "hello": "Bonjour.",
        "hi": "Bonjour.",
    },
    "spanish": {
        "okay": "De acuerdo.",
        "ok": "De acuerdo.",
        "yes": "Sí.",
        "yeah": "Sí.",
        "yep": "Sí.",
        "no": "No.",
        "nope": "No.",
        "thank you": "¡Gracias!",
        "thanks": "¡Gracias!",
        "all right": "Muy bien.",
        "alright": "Muy bien.",
        "sure": "Claro.",
        "sounds good": "Suena bien.",
        "bye": "Adiós.",
        "goodbye": "Adiós.",
        "hello": "Hola.",
        "hi": "Hola.",
    },
    "german": {
        "okay": "In Ordnung.",
        "ok": "In Ordnung.",
        "yes": "Ja.",
        "yeah": "Ja.",
        "yep": "Ja.",
        "no": "Nein.",
        "nope": "Nein.",
        "thank you": "Danke!",
        "thanks": "Danke!",
        "all right": "Alles klar.",
        "alright": "Alles klar.",
        "sure": "Sicher.",
        "sounds good": "Klingt gut.",
        "bye": "Tschüss.",
        "goodbye": "Auf Wiedersehen.",
        "hello": "Hallo.",
        "hi": "Hallo.",
    },
    "arabic": {
        "okay": "حسناً.",
        "ok": "حسناً.",
        "yes": "نعم.",
        "yeah": "نعم.",
        "yep": "نعم.",
        "no": "لا.",
        "nope": "لا.",
        "thank you": "شكراً لك.",
        "thanks": "شكراً.",
        "all right": "حسناً.",
        "alright": "حسناً.",
        "sure": "بالتأكيد.",
        "sounds good": "يبدو جيداً.",
        "bye": "مع السلامة.",
        "goodbye": "مع السلامة.",
        "hello": "مرحباً.",
        "hi": "أهلاً.",
    },
}


def normalize_target_language(lang: str) -> str:
    """Normalize language code or name to canonical key."""
    clean = lang.strip().lower()
    return LANG_ALIASES.get(clean, clean)


def check_fast_bypass(text: str, target_lang: str) -> Optional[str]:
    """
    Check if text matches a common conversational phrase in the bypass dictionary.
    Returns translated string if found, otherwise None.
    """
    if not text:
        return None

    # Strip punctuation and whitespace
    norm = re.sub(r"[^\w\s]", "", text).strip().lower()
    norm = re.sub(r"\s+", " ", norm)
    if not norm:
        return None

    canonical_lang = normalize_target_language(target_lang)
    lang_dict = FAST_BYPASS_DICT.get(canonical_lang)
    if not lang_dict:
        return None

    return lang_dict.get(norm)
