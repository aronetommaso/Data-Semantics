"""Shared Groq API key rotation utilities.

Keys are read from `.env` variables named:
  - GROQ_API_KEY
  - GROQ_API_KEY1
  - GROQ_API_KEY2
  - ...

The first configured key is used until Groq returns a 429 rate-limit response,
then the process moves to the next configured key.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
env_path = SCRIPT_DIR / ".env"
if not env_path.exists():
    env_path = SCRIPT_DIR.parent / ".env"
load_dotenv(dotenv_path=env_path)

_KEY_PATTERN = re.compile(r"^GROQ_API_KEY(\d*)$")
_KEYS: list[tuple[str, str]] | None = None
_CURRENT_INDEX = 0


def _key_sort(item: tuple[str, str]) -> tuple[int, str]:
    match = _KEY_PATTERN.match(item[0])
    suffix = match.group(1) if match else ""
    return (0 if suffix == "" else int(suffix), item[0])


def groq_api_keys() -> list[tuple[str, str]]:
    global _KEYS
    if _KEYS is None:
        keys = [
            (name, value.strip())
            for name, value in os.environ.items()
            if _KEY_PATTERN.match(name) and value.strip()
        ]
        _KEYS = sorted(keys, key=_key_sort)
    return _KEYS


def groq_api_key_count() -> int:
    return len(groq_api_keys())


def current_groq_api_key() -> str:
    keys = groq_api_keys()
    if not keys:
        raise ValueError("No Groq API key found. Set GROQ_API_KEY or GROQ_API_KEY1 in .env")
    return keys[_CURRENT_INDEX % len(keys)][1]


def current_groq_key_name() -> str:
    keys = groq_api_keys()
    if not keys:
        return "none"
    return keys[_CURRENT_INDEX % len(keys)][0]


def rotate_groq_api_key() -> str:
    global _CURRENT_INDEX
    keys = groq_api_keys()
    if not keys:
        raise ValueError("No Groq API key found. Set GROQ_API_KEY or GROQ_API_KEY1 in .env")
    _CURRENT_INDEX = (_CURRENT_INDEX + 1) % len(keys)
    return current_groq_api_key()


def require_groq_api_key() -> str:
    return current_groq_api_key()
