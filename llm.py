"""Thin wrapper around the Anthropic API."""
import json
import re

import anthropic

from dotenv import load_dotenv
load_dotenv()

from config import GEN_MODEL, JUDGE_MODEL

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    return _client


def complete(user: str, system: str = "", model: str = None, max_tokens: int = 1500) -> str:
    r = _get_client().messages.create(
        model=model or GEN_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in r.content if b.type == "text").strip()


def complete_json(user: str, system: str = "", model: str = None, max_tokens: int = 1500):
    """Ask for JSON only and parse defensively. Returns None if unparseable."""
    txt = complete(
        user,
        system + "\nRespond with valid JSON only: no prose, no code fences.",
        model or JUDGE_MODEL,
        max_tokens,
    )
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", txt, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                return None
        return None
