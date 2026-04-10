"""Extract and normalize a ConvFinQA-style scalar from the model reply."""

from __future__ import annotations

import re
from typing import Any

_LONE_SCALAR = re.compile(r"^\s*-?[\d,]+(?:\.\d+)?%?\s*$")


def extract_ai_text(msg: Any) -> str:
    """Plain text from an AIMessage / BaseMessage."""
    content = getattr(msg, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts).strip()
    return str(content or "").strip()


def _canonical_scalar(s: str) -> str:
    """Drop thousands separators so -4,000,000 → -4000000."""
    s = s.strip()
    if s.endswith("%"):
        return s[:-1].replace(",", "").strip() + "%"
    return s.replace(",", "")


def normalize_convfinqa_answer(text: str) -> str:
    """Single scalar line; strip fences and 'Answer:' prefixes."""
    raw = (text or "").strip()
    if not raw:
        return ""
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
    low = raw.lower()
    if low.startswith("answer:"):
        raw = raw.split(":", 1)[1].strip()
    if _LONE_SCALAR.match(raw):
        return _canonical_scalar(raw)
    for line in raw.splitlines():
        s = line.strip()
        if _LONE_SCALAR.match(s):
            return _canonical_scalar(s)
    return raw.strip()
