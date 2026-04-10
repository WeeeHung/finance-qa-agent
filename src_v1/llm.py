"""Single entry: build a LangChain chat model from env (no separate integrations package)."""

from __future__ import annotations

import os
from typing import Any


def make_chat_model(model: str | None = None, temperature: float = 0.0) -> Any:
    """
    ``CHAT_MODEL`` / ``OPENAI_MODEL``; bare name → ``openai:<name>``.
    Default: ``openai:gpt-4o-mini``.
    """
    from langchain.chat_models import init_chat_model

    raw = (model or os.environ.get("CHAT_MODEL") or os.environ.get("OPENAI_MODEL") or "").strip()
    if not raw:
        spec = "openai:gpt-4o-mini"
    elif ":" in raw:
        spec = raw
    else:
        spec = f"openai:{raw}"
    return init_chat_model(spec, temperature=temperature)
