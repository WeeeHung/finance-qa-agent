"""One system + one human message → one assistant message. No LangGraph, no tools."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src_v1.answer_parse import extract_ai_text, normalize_convfinqa_answer
from src_v1.context import format_convfinqa_context
from src_v1.llm import make_chat_model

SYSTEM = """You answer one ConvFinQA-style question from the document (text + table).

**User message:** JSON with ``question_history`` (prior turns, context only) and ``current_question`` (answer only this). Vague follow-ups usually continue the same table topic as the **immediately previous** question.

**Rules:**
- Answer only ``current_question``.
- If the question names a row or metric, use that row’s figures.
- When the document states amounts are in thousands, millions, billions, etc., **convert to absolute numeric values** in your head and output the **fully expanded** number (no implicit unit). Example: values labeled “in millions” with a change of -4 → output **-4000000**, not **-4**.

**Output:** Your entire reply must be **only** the final numeric value: optional leading `-`, digits, optional `.` and fractional digits. For answers that are genuinely a **rate or percent**, you may use a trailing **%** (e.g. `37.5%`). Do **not** use words (million, k, M), scientific notation, `$`, or commas. No sentences, labels, or prefixes like ``Answer:``."""

# User message is already a single standalone question (after optional rewrite step).
SYSTEM_REWRITTEN_ONLY = """You answer one self-contained ConvFinQA-style question from the document (text + table).

The user message states one question only (it was rewritten from a multi-turn dialogue so references are explicit). Answer that question.

**Rules:**
- If the question names a row or metric, use that row’s figures.
- When the document states amounts are in thousands, millions, billions, etc., **convert to absolute numeric values** in your head and output the **fully expanded** number (no implicit unit). Example: values labeled “in millions” with a change of -4 → output **-4000000**, not **-4**.

**Output:** Your entire reply must be **only** the final numeric value: optional leading `-`, digits, optional `.` and fractional digits. For answers that are genuinely a **rate or percent**, you may use a trailing **%** (e.g. `37.5%`). Do **not** use words (million, k, M), scientific notation, `$`, or commas. No sentences, labels, or prefixes like ``Answer:``."""


def run_vanilla_turn(
    raw_data: dict[str, Any],
    user_message: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    verbose: bool = False,
    answer_style: str = "history_json",
) -> dict[str, Any]:
    """
    Returns ``answer_text``, ``latency_ms`` (wall / LLM invoke), and ``steps`` for logging.

    ``answer_style``:
    - ``history_json`` — user message matches :func:`build_vanilla_user_message` (default ``SYSTEM``).
    - ``rewritten_only`` — user message is a standalone question (:func:`build_rewritten_answer_user_message`).
    """
    ctx = format_convfinqa_context(raw_data)
    sys_block = SYSTEM_REWRITTEN_ONLY if answer_style == "rewritten_only" else SYSTEM
    system = SystemMessage(content=f"{sys_block}\n\n=== Document context ===\n{ctx}")
    human = HumanMessage(content=user_message)
    chat = make_chat_model(model, temperature)

    if verbose:
        rid = raw_data.get("id", "")
        print(f"[src_v1] id={rid!r} | invoke …", flush=True)

    t0 = time.perf_counter()
    ai_msg = chat.invoke([system, human])
    llm_ms = (time.perf_counter() - t0) * 1000.0

    raw_reply = extract_ai_text(ai_msg)
    answer_text = normalize_convfinqa_answer(raw_reply)

    if verbose:
        print(f"[src_v1] {llm_ms:.0f} ms | reply={answer_text!r}", flush=True)

    steps: list[dict[str, Any]] = [
        {
            "type": "timing",
            "latency_ms": round(llm_ms, 3),
            "llm_ms_total": round(llm_ms, 3),
        },
        {"type": "human", "content": user_message},
        {"type": "assistant", "content": raw_reply},
    ]
    return {
        "answer_text": answer_text,
        "latency_ms": llm_ms,
        "llm_ms_total": llm_ms,
        "steps": steps,
    }
