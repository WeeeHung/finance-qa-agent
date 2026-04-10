"""Optional per-turn rewrite: history + current → standalone question (no document)."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src_v1.answer_parse import extract_ai_text
from src_v1.llm import make_chat_model

REWRITE_SYSTEM = """You rewrite multi-turn financial dialogue into one standalone question.

You receive prior user questions in order (dialogue only — no answers) and the latest question. Rewrite **only** the latest question so it stands alone: resolve vague references (“this”, “that”, “during this time”, “the same”, “it”) using the prior questions so metrics, periods, and table rows are explicit.

Output rules:
- Output **only** the rewritten question text (one or two sentences max if needed).
- No preamble (“Rewritten:”), no JSON wrapper, no bullet list.
- Do not answer, compute, or add numbers — rewrite text only."""


def build_rewrite_human_message(question_history: list[str], current_question: str) -> str:
    payload = {
        "question_history": list(question_history),
        "current_question": current_question,
    }
    return (
        "Produce one standalone version of `current_question`.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def normalize_rewritten_question(text: str) -> str:
    """Strip fences, labels, outer quotes; collapse whitespace."""
    raw = (text or "").strip()
    if not raw:
        return ""
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()
    low = raw.lower()
    for prefix in ("rewritten question:", "rewritten:", "standalone question:", "question:"):
        if low.startswith(prefix):
            raw = raw.split(":", 1)[1].strip()
            low = raw.lower()
            break
    raw = raw.strip().strip('"').strip("'").strip()
    raw = " ".join(raw.split())
    return raw


def rewrite_current_question(
    question_history: list[str],
    current_question: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    One LLM call, no document. If ``question_history`` is empty, returns ``current_question`` as-is
    without calling the model.
    """
    if not question_history:
        return {
            "rewritten_question": current_question.strip(),
            "latency_ms": 0.0,
            "llm_ms_total": 0.0,
            "steps": [],
            "skipped": True,
        }

    human_text = build_rewrite_human_message(question_history, current_question)
    system = SystemMessage(content=REWRITE_SYSTEM)
    human = HumanMessage(content=human_text)
    chat = make_chat_model(model, temperature)

    if verbose:
        print("[src_v1] rewrite | invoke …", flush=True)

    t0 = time.perf_counter()
    ai_msg = chat.invoke([system, human])
    llm_ms = (time.perf_counter() - t0) * 1000.0

    raw_reply = extract_ai_text(ai_msg)
    rewritten = normalize_rewritten_question(raw_reply)
    if not rewritten:
        rewritten = current_question.strip()

    if verbose:
        print(f"[src_v1] rewrite | {llm_ms:.0f} ms | {rewritten!r}", flush=True)

    steps: list[dict[str, Any]] = [
        {"type": "rewrite_timing", "latency_ms": round(llm_ms, 3), "llm_ms_total": round(llm_ms, 3)},
        {"type": "rewrite_human", "content": human_text},
        {"type": "rewrite_assistant", "content": raw_reply},
    ]
    return {
        "rewritten_question": rewritten,
        "latency_ms": llm_ms,
        "llm_ms_total": llm_ms,
        "steps": steps,
        "skipped": False,
    }
