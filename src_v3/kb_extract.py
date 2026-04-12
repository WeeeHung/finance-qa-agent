"""LLM-assisted KB extraction for SRC_V3."""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src_v1.llm import make_chat_model
from src_v3.kb_document_context import build_initial_kb_document_payload
from src_v3.models import KBExtraction, KBItemDraft
from src_v3.prompts import INITIAL_KB_SYSTEM, TURN_APPEND_SYSTEM
from src_v3.kb_store import KnowledgeBase


def _extract_json_object(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    if not s:
        raise ValueError("Empty model output")
    if "```" in s:
        parts = [p.strip() for p in s.split("```") if p.strip()]
        for p in parts:
            if p.startswith("{") and p.endswith("}"):
                s = p
                break
            if "\n" in p:
                maybe = p.split("\n", 1)[1].strip()
                if maybe.startswith("{") and maybe.endswith("}"):
                    s = maybe
                    break
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return json.loads(s[start : end + 1])


def _run_structured_extraction(
    *,
    system_prompt: str,
    payload: dict[str, Any],
    model: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    chat = make_chat_model(model, temperature)
    user_payload = json.dumps(payload, ensure_ascii=False, indent=2)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_payload),
    ]

    t0 = time.perf_counter()
    out = chat.invoke(messages)
    llm_ms = (time.perf_counter() - t0) * 1000.0

    content = getattr(out, "content", "")
    raw_text = content if isinstance(content, str) else str(content)
    parsed = _extract_json_object(raw_text)
    extraction = KBExtraction.model_validate(parsed)
    return {
        "items": extraction.items,
        "llm_ms_total": llm_ms,
        "llm_invocations": 1,
        "raw_response": raw_text,
    }


def _sanitize_initial_items(items: list[KBItemDraft]) -> list[KBItemDraft]:
    """Initial KB build accepts explicit facts only."""
    cleaned: list[KBItemDraft] = []
    for item in items:
        if item.type != "explicit":
            continue
        cleaned.append(
            KBItemDraft(
                statement=item.statement,
                type="explicit",
                value=item.value,
                unit=item.unit,
                derived_from=None,
                reasoning=None,
            )
        )
    return cleaned


def _sanitize_append_items(items: list[KBItemDraft], *, allowed_refs: set[int]) -> list[KBItemDraft]:
    """Post-turn append accepts implicit facts only and requires grounded references."""
    cleaned: list[KBItemDraft] = []
    for item in items:
        if item.type != "implicit":
            continue
        refs = item.derived_from or []
        if not refs or item.reasoning is None:
            continue
        if not all(isinstance(r, int) and r in allowed_refs for r in refs):
            continue
        cleaned.append(
            KBItemDraft(
                statement=item.statement,
                type="implicit",
                value=item.value,
                unit=item.unit,
                derived_from=list(refs),
                reasoning=item.reasoning,
            )
        )
    return cleaned


def empty_turn_kb_updates(*, reason: str = "no_execute_python") -> dict[str, Any]:
    """Same shape as ``extract_turn_kb_updates`` when no LLM append is needed (e.g. no tool runs)."""
    return {
        "items": [],
        "llm_ms_total": 0.0,
        "llm_invocations": 0,
        "steps": [
            {
                "type": "kb_extraction",
                "stage": "append",
                "skipped": True,
                "reason": reason,
                "llm_ms_total": 0.0,
                "extracted_items": 0,
                "accepted_items": 0,
            },
        ],
    }


def extract_initial_kb(
    raw_data: dict[str, Any],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    sentences_per_chunk: int = 4,
    sentence_overlap: int = 1,
) -> dict[str, Any]:
    doc_payload = build_initial_kb_document_payload(
        raw_data,
        sentences_per_chunk=sentences_per_chunk,
        sentence_overlap=sentence_overlap,
    )
    n_chunks = len(doc_payload.get("text_chunks") or [])
    payload = {"task": "initial_kb_build", **doc_payload}
    out = _run_structured_extraction(
        system_prompt=INITIAL_KB_SYSTEM,
        payload=payload,
        model=model,
        temperature=temperature,
    )
    raw_items = [KBItemDraft.model_validate(i) for i in out["items"]]
    items = _sanitize_initial_items(raw_items)
    return {
        "items": items,
        "llm_ms_total": out["llm_ms_total"],
        "llm_invocations": out["llm_invocations"],
        "steps": [
            {
                "type": "kb_extraction",
                "stage": "initial",
                "llm_ms_total": round(out["llm_ms_total"], 3),
                "extracted_items": len(raw_items),
                "accepted_items": len(items),
                "text_chunks": n_chunks,
                "sentences_per_chunk": sentences_per_chunk,
                "sentence_overlap": sentence_overlap,
            },
        ],
    }


def extract_turn_kb_updates(
    *,
    kb: KnowledgeBase,
    question_history: list[str],
    question: str,
    rewritten_question: str | None,
    final_answer: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    payload = {
        "task": "post_turn_kb_update",
        "question_history": question_history,
        "question": question,
        "rewritten_question": rewritten_question,
        "final_answer": final_answer,
        "existing_kb_items": [it.model_dump(exclude_none=False) for it in kb.items],
    }
    out = _run_structured_extraction(
        system_prompt=TURN_APPEND_SYSTEM,
        payload=payload,
        model=model,
        temperature=temperature,
    )
    raw_items = [KBItemDraft.model_validate(i) for i in out["items"]]
    allowed_refs = {it.id for it in kb.items}
    items = _sanitize_append_items(raw_items, allowed_refs=allowed_refs)
    return {
        "items": items,
        "llm_ms_total": out["llm_ms_total"],
        "llm_invocations": out["llm_invocations"],
        "steps": [
            {
                "type": "kb_extraction",
                "stage": "append",
                "llm_ms_total": round(out["llm_ms_total"], 3),
                "extracted_items": len(raw_items),
                "accepted_items": len(items),
            },
        ],
    }
