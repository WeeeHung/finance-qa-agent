"""Structured document views for SRC_V3 KB extraction: sentence chunks + table context."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.utils.data.types import Document

# Sentence boundaries; FinQA prose often uses ". " between clauses.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = _SENTENCE_SPLIT.split(t)
    return [p.strip() for p in parts if p.strip()]


def chunk_sentences(
    sentences: list[str],
    *,
    window: int = 4,
    overlap: int = 1,
) -> list[str]:
    """Sliding windows of `window` sentences, advancing by `window - overlap` sentences."""
    if overlap >= window:
        raise ValueError("sentence_overlap must be less than sentences_per_chunk")
    if not sentences:
        return []
    step = window - overlap
    out: list[str] = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i : i + window]
        if chunk:
            out.append(" ".join(chunk))
        if len(chunk) < window:
            break
        i += step
    return out


def table_context_fields(doc: Document) -> dict[str, str]:
    """Lines adjacent to the table (same semantics as ``Document`` properties, empty-safe)."""
    pre_lines = [line for line in doc.pre_text.split("\n") if line.strip()]
    post_lines = [line for line in doc.post_text.split("\n") if line.strip()]
    last_pre = pre_lines[-1] if pre_lines else ""
    first_post = post_lines[0] if post_lines else ""
    combined = f"{last_pre}\n{first_post}" if (last_pre or first_post) else ""
    return {
        "last_pre_text_line": last_pre,
        "first_post_text_line": first_post,
        "combined": combined.strip(),
    }


def format_table_block(table: dict[str, Any] | Any) -> str:
    if isinstance(table, dict) and table:
        try:
            df = pd.DataFrame.from_dict(table)
            return df.to_string(index=True)
        except (ValueError, TypeError):
            return str(table)
    return "(no table)"


def build_initial_kb_document_payload(
    raw_data: dict[str, Any],
    *,
    sentences_per_chunk: int = 4,
    sentence_overlap: int = 1,
) -> dict[str, Any]:
    """
    Build JSON-serializable document fields for initial KB extraction.

    Prose is split into sentence windows (with overlap). Table numeric block and
    immediate table-adjacent lines (``Document``-style table context) are separate
    from those chunks so the model can anchor facts to the table.
    """
    doc_dict = raw_data.get("doc") or {}
    doc = Document(
        pre_text=doc_dict.get("pre_text") or "",
        post_text=doc_dict.get("post_text") or "",
        table=doc_dict.get("table") or {},
    )
    ctx = table_context_fields(doc)
    table_block = format_table_block(doc_dict.get("table") or {})

    pre_chunks = chunk_sentences(
        split_sentences(doc.pre_text),
        window=sentences_per_chunk,
        overlap=sentence_overlap,
    )
    post_chunks = chunk_sentences(
        split_sentences(doc.post_text),
        window=sentences_per_chunk,
        overlap=sentence_overlap,
    )

    text_chunks: list[dict[str, Any]] = []
    for idx, text in enumerate(pre_chunks):
        text_chunks.append({"section": "pre_table", "chunk_index": idx, "text": text})
    for idx, text in enumerate(post_chunks):
        text_chunks.append({"section": "post_table", "chunk_index": idx, "text": text})

    record_id = raw_data.get("id") or ""

    return {
        "document_id": record_id,
        "table_context": {
            "last_pre_text_line": ctx["last_pre_text_line"],
            "first_post_text_line": ctx["first_post_text_line"],
            "combined": ctx["combined"],
        },
        "table": table_block,
        "scale_units_note": (
            "Infer thousands/millions/billions from captions and headers; "
            "combine magnitudes correctly."
        ),
        "text_chunks": text_chunks,
        "chunking": {
            "sentences_per_chunk": sentences_per_chunk,
            "sentence_overlap": sentence_overlap,
        },
    }
