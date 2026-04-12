"""Format ConvFinQA-style ``raw_data`` (record JSON) into a single context block."""

from __future__ import annotations

from typing import Any

import pandas as pd


def format_convfinqa_context(raw_data: dict[str, Any]) -> str:
    """Turn parsed record JSON into pre/table/post text for the LLM."""
    doc = raw_data.get("doc") or {}
    pre = (doc.get("pre_text") or "").strip()
    post = (doc.get("post_text") or "").strip()
    table = doc.get("table") or {}
    record_id = raw_data.get("id", "")

    if isinstance(table, dict) and table:
        try:
            df = pd.DataFrame.from_dict(table)
            table_block = df.to_string(index=True)
        except (ValueError, TypeError):
            table_block = str(table)
    else:
        table_block = "(no table)"

    parts = [
        f"Document id: {record_id}" if record_id else "",
        "=== Text before table ===",
        pre or "(empty)",
        "=== Table ===",
        table_block,
        "=== Scale / units ===",
        "Infer thousands/millions/billions from captions and headers; combine magnitudes correctly.",
        "=== Text after table ===",
        post or "(empty)",
    ]
    return "\n".join(p for p in parts if p is not None).strip()
