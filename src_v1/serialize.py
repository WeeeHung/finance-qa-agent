"""Build ``raw_data`` dicts for ``src_v1`` from existing dataset models."""

from __future__ import annotations

from typing import Any

from src.utils.data.types import ConvFinQARecord


def record_to_raw_data(rec: ConvFinQARecord) -> dict[str, Any]:
    """Match ConvFinQA JSON shape enough for ``format_convfinqa_context``."""
    table = rec.doc.table
    table_dict = table.to_dict() if hasattr(table, "to_dict") else dict(table)
    return {
        "id": rec.id,
        "doc": {
            "pre_text": rec.doc.pre_text,
            "post_text": rec.doc.post_text,
            "table": table_dict,
        },
        "dialogue": rec.dialogue.model_dump(),
        "features": rec.features.model_dump(),
    }
