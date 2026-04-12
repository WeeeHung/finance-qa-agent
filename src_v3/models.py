"""Typed schema for SRC_V3 knowledge-base extraction and persistence."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ReasonArgRef(BaseModel):
    ref: int = Field(..., description="Reference ID to another KB item")


class Reasoning(BaseModel):
    op: str = Field(..., description="Operation name, e.g. add/sub/mul/div")
    args: list[ReasonArgRef] = Field(
        default_factory=list,
        description="Operation arguments by KB reference",
    )


class KBItemDraft(BaseModel):
    statement: str = Field(
        ...,
        description="Standalone statement including the metric/topic and year when available",
    )
    type: Literal["explicit", "implicit"] = Field(..., description="Fact type")
    value: float | int | None = Field(
        default=None,
        description="Fully expanded numeric value with document scale applied",
    )
    unit: str | None = Field(default=None, description="Unit symbol/text, e.g. USD, %")
    derived_from: list[int] | None = Field(
        default=None,
        description="References for implicit facts, null for explicit facts",
    )
    reasoning: Reasoning | None = Field(
        default=None,
        description="Derivation plan for implicit facts, null for explicit facts",
    )


class KBItem(KBItemDraft):
    id: int = Field(..., description="Unique KB item ID")


class KBExtraction(BaseModel):
    items: list[KBItemDraft] = Field(default_factory=list)


class KBFile(BaseModel):
    file_id: str
    items: list[KBItem] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
