"""Knowledge-base storage and rendering utilities for SRC_V3."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterable

from src_v3.models import KBFile, KBItem, KBItemDraft


def _item_signature(item: KBItem | KBItemDraft) -> tuple[str, str, str, str]:
    value_key = "" if item.value is None else str(item.value)
    unit_key = (item.unit or "").strip().lower()
    type_key = item.type.strip().lower()
    stmt_key = item.statement.strip().lower()
    return (stmt_key, type_key, value_key, unit_key)


@dataclass
class KnowledgeBase:
    file_id: str
    items: list[KBItem] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._signatures: set[tuple[str, str, str, str]] = {_item_signature(i) for i in self.items}

    @property
    def next_id(self) -> int:
        if not self.items:
            return 1
        return max(i.id for i in self.items) + 1

    def append_drafts(self, drafts: Iterable[KBItemDraft]) -> list[KBItem]:
        added: list[KBItem] = []
        cur_id = self.next_id
        for d in drafts:
            sig = _item_signature(d)
            if sig in self._signatures:
                continue
            item = KBItem(
                id=cur_id,
                statement=d.statement.strip(),
                type=d.type,
                value=d.value,
                unit=d.unit.strip() if isinstance(d.unit, str) else d.unit,
                derived_from=d.derived_from,
                reasoning=d.reasoning,
            )
            self.items.append(item)
            self._signatures.add(sig)
            added.append(item)
            cur_id += 1
        return added

    def to_file_model(self) -> KBFile:
        return KBFile(file_id=self.file_id, items=list(self.items), metadata=dict(self.metadata))

    def to_context(self) -> str:
        if not self.items:
            return "No KB items available."
        lines: list[str] = []
        for item in self.items:
            unit = f" {item.unit}" if item.unit else ""
            lines.append(f"[{item.id}] {item.statement} | value={item.value}{unit} | type={item.type}")
            if item.derived_from:
                lines.append(f"  derived_from={item.derived_from}")
            if item.reasoning:
                lines.append(f"  reasoning={item.reasoning.model_dump(exclude_none=True)}")
        return "\n".join(lines)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = self.to_file_model().model_dump(exclude_none=False)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")

    @classmethod
    def load_json(cls, path: str) -> "KnowledgeBase":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        file_model = KBFile.model_validate(data)
        return cls(file_id=file_model.file_id, items=file_model.items, metadata=file_model.metadata)
