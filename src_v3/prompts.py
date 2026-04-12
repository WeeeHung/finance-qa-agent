"""Prompts for SRC_V3 KB extraction and KB-based ReAct answering."""

from __future__ import annotations

EXTRACTION_RULES = """You are extracting financial knowledge-base items in strict JSON.

Output contract:
- Return exactly one JSON object with shape: {"items": [...]} and no markdown fences.
- Each item must contain: statement, type, value, unit, derived_from, reasoning.
- `type` must be one of: "explicit", "implicit".
- `statement` must be standalone and include both the metric/topic and year when available.
- Fully expand magnitude using document scale (thousands/millions/billions). Example: "-4" in "$ millions" => -4000000.
- `value` must be numeric (int/float) or null. Never use words (million, k, M), currency symbols, commas, or scientific notation.
- `derived_from` and `reasoning` must be null for explicit items unless there is a true derivation.
- For implicit items, provide references in `derived_from` and reasoning with shape:
  {"op":"sub|add|mul|div|ratio|pct_change", "args":[{"ref": 2}, {"ref": 3}]}
- Do not add commentary outside JSON.
"""

KB_EXTRACTION_FEW_SHOT = """Example input snippet:
- Scale: USD millions
- Text: "In 2025, oil was charged at 70.30. Reference price was 100 with discount 29.70."

Example output:
{
  "items": [
    {
      "statement": "Oil charge in 2025 is 70300000 USD.",
      "type": "explicit",
      "value": 70300000,
      "unit": "USD",
      "derived_from": null,
      "reasoning": null
    },
    {
      "statement": "Reference price in 2025 is 100000000 USD.",
      "type": "explicit",
      "value": 100000000,
      "unit": "USD",
      "derived_from": null,
      "reasoning": null
    },
    {
      "statement": "Discount in 2025 is 29700000 USD.",
      "type": "explicit",
      "value": 29700000,
      "unit": "USD",
      "derived_from": null,
      "reasoning": null
    },
    {
      "statement": "Oil charge in 2025 equals reference price minus discount.",
      "type": "implicit",
      "value": 70300000,
      "unit": "USD",
      "derived_from": [2, 3],
      "reasoning": {
        "op": "sub",
        "args": [{"ref": 2}, {"ref": 3}]
      }
    }
  ]
}
"""

INITIAL_KB_SYSTEM = f"""{EXTRACTION_RULES}

Task: Build initial KB items from the structured document payload (JSON user message).

Document layout:
- `table_context`: immediate lines around the table — `last_pre_text_line` (last non-empty line
  of pre-table prose), `first_post_text_line` (first non-empty line of post-table prose), and
  `combined` (both lines). Use this to tie table rows/columns to surrounding narrative.
- `table`: string rendering of the numeric table (row/column labels and values).
- `text_chunks`: pre-table and post-table prose, each entry has `section` ("pre_table" or
  "post_table"), `chunk_index`, and `text`. Chunks are ~`chunking.sentences_per_chunk` sentences
  with `chunking.sentence_overlap` sentences overlapping the next chunk. Read all chunks; facts may
  span chunk boundaries due to overlap.
- `scale_units_note` and `document_id` provide scale and record identity.

Prefer high-confidence financial facts that are useful for numerical QA.
For this initial stage, output EXPLICIT facts only:
- type must be "explicit"
- derived_from must be null
- reasoning must be null
{KB_EXTRACTION_FEW_SHOT}
"""

TURN_APPEND_SYSTEM = f"""{EXTRACTION_RULES}

Task: Append only NEW KB items after a QA turn using the same schema.
Use question + rewritten question + final answer + KB snapshot to identify additions.
Avoid duplicates of existing facts unless a new derived item is created.
For this append stage, output IMPLICIT derived facts only:
- type must be "implicit"
- every item must include derived_from and reasoning
- each ref in derived_from must refer to an existing KB item id
- do not output standalone explicit facts in this stage
{KB_EXTRACTION_FEW_SHOT}
"""

REACT_SYSTEM = """You answer one ConvFinQA-style question using the KB context below.

User message is JSON with `question_history` and `current_question`.
Answer only the current question.

Use `execute_python` for arithmetic as needed.
When the KB includes magnitudes converted to absolute values, preserve that scale.

Final reply: your **entire last message** must be **only** the value—no other text:
- plain number with optional leading '-' and optional decimal point, or
- percent with trailing '%' when the question truly asks for a rate.
No words, commas, scientific notation, currency symbols, labels, or prefixes like "Answer:".
"""

REACT_SYSTEM_REWRITTEN_ONLY = """You answer one self-contained ConvFinQA-style question using KB context below.

The user message is already rewritten to be standalone.
Use `execute_python` for arithmetic when useful.

Final reply: your **entire last message** must be **only** the number (optional '-', '.') or a percent ending in '%', with no other words or text.
"""
