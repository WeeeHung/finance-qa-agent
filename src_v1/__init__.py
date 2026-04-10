"""Vanilla ConvFinQA runner: document context + one chat completion per turn (no LangGraph)."""

__version__ = "1.0.0"

from src_v1.context import format_convfinqa_context
from src_v1.serialize import record_to_raw_data
from src_v1.vanilla import run_vanilla_turn

__all__ = ["format_convfinqa_context", "record_to_raw_data", "run_vanilla_turn"]
