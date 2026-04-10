"""Batch vanilla v1: one chat invoke per turn. Writes ``data/results_v1/<file_id>.json``."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Union

from src.bootstrap_env import load_project_env

load_project_env()

from tqdm import tqdm

from src.utils.data.read_dataset import DatasetDict
from src.utils.filepaths import dataset_fpath, results_v1_dir
from src_v1.prompt import build_vanilla_user_message
from src_v1.serialize import record_to_raw_data
from src_v1.vanilla import run_vanilla_turn

DEFAULT_DEV_SLICE = slice(0, 10)


def _final_answer_json_value(text: str) -> Union[str, float, int]:
    t = (text or "").strip()
    if not t:
        return ""
    if t.endswith("%"):
        return t
    try:
        if "." in t or "e" in t.lower() or "E" in t:
            return float(t)
        return int(t)
    except ValueError:
        return t


def _turn_record(
    *,
    question_history: list[str],
    question: str,
    final_answer: Any,
    latency_ms: float,
    steps: list[dict[str, Any]],
    llm_ms_total: float,
) -> dict[str, Any]:
    return {
        "question_history": list(question_history),
        "question": question,
        "final_answer": final_answer,
        "latency_ms": round(latency_ms, 3),
        "steps": steps,
        "metadata": {
            "llm_ms_total": llm_ms_total,
            "reason_pass": 1,
            "sandbox_invocations": 0,
        },
    }


def run(
    subset: str = "dev",
    record_slice: slice | None = None,
    *,
    overwrite: bool = False,
    verbose: bool = False,
) -> None:
    ds = DatasetDict(dataset_fpath)
    sl = record_slice if record_slice is not None else DEFAULT_DEV_SLICE
    records = ds.get_subset(subset).get_records()[sl]

    os.makedirs(results_v1_dir, exist_ok=True)

    for rec in tqdm(records, desc="src_v1"):
        out_path = os.path.join(results_v1_dir, f"{rec.file_id}.json")
        if os.path.isfile(out_path) and not overwrite:
            print(f"Skipping {rec.file_id}, exists: {out_path}")
            continue

        raw = record_to_raw_data(rec)
        prev: list[str] = []
        chunks: list[str] = []

        for q in rec.dialogue.conv_questions:
            user_message = build_vanilla_user_message(prev, q)
            t0 = time.perf_counter()
            out = run_vanilla_turn(raw, user_message, verbose=verbose)
            wall_ms = (time.perf_counter() - t0) * 1000.0

            answer_text = out["answer_text"]
            steps = list(out["steps"])
            # Prefer inner timing row as source of truth; wall includes tiny Python overhead
            if steps and steps[0].get("type") == "timing":
                steps[0]["latency_ms"] = round(wall_ms, 3)
                steps[0]["llm_ms_total"] = round(out["llm_ms_total"], 3)

            turn_obj = _turn_record(
                question_history=list(prev),
                question=q,
                final_answer=_final_answer_json_value(answer_text),
                latency_ms=wall_ms,
                steps=steps,
                llm_ms_total=float(out["llm_ms_total"]),
            )
            chunks.append(json.dumps(turn_obj, ensure_ascii=False, indent=2))
            prev.append(q)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(chunks) + ("\n" if chunks else ""))

        print(f"Wrote {out_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Vanilla v1: one LLM call per turn → results_v1/")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    run(
        record_slice=slice(args.start, args.end),
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
