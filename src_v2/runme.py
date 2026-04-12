"""Batch SRC_V2: ReAct per turn with optional rewrite step.

Writes under ``data/results_v2/``.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Union

from src.bootstrap_env import load_project_env

load_project_env()

from tqdm import tqdm

from src.utils.data.read_dataset import DatasetDict
from src.utils.filepaths import dataset_fpath, results_v2_dir
from src_v1.prompt import build_rewritten_answer_user_message, build_vanilla_user_message
from src_v1.rewrite import rewrite_current_question
from src_v1.serialize import record_to_raw_data
from src_v2.react_turn import run_react_turn

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
    mode: str,
    rewritten_question: str | None,
    llm_invocations: int,
    sandbox_invocations: int,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "mode": mode,
        "question_history": list(question_history),
        "question": question,
        "final_answer": final_answer,
        "latency_ms": round(latency_ms, 3),
        "steps": steps,
        "metadata": {
            "llm_ms_total": llm_ms_total,
            "llm_invocations": llm_invocations,
            "reason_pass": llm_invocations,
            "sandbox_invocations": sandbox_invocations,
        },
    }
    if rewritten_question is not None:
        rec["rewritten_question"] = rewritten_question
    return rec


def run(
    subset: str = "dev",
    record_slice: slice | None = None,
    *,
    overwrite: bool = False,
    verbose: bool = False,
    use_rewrite: bool = True,
) -> None:
    ds = DatasetDict(dataset_fpath)
    sl = record_slice if record_slice is not None else DEFAULT_DEV_SLICE
    records = ds.get_subset(subset).get_records()[sl]

    out_dir = results_v2_dir
    os.makedirs(out_dir, exist_ok=True)

    for rec in tqdm(records, desc="src_v2"):
        out_path = os.path.join(out_dir, f"{rec.file_id}.json")
        if os.path.isfile(out_path) and not overwrite:
            print(f"Skipping {rec.file_id}, exists: {out_path}")
            continue

        raw = record_to_raw_data(rec)
        prev: list[str] = []
        chunks: list[str] = []

        for q in rec.dialogue.conv_questions:
            t0 = time.perf_counter()
            steps_acc: list[dict[str, Any]] = []
            llm_sum = 0.0
            n_llm = 0
            n_sandbox = 0
            rewritten: str | None = None
            mode = "rewrite" if use_rewrite else "react"

            if use_rewrite:
                rw = rewrite_current_question(prev, q, verbose=verbose)
                steps_acc.extend(rw["steps"])
                llm_sum += float(rw["llm_ms_total"])
                if not rw.get("skipped"):
                    n_llm += 1
                rewritten = rw["rewritten_question"]
                user_message = build_rewritten_answer_user_message(rewritten)
                answer_style = "rewritten_only"
            else:
                user_message = build_vanilla_user_message(prev, q)
                answer_style = "history_json"

            out = run_react_turn(
                raw,
                user_message,
                verbose=verbose,
                answer_style=answer_style,
            )
            llm_sum += float(out["llm_ms_total"])
            n_llm += int(out.get("llm_invocations", 0))
            n_sandbox += int(out.get("sandbox_invocations", 0))

            steps_acc.extend(list(out["steps"]))
            wall_ms = (time.perf_counter() - t0) * 1000.0

            answer_text = out["answer_text"]
            turn_obj = _turn_record(
                question_history=list(prev),
                question=q,
                final_answer=_final_answer_json_value(answer_text),
                latency_ms=wall_ms,
                steps=steps_acc,
                llm_ms_total=llm_sum,
                mode=mode,
                rewritten_question=rewritten if use_rewrite else None,
                llm_invocations=n_llm,
                sandbox_invocations=n_sandbox,
            )
            chunks.append(json.dumps(turn_obj, ensure_ascii=False, indent=2))
            prev.append(q)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(chunks) + ("\n" if chunks else ""))

        print(f"Wrote {out_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="SRC_V2: ReAct + rewrite by default; writes to data/results_v2/.",
    )
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    rewrite_group = ap.add_mutually_exclusive_group()
    rewrite_group.add_argument(
        "--rewrite",
        dest="use_rewrite",
        action="store_true",
        help="Enable rewrite step (default).",
    )
    rewrite_group.add_argument(
        "--no-rewrite",
        dest="use_rewrite",
        action="store_false",
        help="Disable rewrite step and answer directly from question_history/current_question JSON.",
    )
    ap.set_defaults(use_rewrite=True)
    args = ap.parse_args()
    run(
        record_slice=slice(args.start, args.end),
        overwrite=args.overwrite,
        verbose=args.verbose,
        use_rewrite=args.use_rewrite,
    )
