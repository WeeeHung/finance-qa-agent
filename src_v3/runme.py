"""Batch SRC_V3: rewrite + ReAct over KB with per-turn KB append."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Union

from src.bootstrap_env import load_project_env

load_project_env()

from tqdm import tqdm

from src.utils.data.read_dataset import DatasetDict
from src.utils.filepaths import dataset_fpath, kb_v3_dir, results_v3_dir
from src_v1.prompt import build_rewritten_answer_user_message, build_vanilla_user_message
from src_v1.rewrite import rewrite_current_question
from src_v1.serialize import record_to_raw_data
from src_v3.kb_extract import extract_initial_kb, extract_turn_kb_updates
from src_v3.kb_store import KnowledgeBase
from src_v3.react_turn import run_react_turn

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
    kb_size_before: int,
    kb_size_after: int,
    kb_update: dict[str, Any],
    metadata_extra: dict[str, Any] | None = None,
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
            "kb_size_before": kb_size_before,
            "kb_size_after": kb_size_after,
        },
        "kb_update": kb_update,
    }
    if metadata_extra:
        rec["metadata"].update(metadata_extra)
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

    os.makedirs(results_v3_dir, exist_ok=True)
    os.makedirs(kb_v3_dir, exist_ok=True)

    for rec in tqdm(records, desc="src_v3"):
        out_path = os.path.join(results_v3_dir, f"{rec.file_id}.json")
        kb_path = os.path.join(kb_v3_dir, f"{rec.file_id}.kb.json")

        if os.path.isfile(out_path) and not overwrite:
            print(f"Skipping {rec.file_id}, exists: {out_path}")
            continue

        raw = record_to_raw_data(rec)
        prev: list[str] = []
        chunks: list[str] = []

        kb = KnowledgeBase(file_id=rec.file_id)
        kb_init_meta: dict[str, Any] = {"kb_initial_items": 0, "kb_initial_llm_ms_total": 0.0}
        kb_init = extract_initial_kb(raw)
        init_added = kb.append_drafts(kb_init["items"])
        kb_init_meta = {
            "kb_initial_items": len(init_added),
            "kb_initial_llm_ms_total": round(float(kb_init["llm_ms_total"]), 3),
        }

        for idx, q in enumerate(rec.dialogue.conv_questions):
            # Question->answer latency only; excludes KB append/save work after this timer stops.
            qa_t0 = time.perf_counter()

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

            kb_size_before = len(kb.items)
            out = run_react_turn(
                kb.to_context(),
                user_message,
                verbose=verbose,
                answer_style=answer_style,
            )
            llm_sum += float(out["llm_ms_total"])
            n_llm += int(out.get("llm_invocations", 0))
            n_sandbox += int(out.get("sandbox_invocations", 0))
            steps_acc.extend(list(out["steps"]))

            answer_text = out["answer_text"]
            qa_wall_ms = (time.perf_counter() - qa_t0) * 1000.0

            kb_upd = extract_turn_kb_updates(
                kb=kb,
                question_history=list(prev),
                question=q,
                rewritten_question=rewritten,
                final_answer=answer_text,
            )
            kb_added = kb.append_drafts(kb_upd["items"])
            kb_size_after = len(kb.items)

            turn_obj = _turn_record(
                question_history=list(prev),
                question=q,
                final_answer=_final_answer_json_value(answer_text),
                latency_ms=qa_wall_ms,
                steps=steps_acc,
                llm_ms_total=llm_sum,
                mode=mode,
                rewritten_question=rewritten if use_rewrite else None,
                llm_invocations=n_llm,
                sandbox_invocations=n_sandbox,
                kb_size_before=kb_size_before,
                kb_size_after=kb_size_after,
                kb_update={
                    "candidate_items": len(kb_upd["items"]),
                    "added_items": len(kb_added),
                    "added_ids": [it.id for it in kb_added],
                    "llm_ms_total": round(float(kb_upd["llm_ms_total"]), 3),
                },
                metadata_extra=kb_init_meta if idx == 0 else None,
            )
            chunks.append(json.dumps(turn_obj, ensure_ascii=False, indent=2))
            prev.append(q)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(chunks) + ("\n" if chunks else ""))
        kb.save_json(kb_path)

        print(f"Wrote {out_path}")
        print(f"Wrote {kb_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="SRC_V3: ReAct + rewrite over KB with per-turn KB updates.",
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
