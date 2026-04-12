"""Interactive CLI for src_v1: pick a ConvFinQA record, ask questions in a loop.

Defaults match a dev workflow: verbose LLM logging on, and ``--out`` overwrites
existing files unless ``--no-overwrite`` is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, TextIO

from src.bootstrap_env import load_project_env

load_project_env()

from src.utils.data.read_dataset import DatasetDict
from src.utils.data.types import ConvFinQARecord
from src.utils.filepaths import dataset_fpath
from src_v1.prompt import build_rewritten_answer_user_message, build_vanilla_user_message
from src_v1.rewrite import rewrite_current_question
from src_v1.serialize import record_to_raw_data
from src_v1.vanilla import run_vanilla_turn


def _find_record(ds: DatasetDict, subset: str, id_or_file_id: str) -> ConvFinQARecord:
    for r in ds.get_subset(subset).get_records():
        if r.id == id_or_file_id or r.file_id == id_or_file_id:
            return r
    raise KeyError(f"No record matching {id_or_file_id!r} in subset {subset!r}")


def _print_help() -> None:
    print(
        "Commands: help | quit | reset | record | scripted | gold [i] | use <id-or-file_id>\n"
        "Otherwise type a question (vanilla JSON message to the answer model)."
    )


def _run_one_turn(
    raw: dict[str, Any],
    prev: list[str],
    question: str,
    *,
    verbose: bool,
    use_rewrite: bool,
) -> tuple[str, list[dict[str, Any]], float]:
    t0 = time.perf_counter()
    steps_acc: list[dict[str, Any]] = []
    llm_sum = 0.0

    if use_rewrite:
        rw = rewrite_current_question(prev, question, verbose=verbose)
        steps_acc.extend(rw["steps"])
        llm_sum += float(rw["llm_ms_total"])
        user_message = build_rewritten_answer_user_message(rw["rewritten_question"])
        answer_style = "rewritten_only"
        if verbose and rw.get("rewritten_question"):
            print(f"[rewrite] {rw['rewritten_question']!r}", flush=True)
    else:
        user_message = build_vanilla_user_message(prev, question)
        answer_style = "history_json"

    out = run_vanilla_turn(
        raw,
        user_message,
        verbose=verbose,
        answer_style=answer_style,
    )
    llm_sum += float(out["llm_ms_total"])
    steps_acc.extend(list(out["steps"]))
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return out["answer_text"], steps_acc, wall_ms


def _append_out(
    fh: TextIO | None,
    payload: dict[str, Any],
) -> None:
    if fh is None:
        return
    fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    fh.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Interactive src_v1 vanilla CLI (debug + overwrite on by default).")
    ap.add_argument("--subset", choices=("dev", "train"), default="dev")
    ap.add_argument("--id", metavar="ID", default=None, help="Record id or file_id to load.")
    ap.add_argument(
        "--rewrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rewrite history+current into one question before answering (default: off).",
    )
    ap.add_argument("--quiet", action="store_true", help="Disable verbose / debug prints.")
    ap.add_argument("--out", metavar="PATH", default=None, help="Append one JSON line per answered turn.")
    ap.add_argument(
        "--no-overwrite",
        action="store_true",
        help="If --out exists, exit instead of truncating (default is overwrite).",
    )
    ap.add_argument("--question", metavar="Q", default=None, help="Single question then exit (non-interactive).")
    args = ap.parse_args(argv)

    if args.question and not args.id:
        print("Non-interactive --question requires --id", file=sys.stderr)
        return 2

    verbose = not args.quiet
    overwrite = not args.no_overwrite
    out_fh: TextIO | None = None
    if args.out:
        if os.path.isfile(args.out) and not overwrite:
            print(f"Refusing to write: file exists and --no-overwrite: {args.out}", file=sys.stderr)
            return 2
        # Default overwrite: truncate at session start; new file uses "w" as well.
        out_fh = open(args.out, "w", encoding="utf-8")

    ds = DatasetDict(dataset_fpath)
    records = ds.get_subset(args.subset).get_records()

    rec: ConvFinQARecord | None = None
    if args.id:
        rec = _find_record(ds, args.subset, args.id)

    prev: list[str] = []

    def bind_record(r: ConvFinQARecord) -> None:
        nonlocal rec, prev
        rec = r
        prev = []
        print(f"Using record id={rec.id!r} file_id={rec.file_id!r}", flush=True)

    if rec is None:
        print(f"Subset {args.subset!r}: {len(records)} records. Commands: use <id-or-file_id> | list | quit")
    else:
        bind_record(rec)

    if args.question:
        assert rec is not None
        raw = record_to_raw_data(rec)
        ans, steps, wall_ms = _run_one_turn(
            raw, prev, args.question, verbose=verbose, use_rewrite=args.rewrite
        )
        print(ans, flush=True)
        _append_out(
            out_fh,
            {
                "record_id": rec.id,
                "question": args.question,
                "answer": ans,
                "latency_ms": round(wall_ms, 3),
                "steps": steps,
            },
        )
        if out_fh:
            out_fh.close()
        return 0

    try:
        while True:
            if rec is None:
                line = input("cli> ").strip()
            else:
                line = input("q> ").strip()

            if not line:
                continue
            low = line.lower()
            if low in ("quit", "exit", "q"):
                break
            if low == "help":
                _print_help()
                continue
            if low == "list":
                for i, r in enumerate(records[:50]):
                    print(f"  [{i}] {r.file_id}")
                if len(records) > 50:
                    print(f"  ... and {len(records) - 50} more")
                continue
            if low.startswith("use "):
                key = line[4:].strip()
                try:
                    if key.isdigit():
                        bind_record(records[int(key)])
                    else:
                        bind_record(_find_record(ds, args.subset, key))
                except (KeyError, IndexError, ValueError) as e:
                    print(e, file=sys.stderr)
                continue
            if rec is None:
                print("Pick a record first: use <id-or-file_id> or use <index>", file=sys.stderr)
                continue
            if low == "reset":
                prev.clear()
                print("Cleared question_history for this session.", flush=True)
                continue
            if low == "record":
                print(json.dumps({"id": rec.id, "file_id": rec.file_id}, indent=2))
                continue
            if low == "scripted":
                for i, q in enumerate(rec.dialogue.conv_questions):
                    g = rec.dialogue.executed_answers[i] if i < len(rec.dialogue.executed_answers) else None
                    print(f"  [{i}] Q: {q}\n      gold: {g!r}")
                continue
            if low == "gold" or low.startswith("gold "):
                parts = line.split()
                if len(parts) == 1:
                    print(rec.dialogue.executed_answers)
                else:
                    idx = int(parts[1])
                    print(rec.dialogue.executed_answers[idx])
                continue

            raw = record_to_raw_data(rec)
            ans, steps, wall_ms = _run_one_turn(raw, prev, line, verbose=verbose, use_rewrite=args.rewrite)
            print(f"answer: {ans!r} ({wall_ms:.0f} ms wall)", flush=True)
            prev.append(line)
            _append_out(
                out_fh,
                {
                    "record_id": rec.id,
                    "question": line,
                    "answer": ans,
                    "latency_ms": round(wall_ms, 3),
                    "steps": steps,
                },
            )
    except KeyboardInterrupt:
        print("\n(interrupted)", flush=True)
    finally:
        if out_fh:
            out_fh.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
