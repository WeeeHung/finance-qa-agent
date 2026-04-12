"""Interactive CLI for src_v2: pick a ConvFinQA record, ask questions (ReAct) in a loop.

Defaults match a dev workflow: verbose LLM logging on, rewrite on (same as batch ``runme``),
and ``--out`` overwrites existing files unless ``--no-overwrite`` is passed.
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
from src_v2.react_turn import TOOL_NAME, run_react_turn

_MAX_CODE_DISPLAY_CHARS = 12_000


def _stdout_color_ok() -> bool:
    if os.environ.get("NO_COLOR", "").strip():
        return False
    return sys.stdout.isatty()


def _log_execute_python_from_steps(steps: list[dict[str, Any]], *, use_color: bool) -> None:
    """Print each sandbox script from ReAct steps (assistant tool_calls), highlighted for the CLI."""
    reset = "\033[0m"
    bold = "\033[1m"
    dim = "\033[2m"
    hdr = "\033[1;35m"  # bright magenta
    code_fg = "\033[96m"  # bright cyan
    n = 0
    for step in steps:
        if step.get("type") != "assistant":
            continue
        for tc in step.get("tool_calls") or []:
            if not isinstance(tc, dict) or (tc.get("name") or "") != TOOL_NAME:
                continue
            args = tc.get("args")
            if isinstance(args, dict):
                raw = args.get("code")
                code = raw if isinstance(raw, str) else (str(raw) if raw is not None else "")
            else:
                code = (str(args) if args is not None else "") or ""
            code = code.strip()
            if not code:
                continue
            n += 1
            body = (
                code
                if len(code) <= _MAX_CODE_DISPLAY_CHARS
                else code[:_MAX_CODE_DISPLAY_CHARS] + "\n...(truncated for display)"
            )
            if use_color:
                print(
                    f"{bold}{hdr}── {TOOL_NAME} ({n}) ──{reset}\n{code_fg}{body}{reset}",
                    flush=True,
                )
                print(f"{dim}(end {TOOL_NAME} {n}){reset}", flush=True)
            else:
                print(f"--- {TOOL_NAME} ({n}) ---\n{body}", flush=True)
                print(f"(end {TOOL_NAME} {n})", flush=True)


def _find_record(ds: DatasetDict, subset: str, id_or_file_id: str) -> ConvFinQARecord:
    for r in ds.get_subset(subset).get_records():
        if r.id == id_or_file_id or r.file_id == id_or_file_id:
            return r
    raise KeyError(f"No record matching {id_or_file_id!r} in subset {subset!r}")


def _print_help() -> None:
    print(
        "Commands: help | quit | reset | record | scripted | gold [i] | use <id-or-file_id>\n"
        "Otherwise type a question (same messaging as batch src_v2)."
    )


def _run_one_turn(
    raw: dict[str, Any],
    prev: list[str],
    question: str,
    *,
    verbose: bool,
    use_rewrite: bool,
) -> tuple[str, list[dict[str, Any]], float, int, int]:
    t0 = time.perf_counter()
    steps_acc: list[dict[str, Any]] = []

    if use_rewrite:
        rw = rewrite_current_question(prev, question, verbose=verbose)
        steps_acc.extend(rw["steps"])
        user_message = build_rewritten_answer_user_message(rw["rewritten_question"])
        answer_style = "rewritten_only"
        if verbose and rw.get("rewritten_question"):
            print(f"[rewrite] {rw['rewritten_question']!r}", flush=True)
    else:
        user_message = build_vanilla_user_message(prev, question)
        answer_style = "history_json"

    out = run_react_turn(
        raw,
        user_message,
        verbose=verbose,
        answer_style=answer_style,
    )
    react_steps = list(out["steps"])
    steps_acc.extend(react_steps)
    if verbose:
        _log_execute_python_from_steps(react_steps, use_color=_stdout_color_ok())
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return (
        out["answer_text"],
        steps_acc,
        wall_ms,
        int(out.get("llm_invocations", 0)),
        int(out.get("sandbox_invocations", 0)),
    )


def _append_out(
    fh: TextIO | None,
    payload: dict[str, Any],
) -> None:
    if fh is None:
        return
    fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    fh.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Interactive src_v2 ReAct CLI (debug + overwrite on by default).")
    ap.add_argument("--subset", choices=("dev", "train"), default="dev")
    ap.add_argument("--id", metavar="ID", default=None, help="Record id or file_id to load.")
    ap.add_argument(
        "--rewrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rewrite history+current into one question before answering (default: on).",
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
        ans, steps, wall_ms, n_llm, n_sb = _run_one_turn(
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
                "llm_invocations": n_llm,
                "sandbox_invocations": n_sb,
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
            ans, steps, wall_ms, n_llm, n_sb = _run_one_turn(
                raw, prev, line, verbose=verbose, use_rewrite=args.rewrite
            )
            print(f"answer: {ans!r} ({wall_ms:.0f} ms wall, llm={n_llm}, tools={n_sb})", flush=True)
            prev.append(line)
            _append_out(
                out_fh,
                {
                    "record_id": rec.id,
                    "question": line,
                    "answer": ans,
                    "latency_ms": round(wall_ms, 3),
                    "llm_invocations": n_llm,
                    "sandbox_invocations": n_sb,
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
