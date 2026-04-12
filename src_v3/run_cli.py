"""Interactive CLI for src_v3: KB + rewrite + ReAct; per-turn KB append like batch ``runme``.

If ``data/results_v3/kb/<file_id>.kb.json`` already exists, initial KB curation (``extract_initial_kb``)
is skipped and the KB is loaded from disk.

Turn logs under ``data/results_v3/<file_id>.json`` use the same pretty JSON blocks as ``runme``; new
turns are **appended** when the file exists (nothing is truncated). The KB file is overwritten after
each answered turn with the full in-memory KB (same as batch end-of-doc save).

``--no-persist`` disables all disk writes. Verbose logging stays on unless ``--quiet``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Union

from src.bootstrap_env import load_project_env

load_project_env()

from src.utils.data.read_dataset import DatasetDict
from src.utils.data.types import ConvFinQARecord
from src.utils.filepaths import dataset_fpath, kb_v3_dir, results_v3_dir
from src_v1.prompt import build_rewritten_answer_user_message, build_vanilla_user_message
from src_v1.rewrite import rewrite_current_question
from src_v1.serialize import record_to_raw_data
from src_v3.kb_extract import empty_turn_kb_updates, extract_initial_kb, extract_turn_kb_updates
from src_v3.kb_store import KnowledgeBase
from src_v3.react_turn import TOOL_NAME, run_react_turn

_MAX_CODE_DISPLAY_CHARS = 12_000


def _stdout_color_ok() -> bool:
    if os.environ.get("NO_COLOR", "").strip():
        return False
    return sys.stdout.isatty()


def _log_execute_python_from_steps(steps: list[dict[str, Any]], *, use_color: bool) -> None:
    reset = "\033[0m"
    bold = "\033[1m"
    dim = "\033[2m"
    hdr = "\033[1;35m"
    code_fg = "\033[96m"
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


def _parse_existing_results_turns(text: str) -> list[dict[str, Any]]:
    """Decode concatenated pretty-printed JSON objects (same layout as ``runme`` output)."""
    text = (text or "").strip()
    if not text:
        return []
    dec = json.JSONDecoder()
    out: list[dict[str, Any]] = []
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = dec.raw_decode(text, idx)
        out.append(obj)
        idx = end
    return out


def _append_results_turn_block(path: str, turn_obj: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    block = json.dumps(turn_obj, ensure_ascii=False, indent=2)
    exists_nonempty = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", encoding="utf-8") as f:
        if exists_nonempty:
            f.write("\n")
        f.write(block + "\n")


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


def _print_help() -> None:
    print(
        "Commands: help | quit | reset | record | kb | scripted | gold [i] | use <id-or-file_id>\n"
        "Otherwise type a question (same messaging as batch src_v3).\n"
        "With default --persist, turns append to data/results_v3/<file_id>.json; KB loads/saves under kb/."
    )


def _run_one_turn(
    kb: KnowledgeBase,
    prev: list[str],
    question: str,
    *,
    verbose: bool,
    use_rewrite: bool,
) -> tuple[
    str,
    list[dict[str, Any]],
    float,
    float,
    int,
    int,
    int,
    int,
    dict[str, Any],
    str | None,
]:
    """Returns answer, steps (rewrite+react only, matching ``runme``), QA wall ms, llm_sum, n_llm, n_sb, kb sizes, kb_update, rewritten."""
    steps_acc: list[dict[str, Any]] = []
    llm_sum = 0.0
    n_llm = 0
    rewritten: str | None = None

    qa_t0 = time.perf_counter()
    if use_rewrite:
        rw = rewrite_current_question(prev, question, verbose=verbose)
        steps_acc.extend(rw["steps"])
        llm_sum += float(rw["llm_ms_total"])
        if not rw.get("skipped"):
            n_llm += 1
        rewritten = rw["rewritten_question"]
        user_message = build_rewritten_answer_user_message(rewritten)
        answer_style = "rewritten_only"
        if verbose and rewritten:
            print(f"[rewrite] {rewritten!r}", flush=True)
    else:
        user_message = build_vanilla_user_message(prev, question)
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
    n_sb = int(out.get("sandbox_invocations", 0))
    react_steps = list(out["steps"])
    steps_acc.extend(react_steps)
    if verbose:
        _log_execute_python_from_steps(react_steps, use_color=_stdout_color_ok())

    answer_text = out["answer_text"]
    qa_wall_ms = (time.perf_counter() - qa_t0) * 1000.0

    if n_sb == 0:
        kb_upd = empty_turn_kb_updates()
    else:
        kb_upd = extract_turn_kb_updates(
            kb=kb,
            question_history=list(prev),
            question=question,
            rewritten_question=rewritten if use_rewrite else None,
            final_answer=answer_text,
        )
    kb_added = kb.append_drafts(kb_upd["items"])
    kb_size_after = len(kb.items)

    if verbose:
        if n_sb == 0:
            print(
                f"[src_v3] KB append skipped (no {TOOL_NAME}); "
                f"{kb_size_before} items unchanged",
                flush=True,
            )
        else:
            print(
                f"[src_v3] KB append: +{len(kb_added)} items "
                f"({kb_size_before} → {kb_size_after}), "
                f"{float(kb_upd['llm_ms_total']):.0f} ms",
                flush=True,
            )

    kb_update = {
        "candidate_items": len(kb_upd["items"]),
        "added_items": len(kb_added),
        "added_ids": [it.id for it in kb_added],
        "llm_ms_total": round(float(kb_upd["llm_ms_total"]), 3),
    }
    return (
        answer_text,
        steps_acc,
        qa_wall_ms,
        llm_sum,
        n_llm,
        n_sb,
        kb_size_before,
        kb_size_after,
        kb_update,
        rewritten if use_rewrite else None,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Interactive src_v3 ReAct-over-KB CLI. Loads existing KB/results when present; appends turns by default.",
    )
    ap.add_argument("--subset", choices=("dev", "train"), default="dev")
    ap.add_argument("--id", metavar="ID", default=None, help="Record id or file_id to load.")
    ap.add_argument(
        "--rewrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rewrite history+current into one question before answering (default: on).",
    )
    ap.add_argument("--quiet", action="store_true", help="Disable verbose / debug prints.")
    ap.add_argument(
        "--persist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append runme-style turn JSON to results file and save KB after each answer (default: on).",
    )
    ap.add_argument(
        "--out",
        metavar="PATH",
        default=None,
        help="Override results JSON path (append). KB still saves under data/results_v3/kb/ when persisting.",
    )
    ap.add_argument("--question", metavar="Q", default=None, help="Single question then exit (non-interactive).")
    args = ap.parse_args(argv)

    if args.question and not args.id:
        print("Non-interactive --question requires --id", file=sys.stderr)
        return 2

    verbose = not args.quiet
    persist = args.persist

    ds = DatasetDict(dataset_fpath)
    records = ds.get_subset(args.subset).get_records()

    rec: ConvFinQARecord | None = None
    kb: KnowledgeBase | None = None
    raw_data: dict[str, Any] | None = None
    prev: list[str] = []
    kb_path: str | None = None
    results_path: str | None = None
    kb_init_meta: dict[str, Any] | None = None
    first_kb_meta_remaining = False

    if args.id:
        rec = _find_record(ds, args.subset, args.id)

    def bind_record(r: ConvFinQARecord) -> None:
        nonlocal rec, prev, kb, raw_data, kb_path, results_path, kb_init_meta, first_kb_meta_remaining
        rec = r
        raw_data = record_to_raw_data(r)
        kb_path = os.path.join(kb_v3_dir, f"{r.file_id}.kb.json")
        results_path = args.out or os.path.join(results_v3_dir, f"{r.file_id}.json")
        kb_init_meta = None
        first_kb_meta_remaining = False

        if os.path.isfile(kb_path):
            kb = KnowledgeBase.load_json(kb_path)
            print(f"Using record id={rec.id!r} file_id={rec.file_id!r} (loaded KB: {len(kb.items)} items)", flush=True)
            if verbose:
                print(f"[src_v3] skipped initial KB curation — using {kb_path!r}", flush=True)
        else:
            kb = KnowledgeBase(file_id=r.file_id)
            t0 = time.perf_counter()
            kb_init = extract_initial_kb(raw_data)
            init_added = kb.append_drafts(kb_init["items"])
            init_ms = (time.perf_counter() - t0) * 1000.0
            kb_init_meta = {
                "kb_initial_items": len(init_added),
                "kb_initial_llm_ms_total": round(float(kb_init["llm_ms_total"]), 3),
            }
            first_kb_meta_remaining = True
            print(f"Using record id={rec.id!r} file_id={rec.file_id!r}", flush=True)
            if verbose:
                print(
                    f"[src_v3] initial KB: {len(init_added)} items accepted "
                    f"(drafts={len(kb_init['items'])}), {init_ms:.0f} ms wall",
                    flush=True,
                )

        if os.path.isfile(results_path):
            with open(results_path, encoding="utf-8") as f:
                existing = _parse_existing_results_turns(f.read())
            prev = [str(t.get("question", "")) for t in existing]
            if verbose:
                print(
                    f"[src_v3] restored {len(prev)} prior turn(s) from {results_path!r} "
                    f"(question_history for rewrite)",
                    flush=True,
                )
        else:
            prev = []
            if verbose and persist:
                print(f"[src_v3] no existing results at {results_path!r} — new turns will create/append", flush=True)

        if not persist:
            first_kb_meta_remaining = False

    def persist_answered_turn(
        *,
        question: str,
        answer_text: str,
        steps_acc: list[dict[str, Any]],
        qa_wall_ms: float,
        llm_sum: float,
        n_llm: int,
        n_sb: int,
        kb_b: int,
        kb_a: int,
        kb_update: dict[str, Any],
        rewritten_question: str | None,
    ) -> None:
        nonlocal first_kb_meta_remaining
        if not persist or kb is None or kb_path is None or results_path is None:
            return
        meta_extra = kb_init_meta if first_kb_meta_remaining and kb_init_meta else None
        if first_kb_meta_remaining:
            first_kb_meta_remaining = False
        mode = "rewrite" if args.rewrite else "react"
        turn_obj = _turn_record(
            question_history=list(prev),
            question=question,
            final_answer=_final_answer_json_value(answer_text),
            latency_ms=qa_wall_ms,
            steps=steps_acc,
            llm_ms_total=llm_sum,
            mode=mode,
            rewritten_question=rewritten_question,
            llm_invocations=n_llm,
            sandbox_invocations=n_sb,
            kb_size_before=kb_b,
            kb_size_after=kb_a,
            kb_update=kb_update,
            metadata_extra=meta_extra,
        )
        _append_results_turn_block(results_path, turn_obj)
        kb.save_json(kb_path)
        if verbose:
            print(f"[src_v3] persisted turn → {results_path!r} ; KB → {kb_path!r}", flush=True)

    if rec is None:
        print(f"Subset {args.subset!r}: {len(records)} records. Commands: use <id-or-file_id> | list | quit")
    else:
        bind_record(rec)

    if args.question:
        assert rec is not None and kb is not None and raw_data is not None
        (
            ans,
            steps,
            qa_wall_ms,
            llm_sum,
            n_llm,
            n_sb,
            kb_b,
            kb_a,
            kb_upd,
            rw_q,
        ) = _run_one_turn(
            kb,
            prev,
            args.question,
            verbose=verbose,
            use_rewrite=args.rewrite,
        )
        print(ans, flush=True)
        persist_answered_turn(
            question=args.question,
            answer_text=ans,
            steps_acc=steps,
            qa_wall_ms=qa_wall_ms,
            llm_sum=llm_sum,
            n_llm=n_llm,
            n_sb=n_sb,
            kb_b=kb_b,
            kb_a=kb_a,
            kb_update=kb_upd,
            rewritten_question=rw_q,
        )
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
            if rec is None or kb is None or raw_data is None:
                print("Pick a record first: use <id-or-file_id> or use <index>", file=sys.stderr)
                continue
            if low == "reset":
                prev.clear()
                print(
                    "Cleared in-memory question_history only. "
                    "Reload record (use …) to re-read prior turns from the results file.",
                    flush=True,
                )
                continue
            if low == "record":
                print(json.dumps({"id": rec.id, "file_id": rec.file_id}, indent=2))
                continue
            if low == "kb":
                ctx = kb.to_context()
                if len(ctx) > 8000:
                    print(ctx[:8000] + "\n...(truncated)", flush=True)
                else:
                    print(ctx, flush=True)
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

            (
                ans,
                steps,
                qa_wall_ms,
                llm_sum,
                n_llm,
                n_sb,
                kb_b,
                kb_a,
                kb_upd,
                rw_q,
            ) = _run_one_turn(
                kb,
                prev,
                line,
                verbose=verbose,
                use_rewrite=args.rewrite,
            )
            print(
                f"answer: {ans!r} ({qa_wall_ms:.0f} ms QA wall, llm_calls={n_llm}, tools={n_sb}, "
                f"KB {kb_b}→{kb_a})",
                flush=True,
            )
            persist_answered_turn(
                question=line,
                answer_text=ans,
                steps_acc=steps,
                qa_wall_ms=qa_wall_ms,
                llm_sum=llm_sum,
                n_llm=n_llm,
                n_sb=n_sb,
                kb_b=kb_b,
                kb_a=kb_a,
                kb_update=kb_upd,
                rewritten_question=rw_q,
            )
            prev.append(line)
    except KeyboardInterrupt:
        print("\n(interrupted)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
