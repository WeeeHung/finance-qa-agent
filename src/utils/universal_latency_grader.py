"""Grade `latency_ms` from pipeline result files.

Supports:
- Single-directory stats (min/max/median/q1/q3) for one run.
- Two-directory comparison (baseline vs candidate) for old-vs-new timings.
- Per-turn and aggregate ``reason_pass`` (LLM steps: rewrite + ReAct) and
  ``sandbox_invocations`` when ``metadata`` is present (e.g. src_v2 / src_v3).

This grader never evaluates answer quality or correctness.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, List, Optional

from src.bootstrap_env import load_project_env

load_project_env()

from src.utils.filepaths import results_v1_rewrite_dir, results_v2_dir

DEFAULT_GRADES_NAME = "universal_latency_grades.csv"
DEFAULT_SUMMARY_NAME = "universal_latency_summary.txt"


def load_results_json(path: Path) -> List[dict]:
    raw = path.read_text(encoding="utf-8")
    json_str = "[" + re.sub(r"}\s*{", "},{", raw) + "]"
    return json.loads(json_str)


def _coerce_optional_count(value: Any) -> Optional[int]:
    """Non-negative integer counts from JSON metadata (reason_pass, sandbox_invocations)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if value < 0:
            return None
        return int(round(value))
    if isinstance(value, str):
        try:
            n = int(value.strip())
            return n if n >= 0 else None
        except ValueError:
            return None
    return None


def _turn_reason_pass(turn: dict[str, Any]) -> Optional[int]:
    """LLM steps for the turn (rewrite + ReAct); v2/v3 store as reason_pass or llm_invocations."""
    meta = turn.get("metadata")
    if not isinstance(meta, dict):
        return None
    rp = _coerce_optional_count(meta.get("reason_pass"))
    if rp is not None:
        return rp
    return _coerce_optional_count(meta.get("llm_invocations"))


def _turn_sandbox_invocations(turn: dict[str, Any]) -> Optional[int]:
    meta = turn.get("metadata")
    if not isinstance(meta, dict):
        return None
    return _coerce_optional_count(meta.get("sandbox_invocations"))


def _coerce_latency_ms(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        latency = float(value)
    elif isinstance(value, str):
        try:
            latency = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(latency):
        return None
    return latency


def _fmt(value: Optional[float], ndigits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{ndigits}f}"


def _safe_mean(values: list[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _safe_median(values: list[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _safe_p95(values: list[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    # Inclusive quantiles gives a stable P95 without external deps.
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def _safe_min(values: list[float]) -> Optional[float]:
    return min(values) if values else None


def _safe_max(values: list[float]) -> Optional[float]:
    return max(values) if values else None


def _safe_q1(values: list[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=4, method="inclusive")[0]


def _safe_q3(values: list[float]) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=4, method="inclusive")[2]


def run_compare_grade(
    baseline_dir: Path,
    candidate_dir: Path,
    out_csv: Path,
    out_summary: Path,
) -> dict[str, int]:
    baseline_files = {p.name: p for p in baseline_dir.glob("*.json")}
    candidate_files = {p.name: p for p in candidate_dir.glob("*.json")}
    all_filenames = sorted(set(baseline_files) | set(candidate_files))

    rows: list[dict[str, Any]] = []
    baseline_latencies: list[float] = []
    candidate_latencies: list[float] = []
    deltas_ms: list[float] = []
    speedups: list[float] = []
    baseline_reason_pass: list[int] = []
    candidate_reason_pass: list[int] = []
    baseline_sandbox: list[int] = []
    candidate_sandbox: list[int] = []

    stats = {
        "files_total": len(all_filenames),
        "files_missing_baseline": 0,
        "files_missing_candidate": 0,
        "turns_total": 0,
        "turns_missing_baseline": 0,
        "turns_missing_candidate": 0,
        "turns_missing_latency": 0,
        "turns_compared": 0,
        "turns_candidate_faster": 0,
        "turns_candidate_slower": 0,
        "turns_candidate_equal": 0,
    }

    for fname in all_filenames:
        baseline_path = baseline_files.get(fname)
        candidate_path = candidate_files.get(fname)

        if baseline_path is None:
            stats["files_missing_baseline"] += 1
        if candidate_path is None:
            stats["files_missing_candidate"] += 1

        baseline_turns = load_results_json(baseline_path) if baseline_path else []
        candidate_turns = load_results_json(candidate_path) if candidate_path else []
        n_turns = max(len(baseline_turns), len(candidate_turns))

        for turn_idx in range(n_turns):
            stats["turns_total"] += 1
            baseline_turn = baseline_turns[turn_idx] if turn_idx < len(baseline_turns) else None
            candidate_turn = candidate_turns[turn_idx] if turn_idx < len(candidate_turns) else None

            baseline_latency = _coerce_latency_ms(
                baseline_turn.get("latency_ms") if baseline_turn else None
            )
            candidate_latency = _coerce_latency_ms(
                candidate_turn.get("latency_ms") if candidate_turn else None
            )
            brp = _turn_reason_pass(baseline_turn) if baseline_turn else None
            crp = _turn_reason_pass(candidate_turn) if candidate_turn else None
            bsb = _turn_sandbox_invocations(baseline_turn) if baseline_turn else None
            csb = _turn_sandbox_invocations(candidate_turn) if candidate_turn else None
            question = (candidate_turn or baseline_turn or {}).get("question", "")

            status = "compared"
            if baseline_turn is None:
                status = "missing_baseline_turn"
                stats["turns_missing_baseline"] += 1
            elif candidate_turn is None:
                status = "missing_candidate_turn"
                stats["turns_missing_candidate"] += 1
            elif baseline_latency is None or candidate_latency is None:
                status = "missing_latency"
                stats["turns_missing_latency"] += 1
            else:
                delta = baseline_latency - candidate_latency
                speedup = baseline_latency / candidate_latency if candidate_latency > 0 else None
                faster = candidate_latency < baseline_latency
                slower = candidate_latency > baseline_latency

                baseline_latencies.append(baseline_latency)
                candidate_latencies.append(candidate_latency)
                deltas_ms.append(delta)
                if speedup is not None and math.isfinite(speedup):
                    speedups.append(speedup)
                stats["turns_compared"] += 1
                if faster:
                    stats["turns_candidate_faster"] += 1
                elif slower:
                    stats["turns_candidate_slower"] += 1
                else:
                    stats["turns_candidate_equal"] += 1
                if brp is not None:
                    baseline_reason_pass.append(brp)
                if crp is not None:
                    candidate_reason_pass.append(crp)
                if bsb is not None:
                    baseline_sandbox.append(bsb)
                if csb is not None:
                    candidate_sandbox.append(csb)

            if status != "compared":
                delta = None
                speedup = None
                faster = None

            rows.append(
                {
                    "record_file": fname,
                    "turn_idx": turn_idx,
                    "question": question,
                    "baseline_latency_ms": _fmt(baseline_latency),
                    "candidate_latency_ms": _fmt(candidate_latency),
                    "delta_ms_baseline_minus_candidate": _fmt(delta),
                    "speedup_ratio_baseline_over_candidate": _fmt(speedup, ndigits=4),
                    "candidate_faster": faster if faster is not None else "",
                    "baseline_reason_pass": "" if brp is None else str(brp),
                    "candidate_reason_pass": "" if crp is None else str(crp),
                    "baseline_sandbox_invocations": "" if bsb is None else str(bsb),
                    "candidate_sandbox_invocations": "" if csb is None else str(csb),
                    "status": status,
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_file",
        "turn_idx",
        "question",
        "baseline_latency_ms",
        "candidate_latency_ms",
        "delta_ms_baseline_minus_candidate",
        "speedup_ratio_baseline_over_candidate",
        "candidate_faster",
        "baseline_reason_pass",
        "candidate_reason_pass",
        "baseline_sandbox_invocations",
        "candidate_sandbox_invocations",
        "status",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    baseline_mean = _safe_mean(baseline_latencies)
    candidate_mean = _safe_mean(candidate_latencies)
    delta_mean = _safe_mean(deltas_ms)
    speedup_mean = _safe_mean(speedups)
    baseline_median = _safe_median(baseline_latencies)
    candidate_median = _safe_median(candidate_latencies)
    baseline_p95 = _safe_p95(baseline_latencies)
    candidate_p95 = _safe_p95(candidate_latencies)

    brp_f = [float(x) for x in baseline_reason_pass]
    crp_f = [float(x) for x in candidate_reason_pass]
    bsb_f = [float(x) for x in baseline_sandbox]
    csb_f = [float(x) for x in candidate_sandbox]

    summary_lines = [
        f"files_total: {stats['files_total']}",
        f"files_missing_baseline: {stats['files_missing_baseline']}",
        f"files_missing_candidate: {stats['files_missing_candidate']}",
        "",
        f"turns_total: {stats['turns_total']}",
        f"turns_compared: {stats['turns_compared']}",
        f"turns_missing_baseline: {stats['turns_missing_baseline']}",
        f"turns_missing_candidate: {stats['turns_missing_candidate']}",
        f"turns_missing_latency: {stats['turns_missing_latency']}",
        "",
        f"turns_candidate_faster: {stats['turns_candidate_faster']}",
        f"turns_candidate_slower: {stats['turns_candidate_slower']}",
        f"turns_candidate_equal: {stats['turns_candidate_equal']}",
        "",
        f"baseline_mean_ms: {_fmt(baseline_mean)}",
        f"candidate_mean_ms: {_fmt(candidate_mean)}",
        f"mean_delta_ms_baseline_minus_candidate: {_fmt(delta_mean)}",
        f"mean_speedup_ratio_baseline_over_candidate: {_fmt(speedup_mean, ndigits=4)}",
        "",
        f"baseline_median_ms: {_fmt(baseline_median)}",
        f"candidate_median_ms: {_fmt(candidate_median)}",
        f"baseline_p95_ms: {_fmt(baseline_p95)}",
        f"candidate_p95_ms: {_fmt(candidate_p95)}",
        "",
        "# reason_pass = metadata.reason_pass or llm_invocations (rewrite + ReAct LLM calls)",
        f"turns_with_baseline_reason_pass: {len(baseline_reason_pass)}",
        f"turns_with_candidate_reason_pass: {len(candidate_reason_pass)}",
        f"baseline_mean_reason_pass: {_fmt(_safe_mean(brp_f), ndigits=4)}",
        f"baseline_median_reason_pass: {_fmt(_safe_median(brp_f), ndigits=4)}",
        f"candidate_mean_reason_pass: {_fmt(_safe_mean(crp_f), ndigits=4)}",
        f"candidate_median_reason_pass: {_fmt(_safe_median(crp_f), ndigits=4)}",
        "",
        f"turns_with_baseline_sandbox_invocations: {len(baseline_sandbox)}",
        f"turns_with_candidate_sandbox_invocations: {len(candidate_sandbox)}",
        f"baseline_mean_sandbox_invocations: {_fmt(_safe_mean(bsb_f), ndigits=4)}",
        f"baseline_median_sandbox_invocations: {_fmt(_safe_median(bsb_f), ndigits=4)}",
        f"candidate_mean_sandbox_invocations: {_fmt(_safe_mean(csb_f), ndigits=4)}",
        f"candidate_median_sandbox_invocations: {_fmt(_safe_median(csb_f), ndigits=4)}",
    ]
    out_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Compared turns: {stats['turns_compared']}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_summary}")
    return stats


def run_single_grade(
    results_dir: Path,
    out_csv: Path,
    out_summary: Path,
) -> dict[str, int]:
    files = sorted(results_dir.glob("*.json"))
    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    reason_passes: list[int] = []
    sandboxes: list[int] = []
    stats = {
        "files_total": len(files),
        "turns_total": 0,
        "turns_with_latency": 0,
        "turns_missing_latency": 0,
        "turns_with_reason_pass": 0,
        "turns_with_sandbox_invocations": 0,
    }

    for path in files:
        turns = load_results_json(path)
        for turn_idx, turn in enumerate(turns):
            stats["turns_total"] += 1
            latency = _coerce_latency_ms(turn.get("latency_ms"))
            question = turn.get("question", "")
            rp = _turn_reason_pass(turn)
            sb = _turn_sandbox_invocations(turn)
            status = "ok"
            if latency is None:
                status = "missing_latency"
                stats["turns_missing_latency"] += 1
            else:
                latencies.append(latency)
                stats["turns_with_latency"] += 1
            if rp is not None:
                reason_passes.append(rp)
                stats["turns_with_reason_pass"] += 1
            if sb is not None:
                sandboxes.append(sb)
                stats["turns_with_sandbox_invocations"] += 1

            rows.append(
                {
                    "record_file": path.name,
                    "turn_idx": turn_idx,
                    "question": question,
                    "latency_ms": _fmt(latency),
                    "reason_pass": "" if rp is None else str(rp),
                    "sandbox_invocations": "" if sb is None else str(sb),
                    "status": status,
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_file",
        "turn_idx",
        "latency_ms",
        "reason_pass",
        "sandbox_invocations",
        "question",
        "status",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    rp_f = [float(x) for x in reason_passes]
    sb_f = [float(x) for x in sandboxes]

    summary_lines = [
        f"files_total: {stats['files_total']}",
        f"turns_total: {stats['turns_total']}",
        f"turns_with_latency: {stats['turns_with_latency']}",
        f"turns_missing_latency: {stats['turns_missing_latency']}",
        "",
        f"min_ms: {_fmt(_safe_min(latencies))}",
        f"max_ms: {_fmt(_safe_max(latencies))}",
        f"median_ms: {_fmt(_safe_median(latencies))}",
        f"q1_ms: {_fmt(_safe_q1(latencies))}",
        f"q3_ms: {_fmt(_safe_q3(latencies))}",
        "",
        "# reason_pass = metadata.reason_pass or llm_invocations",
        f"turns_with_reason_pass: {stats['turns_with_reason_pass']}",
        f"mean_reason_pass: {_fmt(_safe_mean(rp_f), ndigits=4)}",
        f"median_reason_pass: {_fmt(_safe_median(rp_f), ndigits=4)}",
        "",
        f"turns_with_sandbox_invocations: {stats['turns_with_sandbox_invocations']}",
        f"mean_sandbox_invocations: {_fmt(_safe_mean(sb_f), ndigits=4)}",
        f"median_sandbox_invocations: {_fmt(_safe_median(sb_f), ndigits=4)}",
    ]
    out_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Turns with latency: {stats['turns_with_latency']}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_summary}")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Universal latency grader.")
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Single-directory mode. If set, compute one-run latency stats "
            "(min/max/median/q1/q3) from this directory only."
        ),
    )
    p.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path(results_v1_rewrite_dir),
        help="Baseline results directory (typically old pipeline).",
    )
    p.add_argument(
        "--candidate-dir",
        type=Path,
        default=Path(results_v2_dir),
        help="Candidate results directory (typically new pipeline).",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help=(
            "Output CSV path (default: single mode -> <results-dir>/..., "
            "compare mode -> <candidate-dir>/...)"
        ),
    )
    p.add_argument(
        "--out-summary",
        type=Path,
        default=None,
        help=(
            "Output summary path (default: single mode -> <results-dir>/..., "
            "compare mode -> <candidate-dir>/...)"
        ),
    )
    args = p.parse_args()

    if args.results_dir is not None:
        results_dir = args.results_dir.expanduser().resolve()
        out_csv = (
            args.out_csv.expanduser().resolve()
            if args.out_csv is not None
            else results_dir / DEFAULT_GRADES_NAME
        )
        out_summary = (
            args.out_summary.expanduser().resolve()
            if args.out_summary is not None
            else results_dir / DEFAULT_SUMMARY_NAME
        )
        run_single_grade(results_dir=results_dir, out_csv=out_csv, out_summary=out_summary)
    else:
        baseline_dir = args.baseline_dir.expanduser().resolve()
        candidate_dir = args.candidate_dir.expanduser().resolve()
        out_csv = (
            args.out_csv.expanduser().resolve()
            if args.out_csv is not None
            else candidate_dir / DEFAULT_GRADES_NAME
        )
        out_summary = (
            args.out_summary.expanduser().resolve()
            if args.out_summary is not None
            else candidate_dir / DEFAULT_SUMMARY_NAME
        )
        run_compare_grade(
            baseline_dir=baseline_dir,
            candidate_dir=candidate_dir,
            out_csv=out_csv,
            out_summary=out_summary,
        )


if __name__ == "__main__":
    main()
