"""Grade pipeline outputs against golden `executed_answers` from a subset JSON.

Flow per row: deterministic lenient match first, then LLM-as-judge (same idea as
``scoring.compare_answers``). Writes CSV rows and a short summary (#correct / #total).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

from src.bootstrap_env import load_project_env

load_project_env()

from pydantic import BaseModel, Field

from src.utils.filepaths import data_dir, results_dir

SUBSET_DEFAULT = Path(data_dir) / "convfinqa_datasubset.json"
DEFAULT_GRADES_NAME = "universal_accuracy_grades.csv"
DEFAULT_SUMMARY_NAME = "universal_accuracy_summary.txt"

PERCENT_SUFFIX = re.compile(r"%\s*$")
# Operand/operand division (not dates like 12/31: reject only if decimals or both ints are long)
_DIVISION_PAIR = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?![\w/.])")


def looks_like_arithmetic_division_expression(text: str) -> bool:
    """True if text contains a/b style operands that look like unstated math, not a calendar date."""
    for m in _DIVISION_PAIR.finditer(text):
        a, b = m.group(1), m.group(2)
        if "." in a or "." in b:
            return True
        if len(a) >= 3 and len(b) >= 3:
            return True
    return False


class AnswerComparatorResponse(BaseModel):
    correct: bool = Field(
        ...,
        description="True only if the predicted string is an acceptable final answer (typically one scalar), not a formula or work-in-progress.",
    )
    reasoning: str = Field(..., description="The reasoning behind the evaluation")


def _to_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isfinite(x) and abs(x - round(x)) < 1e-9 and abs(x) < 1e12:
            return str(int(round(x)))
        return format(x, ".12g")
    if isinstance(x, int) and not isinstance(x, bool):
        return str(x)
    return str(x).strip()


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def strip_wrapping(s: str) -> str:
    s = s.strip()
    for q in ('"', "'"):
        if len(s) >= 2 and s[0] == q and s[-1] == q:
            s = s[1:-1].strip()
    return s


def parse_scalar_to_float(s: str) -> Optional[float]:
    s = strip_wrapping(s)
    s = s.replace(",", "").replace("$", "").strip()
    if PERCENT_SUFFIX.search(s):
        s = PERCENT_SUFFIX.sub("", s).strip()
    try:
        return float(s)
    except ValueError:
        return None


def coerce_to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        return parse_scalar_to_float(x)
    return None


def floats_equivalent(a: float, b: float, rtol: float = 1e-5, atol: float = 1e-4) -> bool:
    if not math.isfinite(a) or not math.isfinite(b):
        return False
    if math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
        return True
    # decimal ratio vs percentage display (e.g. 0.12574 vs 12.6%, 0.10826 vs 10.826%)
    # Use looser bounds on the *percent* scale: one-decimal rounding can differ by ~0.03–0.05.
    scale_rtol = max(rtol, 2e-3)
    scale_atol = max(atol, 0.06)
    if math.isclose(a * 100, b, rel_tol=scale_rtol, abs_tol=scale_atol):
        return True
    if math.isclose(a, b * 100, rel_tol=scale_rtol, abs_tol=scale_atol):
        return True
    return False


def deterministic_match(gold: Any, pred: Any) -> Tuple[Optional[bool], str]:
    """Return (True/False, reason) if decisive; (None, 'llm') if the LLM should decide."""
    if pred is None:
        return False, "missing_prediction"
    if isinstance(pred, str) and not pred.strip():
        return False, "missing_prediction"

    gold_str = _to_str(gold)
    pred_str = _to_str(pred)

    gn = normalize_whitespace(gold_str)
    pn = normalize_whitespace(pred_str.replace("_", " "))
    if gn == pn:
        return True, "exact_normalized"

    g = coerce_to_float(gold)
    p = coerce_to_float(pred)
    if g is not None and p is not None:
        if floats_equivalent(g, p):
            return True, "numeric_tolerance"
        return False, "numeric_mismatch"

    # Mixed or non-numeric: try parsing string gold / pred once more loosely
    if g is None and isinstance(gold, str):
        g = parse_scalar_to_float(gold)
    if p is None and isinstance(pred, str):
        p = parse_scalar_to_float(pred)
    if g is not None and p is not None:
        if floats_equivalent(g, p):
            return True, "numeric_tolerance"
        return False, "numeric_mismatch"

    if g is not None and isinstance(pred, str):
        if looks_like_arithmetic_division_expression(pred_str):
            return False, "expression_not_scalar"

    return None, "llm"


def load_results_json(path: Path) -> List[dict]:
    raw = path.read_text(encoding="utf-8")
    json_str = "[" + re.sub(r"}\s*{", "},{", raw) + "]"
    return json.loads(json_str)


def record_id_to_results_fname(record_id: str) -> str:
    return record_id.replace("/", "-") + ".json"


def llm_compare(
    comparator: Any,
    gold: Any,
    pred: Any,
) -> Tuple[bool, str]:
    ans = {"gold_answer": gold, "predicted_answer": pred}
    prompt = (
        "You are an expert evaluator. The gold answer is the final ground-truth value.\n"
        "Rules:\n"
        "- If the gold is numeric, the prediction must be a single final scalar value in the response, "
        "not an arithmetic expression (e.g. reject '60.00 / 243.00' or any a/b showing work), "
        "not intermediate steps, and not a sentence that merely implies the number.\n"
        "- If any computation beyond trivial rounding is still required to obtain the gold from the "
        "prediction text, mark incorrect.\n"
        "- If the difference is only percentage vs decimal representation of the same value, it can be correct.\n"
        "- Light formatting (currency label, 'per share') is acceptable only when the numeric value itself "
        "is clearly that single final number; expressions and fractions are not.\n"
        f"Task data: {ans}"
    )
    response = comparator.invoke({"messages": [{"role": "user", "content": prompt}]})
    sr = response["structured_response"]
    return sr.correct, sr.reasoning


def run_grade(
    subset_path: Path,
    results_path: Path,
    out_csv: Path,
    out_summary: Path,
    llm_model: str,
    skip_llm: bool,
) -> Tuple[int, int]:
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    dev: List[dict] = subset["dev"]

    comparator = None
    if not skip_llm:
        from langgraph.prebuilt import create_react_agent

        comparator = create_react_agent(
            model=llm_model,
            response_format=AnswerComparatorResponse,
            tools=[],
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    correct = 0
    total = 0

    for record_idx, rec in enumerate(dev):
        record_id = rec["id"]
        golds = rec["dialogue"]["executed_answers"]
        fname = record_id_to_results_fname(record_id)
        fpath = results_path / fname

        if not fpath.is_file():
            for q_idx, gold in enumerate(golds):
                total += 1
                rows.append(
                    {
                        "id": f"{record_idx}-{q_idx}",
                        "record_id": record_id,
                        "golden": gold,
                        "output": "",
                        "matched": False,
                        "match_method": "missing_results_file",
                        "llm_reasoning": "",
                    }
                )
            continue

        responses = load_results_json(fpath)
        for q_idx, gold in enumerate(golds):
            total += 1
            if q_idx >= len(responses):
                rows.append(
                    {
                        "id": f"{record_idx}-{q_idx}",
                        "record_id": record_id,
                        "golden": gold,
                        "output": "",
                        "matched": False,
                        "match_method": "missing_turn",
                        "llm_reasoning": "",
                    }
                )
                continue

            pred = responses[q_idx].get("final_answer")
            det, method = deterministic_match(gold, pred)
            reasoning = ""

            if det is True:
                matched = True
            elif det is False:
                matched = False
            elif skip_llm:
                matched = False
                method = "llm_skipped"
            else:
                matched, reasoning = llm_compare(comparator, gold, pred)
                method = "llm"

            if matched:
                correct += 1

            rows.append(
                {
                    "id": f"{record_idx}-{q_idx}",
                    "record_id": record_id,
                    "golden": gold,
                    "output": pred if pred is not None else "",
                    "matched": matched,
                    "match_method": method,
                    "llm_reasoning": reasoning,
                }
            )

    fieldnames = [
        "id",
        "record_id",
        "golden",
        "output",
        "matched",
        "match_method",
        "llm_reasoning",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})

    summary_line = f"accuracy: {correct}/{total}\n"
    out_summary.write_text(summary_line, encoding="utf-8")
    print(summary_line.strip())
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_summary}")

    return correct, total


def main() -> None:
    p = argparse.ArgumentParser(description="Universal accuracy grader (subset gold vs results JSON).")
    p.add_argument(
        "--subset",
        type=Path,
        default=SUBSET_DEFAULT,
        help="JSON file with dev[] records and dialogue.executed_answers",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path(results_dir),
        help="Directory containing <record-file-id>.json pipeline outputs",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help=f"Output CSV path (default: <results-dir>/{DEFAULT_GRADES_NAME})",
    )
    p.add_argument(
        "--out-summary",
        type=Path,
        default=None,
        help=f"Output summary path (default: <results-dir>/{DEFAULT_SUMMARY_NAME})",
    )
    p.add_argument(
        "--llm-model",
        default="openai:gpt-5-mini",
        help="Model id for LangChain (fallback judge)",
    )
    p.add_argument(
        "--skip-llm",
        action="store_true",
        help="Deterministic matching only; unresolved rows count as not matched",
    )
    args = p.parse_args()

    results_path = Path(args.results_dir).expanduser().resolve()
    out_csv = (
        Path(args.out_csv).expanduser().resolve()
        if args.out_csv is not None
        else results_path / DEFAULT_GRADES_NAME
    )
    out_summary = (
        Path(args.out_summary).expanduser().resolve()
        if args.out_summary is not None
        else results_path / DEFAULT_SUMMARY_NAME
    )

    run_grade(
        subset_path=args.subset,
        results_path=results_path,
        out_csv=out_csv,
        out_summary=out_summary,
        llm_model=args.llm_model,
        skip_llm=args.skip_llm,
    )


if __name__ == "__main__":
    main()
