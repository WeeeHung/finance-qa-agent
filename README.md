# CRDFAgent

This repository contains conversational financial QA work on ConvFinQA-style records: a full multi-agent pipeline under `src/`, plus later **attempts and experimentations** in `src_v1`, `src_v2`, and `src_v3`.

**Submission answer (Fully Human written):** the substantive write-up is **[NOTES.md](NOTES.md)**. It follows the flow (analysis → research → improvements → next steps), covers the original architecture, dataset caveats, evaluation goals, hypotheses behind each code path, and what was measured or left for follow-up. Prefer that document for narrative, motivation, and conclusions; this README stays short and practical.

## Code versions (experimentation)


| Location      | What it is                                                                                                                                                                                                                                                          |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `**src/`**    | Original stack: planner-led pipeline with clarifier, direct QA short-circuit, decomposer, free agents, aggregator, vector RAG over the record, and persisted step history.                                                                                          |
| `**src_v1/**` | Lighter baseline: **vanilla** one LLM call per dialogue turn; optional **rewrite** pass so a first call condenses history + current question and the answer call sees only the rewritten text. Outputs default to `data/results_v1/` or `data/results_v1_rewrite/`. |
| `**src_v2/`** | ReAct-style direction: rewrite plus **Python execution** as a tool for numerical work.                                                                                                                                                                              |
| `**src_v3/`** | Extends v2 with a **KB-building** phase to consolidate context before tool use and answering.                                                                                                                                                                       |


Cross-version utilities (`.env` loading, canonical `data/` paths, universal graders) live under `**global_utils/`**. Dataset helpers and original-pipeline scoring remain under `**src/utils/`**. Version-specific `runme.py` / CLI entry points live next to each `src_v*` tree.

## Universal accuracy and latency graders

Cross-version evaluation is centralized so each experiment writes JSON under a results folder and the same tools can grade it.

- `**global_utils/universal_accuracy_grader.py**` — For each turn, compares model output to golden `**executed_answers**` from a subset JSON (default `data/convfinqa_datasubset.json`). Uses deterministic lenient matching first, then an optional LLM-as-judge fallback. Writes `universal_accuracy_grades.csv` and `universal_accuracy_summary.txt` into the chosen results directory unless you override output paths.
- `**global_utils/universal_latency_grader.py**` — Reads `**latency_ms**` per turn. In **single-directory** mode it reports distribution stats (min, max, median, quartiles). In **compare** mode it aligns files between a baseline and a candidate directory for old-vs-new timing (see script defaults and flags).

Concrete command lines (including `**results_v1_rewrite`** as the example) are in **[Getting started](#getting-started)** at the end of this file.

## Environment and configuration

1. Use **Python 3.10** (as in the original project notes).
2. Install dependencies: `pip install -r requirements.txt`
3. Copy [.env.example](.env.example) to `**.env`** in the project root. Set `**OPENAI_API_KEY**`. Optionally set `**CRDF_SRC_DIR**` and `**CRDF_DATA_DIR**` if `src` or `data` are not the default paths next to the repo root (see comments in `.env.example`).
4. Put the **project root** on `**PYTHONPATH`** so imports such as `src`, `global_utils`, and `src_v1` resolve:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

Runners and graders call `load_project_env()`, which loads `.env` from the project root. The project was developed on **macOS**; other platforms are not guaranteed.

## Original pipeline entry points (`src/`)

For diagrams, agent responsibilities, and pros/cons, see **NOTES.md** (especially §1 Analysis). Overview diagram:

Agentic Framework diagram

- Batch-style driver: `src/runme.py`
- Interactive CLI over a single record: `src/app/cli.py`
- Scoring tailored to the original pipeline outputs: `src/utils/scoring.py`

---

# Getting started

This section is a minimal command reference for **vanilla v1** batch runs (including the optional rewrite pass) and for the **universal accuracy** / **universal latency** graders. It assumes the project root is on `PYTHONPATH` (see [Environment and configuration](#environment-and-configuration)). Run commands from the project root unless you adjust paths.

## Vanilla v1 (`src_v1.runme`)

- **Default output**: `data/results_v1/` (first ten dev records, `slice(0, 10)` unless you pass `--start` / `--end`).
- **With `--rewrite`**: writes to `data/results_v1_rewrite/` (rewrite pass condenses history + current question, then the answer model sees only the rewritten question).
- `**--overwrite**`: regenerate JSON even when an output file already exists; without it, existing files are skipped.

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# v1 without rewrite → data/results_v1/
python3 -m src_v1.runme --overwrite

# v1 with rewrite → data/results_v1_rewrite/
python3 -m src_v1.runme --rewrite --overwrite

# Optional: custom slice (inclusive start, exclusive end)
python3 -m src_v1.runme --rewrite --overwrite --start 0 --end 10
```

If you set `CRDF_DATA_DIR` in `.env`, outputs live under that data directory instead; paths below should match your configured `data` root.

## Universal accuracy (example: `results_v1_rewrite`)

Grades pipeline JSON under `--results-dir` against golden `executed_answers` in the subset file (default `data/convfinqa_datasubset.json`). By default it writes `universal_accuracy_grades.csv` and `universal_accuracy_summary.txt` **into** the results directory.

```bash
python3 -m global_utils.universal_accuracy_grader --results-dir data/results_v1_rewrite
```

Use `--subset` if your gold labels live in another JSON file. Use `--skip-llm` for deterministic matching only.

## Universal latency (example: `results_v1_rewrite`)

**Single-directory** mode (stats only for one run): pass `--results-dir`. Writes `universal_latency_grades.csv` and `universal_latency_summary.txt` into that directory by default.

```bash
python3 -m global_utils.universal_latency_grader --results-dir data/results_v1_rewrite
```

With no `--results-dir`, the script runs **compare** mode (baseline vs candidate directories); defaults are wired for other pipelines—see `global_utils/universal_latency_grader.py` if you need `--baseline-dir` / `--candidate-dir`.