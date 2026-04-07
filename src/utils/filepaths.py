"""Project paths.

Override with CRDF_SRC_DIR and CRDF_DATA_DIR (absolute or relative paths accepted).
Defaults are next to the repo root inferred from this file's location.
"""
import os
from pathlib import Path

from src.bootstrap_env import load_project_env

load_project_env()


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _resolve_dir(env_name: str, default: Path) -> str:
    raw = os.environ.get(env_name)
    if raw:
        return str(Path(raw).expanduser().resolve())
    return str(default.resolve())


src_dir = _resolve_dir("CRDF_SRC_DIR", _project_root() / "src")
data_dir = _resolve_dir("CRDF_DATA_DIR", _project_root() / "data")

dataset_fpath = str(Path(data_dir) / "convfinqa_dataset.json")

vector_db_dir = str(Path(data_dir) / "vector_db")

results_dir = str(Path(data_dir) / "results")
