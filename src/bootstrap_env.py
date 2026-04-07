"""Load ``.env`` from the project root (parent of ``src``)."""
from pathlib import Path

from dotenv import load_dotenv


def load_project_env() -> None:
    # allow us to load the .env file from any directory
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
