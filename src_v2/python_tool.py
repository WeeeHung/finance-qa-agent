"""Sandboxed Python execution exposed as a LangChain tool for the ReAct agent."""

from __future__ import annotations

import io
import math
from contextlib import redirect_stdout
from multiprocessing import Process, Queue
from typing import Any

from langchain_core.tools import tool

# Wall-clock cap for exec (seconds). Keeps runaway loops from hanging the batch job.
_EXEC_TIMEOUT_SEC = 8.0
_MAX_OBSERVATION_CHARS = 12_000

_ALLOWED_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def _run_exec(code: str) -> str:
    buf = io.StringIO()
    safe_globals = {"__builtins__": _ALLOWED_BUILTINS, "math": math}
    safe_locals: dict[str, Any] = {}
    try:
        with redirect_stdout(buf):
            exec(code, safe_globals, safe_locals)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    out = buf.getvalue().strip()
    if "result" in safe_locals:
        r = safe_locals["result"]
        line = repr(r) if not isinstance(r, str) else r
        if out:
            return f"{out}\nresult={line}"
        return f"result={line}"
    if out:
        return out
    return "No output: assign the final numeric value to variable `result` or print it."


def _worker(code: str, q: Queue) -> None:
    q.put(_run_exec(code))


def execute_python_code(code: str) -> str:
    """Run `code` in a restricted environment; use variable `result` or print()."""
    code = (code or "").strip()
    if not code:
        return "Error: empty code."

    q: Queue = Queue(maxsize=1)
    p = Process(target=_worker, args=(code, q), daemon=True)
    p.start()
    p.join(_EXEC_TIMEOUT_SEC)
    if p.is_alive():
        p.terminate()
        p.join()
        return f"Error: execution exceeded {_EXEC_TIMEOUT_SEC}s timeout."

    if q.empty():
        return "Error: execution failed without output."

    text = str(q.get())

    if len(text) > _MAX_OBSERVATION_CHARS:
        return text[: _MAX_OBSERVATION_CHARS] + "\n...(truncated)"
    return text


@tool("execute_python")
def execute_python(code: str) -> str:
    """Run Python for multi-step arithmetic. Copy numbers from the document into literals.

    Assign the final value to `result` (preferred) or `print()` it. Only basic math and the
    `math` module are available; no file or network access.
    """
    return execute_python_code(code)
