"""Single-turn user message: explicit question history array + current question (vanilla v1)."""

from __future__ import annotations

import json


def build_vanilla_user_message(question_history: list[str], current_question: str) -> str:
    """
    One LLM call per turn: prior turns as a JSON array, then the question to answer.
    No clarifier / rewrite — history is passed through verbatim.
    """
    hist = list(question_history)
    payload = {
        "question_history": hist,
        "current_question": current_question,
    }
    return (
        "Use the document in the system message. Below is a JSON object with:\n"
        '- "question_history": prior turns in order (context only; do not answer them again).\n'
        '- "current_question": the only question you must answer now.\n\n'
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Your entire reply must be only that answer: a plain number (or number + trailing % for true "
        "percentage answers), with no other text. **Fully expand magnitude** using the document’s stated scale "
        "(thousands, millions, billions, etc.): e.g. a change of “-4” in “$ millions” must be "
        "output as **-4000000**, not **-4**. Do not use words (million, k, M), scientific notation, "
        "or currency symbols. No sentences, labels, or prefixes like \"Answer:\"."
    )


def build_rewritten_answer_user_message(rewritten_question: str) -> str:
    """
    Answer phase when ``--rewrite`` was used: the model only sees this standalone question
    (plus document in the system message), not the full ``question_history`` JSON.
    """
    rq = (rewritten_question or "").strip()
    return (
        f"Answer this question using the document in the system message:\n\n{rq}\n\n"
        "Your entire reply must be only that answer: a plain number (or number + trailing % for true "
        "percentage answers), with no other text. **Fully expand magnitude** using the document’s stated scale "
        "(thousands, millions, billions, etc.): e.g. a change of “-4” in “$ millions” must be "
        "output as **-4000000**, not **-4**. Do not use words (million, k, M), scientific notation, "
        "or currency symbols. No sentences, labels, or prefixes like \"Answer:\"."
    )
