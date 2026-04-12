"""ReAct agent turn: LangGraph create_react_agent + execute_python tool."""

from __future__ import annotations

import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from src_v1.answer_parse import extract_ai_text, normalize_convfinqa_answer
from src_v1.context import format_convfinqa_context
from src_v1.llm import make_chat_model
from src_v2.prompts import REACT_SYSTEM, REACT_SYSTEM_REWRITTEN_ONLY
from src_v2.python_tool import execute_python

# Graph steps include model and tool nodes; cap total transitions (see langgraph recursion_limit).
RECURSION_LIMIT = 28

TOOL_NAME = "execute_python"


def _serialize_tool_calls(msg: AIMessage) -> list[dict[str, Any]] | None:
    if not msg.tool_calls:
        return None
    out: list[dict[str, Any]] = []
    for tc in msg.tool_calls:
        if isinstance(tc, dict):
            out.append(
                {
                    "name": tc.get("name"),
                    "args": tc.get("args"),
                    "id": tc.get("id"),
                }
            )
        else:
            out.append(
                {
                    "name": getattr(tc, "name", None),
                    "args": getattr(tc, "args", None),
                    "id": getattr(tc, "id", None),
                }
            )
    return out


def messages_to_steps(messages: list[BaseMessage], llm_ms: float) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = [
        {"type": "timing", "latency_ms": round(llm_ms, 3), "llm_ms_total": round(llm_ms, 3)},
    ]
    for msg in messages:
        if isinstance(msg, SystemMessage):
            c = msg.content
            preview = c if isinstance(c, str) else str(c)
            if len(preview) > 4000:
                preview = preview[:4000] + "\n...(truncated)"
            steps.append({"type": "system", "content": preview})
        elif isinstance(msg, HumanMessage):
            steps.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            step: dict[str, Any] = {"type": "assistant", "content": extract_ai_text(msg)}
            tc = _serialize_tool_calls(msg)
            if tc:
                step["tool_calls"] = tc
            steps.append(step)
        elif isinstance(msg, ToolMessage):
            steps.append(
                {
                    "type": "tool",
                    "name": msg.name or "",
                    "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                }
            )
    return steps


def _final_ai_answer_text(messages: list[BaseMessage]) -> str:
    """Prefer the last AIMessage that is not only a tool-call request."""
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if msg.tool_calls and not (extract_ai_text(msg) or "").strip():
            continue
        return extract_ai_text(msg)
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return extract_ai_text(msg)
    return ""


def run_react_turn(
    raw_data: dict[str, Any],
    user_message: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    verbose: bool = False,
    answer_style: str = "history_json",
) -> dict[str, Any]:
    ctx = format_convfinqa_context(raw_data)
    sys_base = REACT_SYSTEM_REWRITTEN_ONLY if answer_style == "rewritten_only" else REACT_SYSTEM
    system_text = f"{sys_base}\n\n=== Document context ===\n{ctx}"

    chat = make_chat_model(model, temperature)
    agent = create_react_agent(
        chat,
        tools=[execute_python],
        checkpointer=InMemorySaver(),
    )

    messages_input = [
        SystemMessage(content=system_text),
        HumanMessage(content=user_message),
    ]

    if verbose:
        rid = raw_data.get("id", "")
        print(f"[src_v2] id={rid!r} | ReAct invoke ...", flush=True)

    t0 = time.perf_counter()
    out = agent.invoke(
        {"messages": messages_input},
        {
            "recursion_limit": RECURSION_LIMIT,
            "configurable": {"thread_id": str(uuid.uuid4())},
        },
    )
    llm_ms = (time.perf_counter() - t0) * 1000.0

    final_messages: list[BaseMessage] = list(out.get("messages", []))
    raw_reply = _final_ai_answer_text(final_messages)
    answer_text = normalize_convfinqa_answer(raw_reply)

    n_llm = sum(1 for m in final_messages if isinstance(m, AIMessage))
    n_sandbox = sum(
        1 for m in final_messages if isinstance(m, ToolMessage) and (m.name or "") == TOOL_NAME
    )

    steps = messages_to_steps(final_messages, llm_ms)

    if verbose:
        print(
            f"[src_v2] {llm_ms:.0f} ms | llm_calls={n_llm} | tools={n_sandbox} | answer={answer_text!r}",
            flush=True,
        )

    return {
        "answer_text": answer_text,
        "latency_ms": llm_ms,
        "llm_ms_total": llm_ms,
        "steps": steps,
        "llm_invocations": n_llm,
        "sandbox_invocations": n_sandbox,
    }
