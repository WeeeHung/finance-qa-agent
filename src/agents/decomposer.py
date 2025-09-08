from __future__ import annotations
from typing import Dict, Any, List
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from src.agents.base import AgentBase
from src.agents.types import State


class DecomposerResponse(BaseModel):
    subproblems: List[str] = Field(description="List of retrieval-only subproblems.")

class Decomposer(AgentBase):
    def __init__(self, memory: InMemorySaver, memory_thread: int, llm: str = 'openai:gpt-5-mini', **kwargs):
        super().__init__(memory=memory, llm=llm, response_format=DecomposerResponse, memory_thread=memory_thread, **kwargs)

    def run(self, state: State, feedback: str = '') -> State:
        try:
            prompt = "You are a Decomposer for financial QA."
            if feedback:
                prompt += f'User feedback on previous output: {feedback}\n'
            prompt += f"""Your task: Split the user's question into an arbitrary number of **retrieval-only** subproblems.
            Rules:
            - Each subproblem must ask only to **retrieve** one specific fact (from text or table), not to compute/compare.
            - Each subproblem should only ask for a numeric answer (either a dollar amount or a percentage only)
            - Each subproblem should never ask for the extent of change (e.g., "how much did X change"), only the value at a specific time (e.g., "what was X in 2020").
            - Avoid words like "compute", "calculate", "difference", "sum", "ratio", "percentage change". Instead, phrase as retrieval queries (e.g., "Find <metric> for <period>").
            - Use as many subproblems as necessary to cover all atomic facts needed to answer the question deterministically later.
            - Be concise but precise.
            - Do not include any explanations, only list the subproblems.
            - Each subproblem should be unambiguous and self-contained. Meaning, it should not contain any ambiguous pronouns or references to other subproblems.

            Question: {state.final_question}
            """
            response = self._call_agent(prompt)

            state.subproblems = response['structured_response'].subproblems
            state.step_history.append({
                "agent": "DecomposerAgent",
                "input": {
                    "final_question": state.final_question
                },
                "output": {
                    "subproblems": state.subproblems,
                    "errors": None,
                }
            })
            return state
        except Exception as e:
            state.errors['decomposer_error'] = e
            state.step_history.append({
                "agent": "DecomposerAgent",
                "input": {
                    "final_question": state.final_question
                },
                "output": {
                    "subproblems": [],
                    "errors": state.errors['decomposer_error'],
                }
            })
            return state

if __name__ == "__main__":
    from src.agents.clarifier import ClarifierAgent
    from src.utils.data.read_dataset import DatasetDict
    from src.utils.filepaths import dataset_fpath
    from src.agents.direct_qa import DirectQA
    from src.utils.data.build_vector_db import VectorDB

    ds = DatasetDict(dataset_fpath)
    rec = ds.get_record('Single_ADBE/2018/page_86.pdf-1', subset='dev')
    vector_db = VectorDB(
        record=rec
    )
    mem = InMemorySaver()
    pq = []
    clarifier = ClarifierAgent(mem, 1)
    dqa = DirectQA(mem, 2)
    decomposer = Decomposer(mem, 3)
    for q in rec.dialogue.conv_questions:
        state = State(question=q, vector_db=vector_db, previous_questions=pq)
        state = clarifier.run(state)
        state = dqa.run(state)
        pq.append(q)

        if state.can_answer_directly:
            print(state)
            continue

        state = decomposer.run(state)
        print(state)
