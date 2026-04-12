from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from src.agents.base import AgentBase
from src.utils.data.build_vector_db import VectorDB
from src.utils.data.read_dataset import DatasetDict
from global_utils.filepaths import dataset_fpath
from src.agents.types import State


class DirectResponse(BaseModel):
    can_answer_directly: bool = Field(description="Indicates if the agent can answer the question without additional computations, based only on the provided information.")
    retrieved_information: str = Field(description="Information retrieved that is relevant to the user's question.")
    answer: str = Field(
        description="If answering: a single numeric literal only—full magnitude in plain digits (e.g. -4000000 not '-4 million', 93000000 not '93' with a millions note). No currency symbols, units, prose, formulas, or fractions. Otherwise None."
    )


class DirectQA(AgentBase):
    def __init__(self, memory: InMemorySaver, memory_thread: int, llm: str = 'openai:gpt-5-mini', **kwargs):
        super().__init__(memory=memory, llm=llm, response_format=DirectResponse, memory_thread=memory_thread, **kwargs)

    def run(self, state: State, feedback: str = '') -> State:

        try:
            retrieved = state.vector_db.query(state.final_question, top_n=5)
            prompt = "You are a hybrid router-cum-retrieval agent."
            prompt += f"User feedback on previous output: {feedback}\n" if feedback else ""
            prompt += f"""Your task is to determine from the provided information, if the question can be answered without requiring any additional computation. That means to say, if the question can be answered directly from the provided information, without needing to perform any calculations, data manipulations, or further analysis, then you should answer it. If the question requires any form of computation, data manipulation, or further analysis beyond simply retrieving and presenting information from the provided context, then you should not attempt to answer it and instead respond with "None".
                            Question: {state.final_question}
                            Information: {retrieved}
                            All non-date table numeric values are in float form rounded to 2 decimal point. Treat the values as is with no scaling factor. Meaning, 1234.00 means one thousand two hundred and thirty four, not one million two hundred and thirty four thousand. The dot does not represent thousands.
                            Remember, all numeric values has no scaling factor unless explicitly stated. Do not assume that a number is in thousands or millions unless explictly stated.
                            If you output a numeric answer, it must be one plain number string only: the full value (apply any millions/billions scaling yourself so the digits match the true magnitude). No $, no 'per share', no explanatory text, no expressions like a/b.
                            """
            response = self._call_agent(prompt)

            state.can_answer_directly = response['structured_response'].can_answer_directly
            state.final_answer = response['structured_response'].answer if state.can_answer_directly else None
            state.step_history.append(
                {
                    "agent": "DirectQAAgent",
                    "input": {
                        "question": state.question,
                        "previous_questions": state.previous_questions,
                        "final_question": state.final_question,
                    },
                    "output": {
                        "can_answer_directly": state.can_answer_directly,
                        #"retrieved_information": response['structured_response'].retrieved_information,
                        "retrieved_chunks": retrieved,
                        "final_answer": state.final_answer,
                        "errors": None,
                    }
                }
            )
            return state
        except Exception as e:
            state.errors['direct_qa_error'] = e
            state.step_history.append(
                {
                    "agent": "DirectQAAgent",
                    "input": {
                        "question": State.question,
                        "previous_questions": State.previous_questions,
                    },
                    "output": {
                        "final_question": None,
                        'can_answer_directly': None,
                        "retrieved_chunks": None,
                        "errors": state.errors['clarifier_error'],
                    }
                }
            )
            return state


if __name__ == "__main__":
    from src.agents.clarifier import ClarifierAgent

    ds = DatasetDict(dataset_fpath)
    rec = ds.get_record('Single_ADBE/2018/page_86.pdf-1', subset='dev')
    vector_db = VectorDB(
        record=rec
    )
    mem = InMemorySaver()
    clarifier = ClarifierAgent(mem, 1)
    dqa = DirectQA(mem, 2)
    pq = []
    for q in rec.dialogue.conv_questions:
        state = State(question=q, vector_db=vector_db, previous_questions=pq)

        state = clarifier.run(state)
        state = dqa.run(state)
        print(state)
        pq.append(q)