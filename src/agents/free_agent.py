import os
from typing import List

import pydantic
from langgraph.checkpoint.memory import InMemorySaver

from src.agents.base import AgentBase
from src.agents.types import State
from src.utils.data.build_vector_db import VectorDB


class SubProblemAnswer(pydantic.BaseModel):
    #relevant_information: str = pydantic.Field(description="Relevant information retrieved from the context.")
    answer: str = pydantic.Field(description="The answer to the subproblem.")


class FreeAgent(AgentBase):
    def __init__(self, memory:InMemorySaver, memory_thread: int, llm: str = 'openai:gpt-5-mini', **kwargs):
        super().__init__(memory=memory, llm=llm, response_format=SubProblemAnswer, memory_thread=memory_thread, **kwargs)

    def run(self, state: State, feedback: str = '') -> State:
        try:
            self.agent_num = state.free_agent_idx
            subproblem = state.current_subproblem
            _vector_db: VectorDB = state.vector_db
            if _vector_db is None:
                raise ValueError("VectorDB instance must be provided in the state.")

            retrieved = _vector_db.query(subproblem, top_n=5)


            prompt = "You are a Retrieval Agent."
            prompt += f'User feedback on previous output: {feedback}\n' if feedback else ""
            prompt += f"""Your task is to retrieve relevant information from the provided context to help answer the user's subproblem. Use the context to find the most pertinent information that can assist in addressing the subproblem. All non-date table numeric values are in float form rounded to 2 decimal point. Treat the values as is with no scaling factor. Meaning, 1234.00 means one thousand two hundred and thirty four, not one million two hundred and thirty four thousand. The dot does not represent thousands.
                    Subproblem: {subproblem}
                    Context: {retrieved}
                    Remember, all numeric values has no scaling factor unless explicitly stated in words, do not assume that a number is in thousands or millions. By default, all non-date numbers are in float form, with the '.' representing the decimal point.
                    """
            response = self._call_agent(prompt)

            state.current_subproblem_answer = response['structured_response'].answer
            state.step_history.append({
                "agent": "FreeAgent",
                "input": {
                    "subproblem": subproblem,
                },
                "output": {
                    "answer": state.current_subproblem_answer,
                    #"retrieved_information": response['structured_response'].relevant_information,
                    "retrieved_chunks": retrieved,
                    "errors": None,
                }
            })
            return state
        except Exception as e:
            print("error:", e)
            print("traceback:", e.__traceback__.__str__())
            state.errors[f'free_agent_error{self.agent_num}'] = e
            state.step_history.append({
                "agent": "FreeAgent",
                "input": {
                    "subproblem": state.current_subproblem,
                },
                "output": {
                    "answer": None,
                    "retrieved_chunks": None,
                    "errors": state.errors[f'free_agent_error{self.agent_num}'],
                }
            })

            return state


if __name__ == "__main__":
    from src.agents.clarifier import ClarifierAgent
    from src.utils.data.read_dataset import DatasetDict
    from src.utils.filepaths import dataset_fpath
    from src.agents.direct_qa import DirectQA
    from src.agents.decomposer import Decomposer
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ds = DatasetDict(dataset_fpath)
    rec = ds.get_record('Single_ADBE/2018/page_86.pdf-1', subset='dev')
    vector_db = VectorDB(
        record=rec
    )
    pq = []
    mem = InMemorySaver()
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
        subproblems = state.subproblems


        def _run_one(sp_idx_sp) -> State:
            sp_idx, sp_text = sp_idx_sp
            agent = FreeAgent(memory=mem, memory_thread=4 + sp_idx)
            sub_state = state.model_copy(deep=True)
            sub_state.current_subproblem = sp_text
            sub_state.free_agent_idx = sp_idx
            return agent.run(sub_state)


        max_workers = min(os.cpu_count(), len(subproblems))
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, sp in enumerate(subproblems):
                futures.append(ex.submit(_run_one, (i, sp)))

            # Collect results (preserve original order)
            state.subproblems_dict_lst = [None] * len(subproblems)
            for fut in as_completed(futures):
                sub_state = fut.result()
                state.subproblems_dict_lst[sub_state.free_agent_idx] = {
                    "subproblem": sub_state.current_subproblem,
                    "answer": sub_state.current_subproblem_answer,
                }
                state.step_history.extend(sub_state.step_history)

        print(state)


