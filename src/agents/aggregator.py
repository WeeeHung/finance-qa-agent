from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from src.agents.base import AgentBase
from src.agents.types import State
from src.utils.data.build_vector_db import VectorDB


class AggregatorResponse(BaseModel):
    code: str = Field(description="The generated python function to be used to compute the final answer.")


def safe_exec(code: str) -> dict:
    """
    Safely executes Python code with restricted builtins.
    Returns the local execution environment (variables, results).
    Generated using ChatGPT
    """

    # Allowed builtins (you can expand this list if needed)
    allowed_builtins = {
        "print": print,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "sorted": sorted,
    }

    # Restrict environment
    safe_globals = {"__builtins__": allowed_builtins}
    safe_locals = {}

    try:
        exec(code, safe_globals, safe_locals)
    except Exception as e:
        return {"error": str(e)}

    return safe_locals



class Aggregator(AgentBase):
    def __init__(self, memory:InMemorySaver, memory_thread: int, llm: str = 'openai:gpt-5-mini', **kwargs):
        super().__init__(memory=memory, llm=llm, response_format=AggregatorResponse, memory_thread=memory_thread, **kwargs)

    def run(self, state: State, feedback: str = '') -> State:
        try:
            subproblems_dict_lst = state.subproblems_dict_lst

            if not subproblems_dict_lst:
                raise ValueError("No subproblem answers provided in the state.")

            if subproblems_dict_lst:
                combined_answers = f"Here are the answers to the subproblems:" + "\n".join(
                    f": {sub_dict['subproblem']}: {sub_dict['answer']}" for sub_dict in subproblems_dict_lst)
            else:
                combined_answers = f"There are no answers to the subproblems as subproblems do not exist. Refer back to the main question instead.\nMain Question: {state.final_question}"

            prompt = "You are an Aggregator Agent."
            prompt += f'User feedback on previous output: {feedback}\n' if feedback else ""
            prompt += f"""Your task is to generate a python program to compute the answer to the main question based on the answers of the multiple subproblems. You should think step by step, as some of the problems may require more complicated math to solve. The python program must be deterministic and should not involve any randomness. The python program must be contained within a function named "compute", and return the float value of the result. The python program should use only basic arithmetic operations (addition, subtraction, multiplication, division) and should not use any external libraries or functions. The python program should not include any explanations or comments.

            {combined_answers}
            """

            response = self._call_agent(prompt)

            generated_fn = response['structured_response'].code
            namespace = safe_exec(generated_fn)

            # get the function from the name space and execute it
            func_handle = namespace[list(namespace.keys())[0]]
            state.computed_answer = func_handle()
            state.step_history.append({
                "agent": "AggregatorAgent",
                "input": {
                    "subproblem_answers": subproblems_dict_lst,
                },
                "output": {
                    "aggregator_code": generated_fn,
                    "computed_answer": state.computed_answer,
                    "errors": None,
                }
            })
            return state
        except Exception as e:
            state.errors['aggregator_error'] = e
            state.step_history.append({
                "agent": "AggregatorAgent",
                "input": {
                    "subproblem_answers": state.subproblems_dict_lst,
                },
                "output": {
                    "aggregator_code": None,
                    "computed_answer": None,
                    "errors": state.errors['aggregator_error'],
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
    import os
    from src.agents.free_agent import FreeAgent

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
    aggregator = Aggregator(mem, 4)
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
            agent = FreeAgent(memory=mem, memory_thread=5 + sp_idx)
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
        state = aggregator.run(state)
        answer = state.final_answer if state.final_answer else state.computed_answer
        print(state)

