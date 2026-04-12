from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from src.agents.base import AgentBase
from src.utils.data.read_dataset import DatasetDict
from global_utils.filepaths import dataset_fpath
from src.agents.types import State


class ClarifierResponse(BaseModel):
    final_question: str = Field(description="The final clarified question or the original question if it was clear.")


class ClarifierAgent(AgentBase):
    def __init__(self, memory: InMemorySaver, memory_thread: int, llm: str = 'openai:gpt-5-mini', **kwargs):
        response_format = ClarifierResponse
        super().__init__(memory=memory, llm=llm, response_format=response_format, memory_thread=memory_thread, **kwargs)

    def run(self, state: State, feedback: str = '') -> State:
        try:
            question = state.question
            prev_qns = state.previous_questions
            if prev_qns:
                prev_qns_str = "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(prev_qns)])
            else:
                prev_qns_str = ""
            prompt = "You are a Clarifier Agent. "
            if feedback:
                prompt += f"User feedback on previous output: {feedback}\n"
            prompt += f"""Your task is to determine if the user's question is ambiguous and if yes, rephrase the question to make it unambiguous by replacing any ambiguous pronouns with specific metrics and dates, referencing the previous question for context. You must be accurate in capturing the user's intention. Therefore, be careful when rephrasing ambiguous portions and ensure that you cross-reference past questions. Do not hallucinate, but rather rephrase the ambiguous question by referring closely to what was previously asked. Any questions are directed to a known report. Thus, any ambiguity regarding companies or reports should be ignored. Focus instead on ambiguous pronouns, and the lack of dates.  Unambiguous means the absence no room for doubt, and no 'or' statements. If the question is clear and unambiguous, return it as is. Otherwise, rephrase the question to eliminate ambiguity. The final question should not contain any clarifying questions, rather, it should be a clear and definitive version of the given question. Unless the given question asks for both dollar amount and percentage change, don't ask for both, only ask for one. Unless the given question asks for currency, do not specify currency. Give only the final question. 
                    Replace any "percent" term with the symbol "%".
                    Question: {question}
                    Previous Questions (for context):
                    {prev_qns_str if prev_qns else "None"}
                    """


            response = self._call_agent(prompt)
            state.final_question = response['structured_response'].final_question
            state.step_history.append(
                {
                    "agent": "ClarifierAgent",
                    "input": {
                        "question": question,
                        "previous_questions": prev_qns,
                    },
                    "output": {
                        "final_question": state.final_question,
                        "errors": None,
                    }
                }
            )
            return state
        except Exception as e:
            state.errors['clarifier_error'] = e
            state.step_history.append(
                {
                    "agent": "ClarifierAgent",
                    "input": {
                        "question": State.question,
                        "previous_questions": State.previous_questions,
                    },
                    "output": {
                        "final_question": None,
                        "errors": state.errors['clarifier_error'],
                    }
                }
            )
            return state


if __name__ == "__main__":
    ds = DatasetDict(dataset_fpath)
    rec = ds.get_record('Double_ADBE/2018/page_86.pdf', subset='dev')
    mem = InMemorySaver()
    clarifier = ClarifierAgent(mem, 1)
    pq = []
    for q in rec.dialogue.conv_questions:
        state = State(question=q, previous_questions=pq)
        new_state = clarifier.run(state)
        print(new_state)
        pq.append(q)