from typing import Union, Literal

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from src.agents.clarifier import ClarifierAgent
from src.agents.types import State
from src.utils.data.types import ConvFinQARecord
from src.agents.direct_qa import DirectQA
from src.agents.decomposer import Decomposer
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.agents.free_agent import FreeAgent
from src.agents.aggregator import Aggregator
from src.utils.data.build_vector_db import VectorDB
import os
from src.agents.base import AgentBase


class PlannerResponse(BaseModel):
    further_action_required: bool = Field(description="If feedback was given to the agent, set to True. Else False.")
    feedback: Union[str, None] = Field(description="Feedback to the previous agent's action, if any.")


class FreeAgentWrapper:
    def run(self, state: State, mem: InMemorySaver, feedback: str) -> State:
        subproblems = state.subproblems

        def _run_one(sp_idx_sp) -> State:
            sp_idx, sp_text = sp_idx_sp
            agent = FreeAgent(memory=mem, memory_thread=5 + sp_idx)
            sub_state = state.model_copy(deep=True)
            sub_state.current_subproblem = sp_text
            sub_state.free_agent_idx = sp_idx
            return agent.run(sub_state, feedback)

        max_workers = min(os.cpu_count(), len(subproblems))
        futures = []
        if max_workers >= 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for i, sp in enumerate(subproblems):
                    futures.append(ex.submit(_run_one, (i, sp)))

                # Collect results (preserve original order)
                state.subproblems_dict_lst = [None] * len(subproblems)
                free_agents_histories = []
                for fut in as_completed(futures):
                    sub_state = fut.result()
                    state.subproblems_dict_lst[sub_state.free_agent_idx] = {
                        "subproblem": sub_state.current_subproblem,
                        "answer": sub_state.current_subproblem_answer,
                    }
                    free_agents_histories.append(sub_state.step_history[-1])
                state.step_history.append(free_agents_histories)

        return state


class Planner(AgentBase):
    def __init__(self, rec: ConvFinQARecord, memory:InMemorySaver, memory_thread: int,
                 verbose_mode:Literal['debug', 'error', 'info'], agent_retry_limit: int=5,
                 llm: str = 'openai:gpt-5-mini', self_consistency:bool = False, **kwargs):
        super().__init__(memory=memory, llm=llm, response_format=PlannerResponse,
                         memory_thread=memory_thread, **kwargs)
        self.vector_db = VectorDB(
            record=rec
        )

        self.verbose_mode = verbose_mode

        self.agent_retry_limit = agent_retry_limit

        self.clarifier = ClarifierAgent(memory, memory_thread + 1)
        self.dqa = DirectQA(memory, memory_thread + 2)
        self.decomposer = Decomposer(memory, memory_thread + 3)
        self.free_agent_wrapper = FreeAgentWrapper()
        self.aggregator = Aggregator(memory, memory_thread + 4)
        self.agent_mapping = {
            "Clarifier": self.clarifier,
            "DirectQA": self.dqa,
            "Decomposer": self.decomposer,
            "FreeAgents": self.free_agent_wrapper,
            "Aggregator": self.aggregator,
        }
        self.execution_order = [
            ("Clarifier", ''), # agent, feedback (initial is empty str bc no feedback)
            ("DirectQA", ''),
            ("Decomposer", ''),
            ("FreeAgents", ''),
            ("Aggregator", ''),
        ]
        self.attempts_d = {agent: 0 for agent in self.agent_mapping.keys()}
        self.feedback_mapping = {
            "Clarifier": f"Given the Clarifier agent's execution summary, you are to check if the agent has been faithful to the previous questions asked. Faithfulness mean the agent did not fundamentally change the metric and/or period in cases where the metric and/or period is already specified clearly in the user's question. Else, explain why it failed and how to improve it. Sometimes, the agent might return the user's original question, that is ok so long as specific metric(s) or period(s) have been mentioned. If no further action required, return None. For feedback, do not suggest to ask the user. Give the best possible suggestion based on the previous questions asked and the user's question.",
            "DirectQA": "Given the DirectQA agent's execution summary, if the agent has given an answer, you are to check if the agent has faithfully extracted the answer from the context. Else if the agent has not given an answer, you are to check if the agent has correctly determined that the question cannot be answered directly from the context without requiring additional computations. Any arithmetic that is required for answering constitutes as additional computation. Therefore, so long as the question cannot be answered directly with retrieved facts without needing any additional arithmetic or math, it is considered to require additional computation. If the agent has correctly determined, the no further action is required, otherwise, explain why it failed and how to improve it. If no further action required, return None. If an answer is given, focus especially on the numerical values, and ensure that all numeric values are treated as is with no scaling factor. Meaning, 1234.00 means one thousand two hundred and thirty four, not one million two hundred and thirty four thousand. The dot does not represent thousands. Ensure that all numerical values are given without words (i.e. 1000000 not 1 million), currency symbols are okay.",
            "Decomposer": "Given the Decomposer agent's execution summary, if the agent has given subproblems, you are to check if the subproblems are relevant to the final question and can lead to the final answer when solved. You do not have to check if the arithmetic step required to derive the final ansewr is present, as the decomposer is supposed to generate non-compute, retrieval-only subproblems. Additionally, check if the subproblems are atomic in nature, and requires only retrieval and no additional computations. You do not need to check for ambiguity. Else, explain why it failed and how to improve it. If no further action required, return None.",
            "FreeAgents": "Given the FreeAgents agent's execution summary, you are to check if each free agent has faithfully retrieved the answer from the context for each subproblem. Else, explain why it failed and how to improve it. Focus especially on the numerical values, and ensure that all numeric values are treated as is with no scaling factor. Meaning, 1234.00 means one thousand two hundred and thirty four, not one million two hundred and thirty four thousand. The dot does not represent thousands. If no further action required, return None. Ensure that all numerical values are given without words (i.e. 1000000 not 1 million), currency symbols are okay. Verify that the free agents output has the appropriate scale with respect to the retrieved chunks. If the retrieved chunks indicate that the values are in thousands or millions, ensure that the free agents output reflects that. For example, if the retrieved chunks indicate that the values are in thousands, and the free agent outputs 1234.00, then the actual value is 1,234,000.00. Ensure that the free agents output reflects that scaling factor.",
            "Aggregator": "Given the Aggregator agent's execution summary, check if the agent's code correctly aggregates the answers from the subproblems to form a final answer that addresses the user's original question. Specifically, if the question asks for net change, ensure that the code reflects net_change = new_value - old_value. Else if the question asks for percent_change, ensure that the code reflects (new_value - old_value) / old_value. If it failed, explain why it failed and how to improve it. If no further action required, return None.",
        }
        self.pipeline = []
        self.self_consistency = self_consistency

    def clear_memory(self, num_subproblems: int) -> None:
        for i in range(self.memory_thread + 4 + num_subproblems + 1):  # 4 base agents (offset by planner's memory thread) + free agents + 1
            self.memory.delete_thread(str(i))

    def run(self, state: State, feedback: str = '') -> State:
        self.pipeline = [*self.execution_order] # reset pipeline
        self.pipeline.reverse() # to use it as a stack

        while self.pipeline: # while there are still agents to execute
            current_agent, feedback = self.pipeline.pop()
            self.attempts_d[current_agent] += 1
            if current_agent == "FreeAgents":
                state = self.agent_mapping[current_agent].run(state, self.memory, feedback)
            else:
                state = self.agent_mapping[current_agent].run(state, feedback)
            if self.attempts_d[current_agent] > self.agent_retry_limit + 1:
                if self.verbose_mode in ['debug', 'error']:
                    print(f"====== {current_agent} exceeded maximum retry limit, skipping further attempts ======")
                continue
            if self.verbose_mode in ['debug', 'error', 'info']:
                print(f"====== {current_agent} finished execution ======")
                if self.verbose_mode in ['debug', 'error']:
                    print(state.step_history[-1])
            prompt = f"You are a planner agent that gives feedback to the {current_agent} agent based on its execution summary. {self.feedback_mapping[current_agent]}'s execution summary is as follows:\n{state.step_history[-1]}"
            response = self._call_agent(prompt)

            if self.self_consistency:
                further_action_required = response['structured_response'].further_action_required
                if further_action_required:
                    feedback = response['structured_response'].feedback
                    if self.verbose_mode in ['debug', 'error']:
                        print("====== Further action required, re-inserting agent into pipeline ======")
                        print(f"Feedback: {feedback}")
                    self.pipeline.append((current_agent, feedback))
            if state.final_answer is not None and not feedback: # if final answer found and no feedback needs to be addressed
                if self.verbose_mode in ['debug', 'error', 'info']:
                    print("====== Direct answer found, terminating pipeline ======")
                break # short circuit for when answer is found by DQA
        # if reached here means all agents executed successfully
        if not state.final_answer: # if final answer not set by DQA
            state.final_answer = state.computed_answer # set final answer to computed answer from aggregator
        state.agent_attempts = self.attempts_d
        self.clear_memory(len(state.subproblems) if state.subproblems else 0)
        self.attempts_d = {agent: 0 for agent in self.agent_mapping.keys()}  # reset attempts_d for next run
        if self.verbose_mode in ['debug', 'error', 'info']:
            print("====== All agents executed successfully, final answer computed ======")
        return state