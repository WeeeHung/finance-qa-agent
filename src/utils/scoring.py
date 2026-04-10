import json
import os
import re
from typing import Dict, Tuple, Union, List

from src.bootstrap_env import load_project_env

load_project_env()

import pandas as pd
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.utils.data.read_dataset import DatasetDict
from src.utils.data.types import ConvFinQARecord
from src.utils.filepaths import results_dir, dataset_fpath


def convert_fname_to_recordid(fname: str) -> str:
    record_title_parts = fname.replace('.json', '').split('.pdf')
    record_id = record_title_parts[0].replace('-', '/') + '.pdf' + record_title_parts[1]
    return record_id


class AnswerComparatorResponse(BaseModel):
    correct: bool = Field(..., description="Whether the predicted answer matches the gold answer without needing to compute anything.")
    reasoning: str = Field(..., description="The reasoning behind the evaluation")


class AgentEvalResponse(BaseModel):
    achieved_task: bool = Field(..., description="Whether the agent achieved the task successfully")
    reasoning: str = Field(..., description="The reasoning behind the evaluation")


class Scorer:
    def __init__(self):
        self.file_lst = os.listdir(results_dir)
        self.llm = 'openai:gpt-5-mini'
        self.answer_comparator = self._create_agent(AnswerComparatorResponse)
        self.agent_eval = self._create_agent(AgentEvalResponse)
        self.eval_mapping = {
            "ClarifierAgent": f"Given the Clarifier agent's execution summary, you are to check if the agent has been faithful to the previous questions asked. Faithfulness mean the agent did not fundamentally change the metric and/or period in cases where the metric and/or period is already specified clearly in the user's question. Else, explain why it failed and how to improve it. Sometimes, the agent might return the user's original question, that is ok so long as specific metric(s) or period(s) have been mentioned.",
            "DirectQAAgent": "Given the DirectQA agent's execution summary, if the agent has given an answer, you are to check if the agent has faithfully extracted the answer from the context. Else if the agent has not given an answer, you are to check if the agent has correctly determined that the question cannot be answered directly from the context without requiring additional computations. Any arithmetic that is required for answering constitutes as additional computation. Therefore, so long as the question cannot be answered directly with retrieved facts without needing any additional arithmetic or math, it is considered to require additional computation. If an answer is given, focus especially on the numerical values, and ensure that all numeric values are treated as is with no scaling factor. Meaning, 1234.00 means one thousand two hundred and thirty four, not one million two hundred and thirty four thousand. The dot does not represent thousands. Ensure that all numerical values are given without words (i.e. 1000000 not 1 million), currency symbols are okay.",
            "DecomposerAgent": "Given the Decomposer agent's execution summary, if the agent has given subproblems, you are to check if the subproblems are relevant to the final question and can lead to the final answer when solved. You do not have to check if the arithmetic step required to derive the final ansewr is present, as the decomposer is supposed to generate non-compute, retrieval-only subproblems. Additionally, check if the subproblems are atomic in nature, and requires only retrieval and no additional computations. You do not need to check for ambiguity.",
            "FreeAgent": "Given the FreeAgents agent's execution summary, you are to check if each free agent has faithfully retrieved the answer from the context for each subproblem. Else, explain why it failed and how to improve it. Focus especially on the numerical values, and ensure that all numeric values are treated as is with no scaling factor. Meaning, 1234.00 means one thousand two hundred and thirty four, not one million two hundred and thirty four thousand. The dot does not represent thousands. Ensure that all numerical values are given without words (i.e. 1000000 not 1 million), currency symbols are okay. Verify that the free agents output has the appropriate scale with respect to the retrieved chunks. If the retrieved chunks indicate that the values are in thousands or millions, ensure that the free agents output reflects that. For example, if the retrieved chunks indicate that the values are in thousands, and the free agent outputs 1234.00, then the actual value is 1,234,000.00. Ensure that the free agents output reflects that scaling factor.",
            "AggregatorAgent": "Given the Aggregator agent's execution summary, check if the agent's code correctly aggregates the answers from the subproblems to form a final answer that addresses the user's original question. Specifically, if the question asks for net change, ensure that the code reflects net_change = new_value - old_value. Else if the question asks for percent_change, ensure that the code reflects (new_value - old_value) / old_value. If it failed, explain why it failed and how to improve it.",
        }

    @staticmethod
    def _get_last_agent_step(step_history: List[dict], agent_name) -> Union[dict, List[dict]]:
        for agent in reversed(step_history):
            if isinstance(agent, list): # is freeagents
                if agent_name == 'FreeAgent':
                    return agent
            else:
                if agent['agent'] == agent_name:
                    return agent


    @staticmethod
    def load_result(f: str) -> list:
        with open(f'{results_dir}/{f}', 'r', encoding='utf-8') as infile:
            if ".json" in f:
                json_str = '[' + re.sub(r"}\s*{", "},{", infile.read()) + ']'
                responses = json.loads(json_str)

        return responses

    def _create_agent(self, response_format):
        return create_react_agent(
            model=self.llm,
            response_format=response_format,
            tools=[],  # No tools needed
        )

    def compare_answers(self, gold: str, pred: str) -> tuple:
        if gold == pred:
            return True, 'Gold and predicted answers match exactly.'
        else:
            ans = {
                "gold_answer": gold,
                "predicted_answer": pred
            }
            prompt = (
                "You are an expert evaluator. Gold is the final ground-truth value.\n"
                "If the gold is numeric, the prediction must be a single final scalar, not an expression (e.g. a/b), "
                "not intermediate steps. If further computation is needed beyond trivial rounding, incorrect. "
                "Percentage vs decimal of the same value is acceptable. "
                f"Data: {ans}"
            )
            response = self.answer_comparator.invoke({
                "messages": [{"role": "user", "content": prompt}],
            })

            return response['structured_response'].correct, response['structured_response'].reasoning

    def _eval_reasoning(self, agent_output: Union[dict, List[dict]]) -> Tuple[str, str]:
        if isinstance(agent_output, list): # is freeagents
            agent_name = "FreeAgent"
        else:
            agent_name = agent_output['agent']

        #("Agent: ", agent_name)

        prompt = "You are an expert evaluator. " + self.eval_mapping[agent_name] + f" Here is the execution summary: {agent_output}"

        response = self.agent_eval.invoke({
            "messages": [{"role": "user", "content": prompt}],
        })

        #print(response['structured_response'])

        return response['structured_response'].achieved_task, response['structured_response'].reasoning


    def _compute_score(self, record: ConvFinQARecord, response: dict, qn_idx: int) -> dict:
        #print(response)
        gen_ans = response['final_answer']
        ans_is_correct, ans_eval_reasoning = self.compare_answers(record.dialogue.executed_answers[qn_idx], gen_ans)
        _agent_score_d = {f"{k}_score": None for k in self.eval_mapping.keys()}
        _agent_reasoning_d = {f"{k}_reasoning": None for k in self.eval_mapping.keys()}
        score_d = {
            "correct": ans_is_correct,
            "answer_evaluation_reasoning": ans_eval_reasoning,
            **_agent_score_d,
            **_agent_reasoning_d,
        }
        running_score = 0

        agent_count = 0
        for agent_name, attempt_num in response['agent_attempts'].items():
            if attempt_num > 0:
                agent_count += 1
                if agent_name == 'FreeAgents':
                    agent_name = 'FreeAgent'
                else:
                    agent_name += 'Agent'

                agent_output = self._get_last_agent_step(response['step_history'], agent_name)

                achieved_task, reasoning = self._eval_reasoning(agent_output)
                agent_score = 0
                if isinstance(agent_output, list):  # is freeagents
                    agent_name = "FreeAgent"
                else:
                    agent_name = agent_output['agent']
                if achieved_task:

                    if agent_name == "FreeAgent":
                        agent_score = 1 / response['agent_attempts']['FreeAgents']
                    else:
                        agent_score = 1 / response['agent_attempts'][agent_name.replace("Agent", "")]
                if ans_is_correct:  # only reward agents if the final answer is correct
                    running_score += agent_score
                score_d[f"{agent_name}_score"] = agent_score
                score_d[f"{agent_name}_reasoning"] = reasoning

        running_score /= agent_count

        score_d['final_score'] = running_score

        return score_d

    def run(self):
        for f in tqdm(self.file_lst):
            if ".json" in f:
                out_fname = f'{results_dir}/scored_{f.replace(".json", ".csv")}'
                if not os.path.isfile(out_fname):
                    print('Scoring file:', f)
                    record_id = convert_fname_to_recordid(f)
                    ds = DatasetDict(dataset_fpath)
                    rec = ds.get_record(record_id, subset='dev')
                    responses = self.load_result(f)
                    results_d = {
                        'correct': [],
                        'answer_evaluation_reasoning': [],
                        **{f"{k}_score": [] for k in self.eval_mapping.keys()},
                        **{f"{k}_reasoning": [] for k in self.eval_mapping.keys()},
                        'final_score': [],
                    }

                    for q_idx, output in enumerate(responses):
                            score = self._compute_score(rec, output, q_idx)
                            print(f'final_score for qn {q_idx}:', score)
                            for k, v in score.items():
                                results_d[k].append(v)
                            for k in set(results_d.keys()).difference(set(score.keys())):
                                results_d[k].append(None)

                    df = pd.DataFrame(results_d)
                    df.to_csv(out_fname, index=False)


def aggregate_scores():
    file_lst = os.listdir(results_dir)
    all_dfs = []
    for f in file_lst:
        if "scored_" in f and ".csv" in f:
            df = pd.read_csv(f'{results_dir}/{f}')
            #df['file'] = f.replace('scored_', '').replace('.csv', '')
            all_dfs.append(df)

    keys_to_retain = [k for k in all_dfs[0].columns if 'score' in k] + [k for k in all_dfs[0].columns if k == 'correct']

    all_scores_df = pd.concat(all_dfs, ignore_index=True)
    all_scores_df = all_scores_df[keys_to_retain]

    # aggregate mean
    meaned = all_scores_df.mean()

    with open(f'{results_dir}/aggregated_scores.txt', 'w') as f:
        f.write(meaned.to_string())


if __name__ == "__main__":
    Scorer().run()
    aggregate_scores()