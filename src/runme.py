import os

from langgraph.checkpoint.memory import InMemorySaver
from tqdm import tqdm

from src.agents.planner import Planner
from src.agents.types import State
from src.utils.data.build_vector_db import VectorDB
from src.utils.data.read_dataset import DatasetDict
from src.utils.filepaths import dataset_fpath, results_dir


def run():
    ds = DatasetDict(dataset_fpath)
    for rec in tqdm(ds.get_subset('dev').get_records()[:10]): # get only 8 records for testing
        if not os.path.isfile(f'{results_dir}/{rec.file_id}.json'):
            print("Processing record:", rec.file_id)
            # rec = ds.get_record('Single_ADBE/2018/page_86.pdf-1', subset='dev')
            vector_db = VectorDB(
                record=rec
            )
            mem = InMemorySaver()
            planner_agent = Planner(rec, mem, 0, verbose_mode='info', agent_retry_limit=5,
                                    self_consistency=True)
            pq = []

            for idx, q in enumerate(rec.dialogue.conv_questions):
                state = State(question=q, vector_db=vector_db, previous_questions=pq)
                state = planner_agent.run(state)
                print("=== Planner finished execution ===")
                with open(f'{results_dir}/{rec.file_id}.json', 'a') as f:
                    f.write(f"{state}\n")
                pq.append(q)

            """try:


            except Exception as e:
                with open(f'{results_dir}/planner_error.log', 'a') as f:
                    f.write(f"Error: {type(e).__name__}: {e}\n")"""
        else:
            print(f"Skipping record {rec.file_id}, already processed.")


if __name__ == "__main__":
    run()