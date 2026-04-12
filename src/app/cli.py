"""
Main typer app for ConvFinQA
"""

from global_utils.bootstrap_env import load_project_env

load_project_env()

import typer
from langgraph.checkpoint.memory import InMemorySaver
from rich import print as rich_print

from src.agents.planner import Planner
from src.agents.types import State
from src.utils.data.build_vector_db import VectorDB
from src.utils.data.read_dataset import DatasetDict
from global_utils.filepaths import dataset_fpath

app = typer.Typer(
    name="main",
    help="Boilerplate app for ConvFinQA",
    add_completion=True,
    no_args_is_help=True,
)


@app.command()
def chat(
    record_id: str = typer.Argument(..., help="ID of the record to chat about"),
    subset: str = typer.Option("dev", help="Subset of the dataset to use (train or dev)"),
) -> None:
    """Ask questions about a specific record"""
    ds = DatasetDict(dataset_fpath)

    try:
        rec = ds.get_record(record_id, subset='dev')
    except KeyError:
        rich_print(f"[red][bold]Error:[/bold] Record ID '{record_id}' not found in dataset.[/red]")
        return

    rich_print(f"[green][bold]Loaded record:[/bold] {rec.file_id}[/green]")
    history = []
    prev_qn = []

    vector_db = VectorDB(
        record=rec
    )

    rich_print("[yellow][bold]Starting chat session. Type 'exit' or 'quit' to end.[/bold][/yellow]")

    mem = InMemorySaver()
    planner_agent = Planner(rec, mem, 0, verbose_mode='info', self_consistency=True)

    while True:
        message = input(">>> ")

        if message.strip().lower() in {"exit", "quit"}:
            break

        prev_qn.append(message) # only append if not exit or quit

        state = State(question=message, vector_db=vector_db, previous_questions=prev_qn)
        state = planner_agent.run(state)

        response = state.final_answer if state.final_answer else "I'm sorry, I couldn't find an answer to that question."
        rich_print(f"[blue][bold]assistant:[/bold] {response}[/blue]")
        history.append({"user": message, "assistant": response})


if __name__ == "__main__":
    app()