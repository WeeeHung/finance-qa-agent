import sys
import traceback
from typing import List, Union, Dict, Literal, Optional, Mapping

from pydantic import BaseModel, Field, field_serializer, field_validator

from src.utils.data.build_vector_db import VectorDB

def _serialize_exception(e: Union[Exception, BaseException, type]) -> dict[str, str]:
    if isinstance(e, type) and issubclass(e, BaseException):
        inst = e("")  # type: ignore[arg-type]
        tb = "".join(traceback.format_exception(inst.__class__, inst, inst.__traceback__))
        return {"type": inst.__class__.__name__, "message": str(inst), "traceback": tb}

    if isinstance(e, BaseException):
        tb = "".join(traceback.format_exception(e.__class__, e, e.__traceback__))
        return {"type": e.__class__.__name__, "message": str(e), "traceback": tb}

        # Fallback: not an exception at all
    return {"type": type(e).__name__, "message": str(e), "traceback": ""}


def _serialize_entry(obj: dict) -> Union[list, dict]:
    if isinstance(obj, type) and issubclass(obj, BaseException):
        return _serialize_exception(obj)

    if isinstance(obj, BaseException):
        return _serialize_exception(obj)

    if isinstance(obj, Mapping):
        # Ensure keys are strings for JSON compatibility
        return {str(k): _serialize_entry(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_serialize_entry(x) for x in obj]


class State(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }

    previous_questions: List[str] = Field(default_factory=list, description="A list of previous questions in the "
                                                                            "conversation to provide context for "
                                                                            "the current question.")
    question: str = Field(default=None, description="The user's original question.")
    final_question: str = Field(default=None, description="The clarified version of the user's question after "
                                                          "rephrasing to avoid ambiguity.")
    can_answer_directly: bool = Field(default=None,
                                      description="Indicates if the agent can answer the question without additional "
                                                  "computations, based only on the provided information.")
    subproblems: List[str] = Field(default=None,
                                   description="A list of subproblems derived from the original question, "
                                               "if it cannot be answered directly.")
    subproblems_dict_lst: List[Dict[str, Union[str, str]]] = Field(default=None,
                                                description="A list of dictionaries containing each subproblem and "
                                                            "its corresponding answer or error message.")
    final_answer: str = Field(default=None,
                              description="The final answer based on the retrieved information and any "
                                         "necessary calculations.")
    errors: Dict[str, Exception] = Field(default_factory=dict, description="Any error encountered during processing.")
    vector_db: VectorDB = Field(default=None, description="The vector database instance used for information retrieval.")
    step_history: List[Union[Dict[str, Union[str, dict]], List[Dict[str, Union[str, dict]]]]] = Field(default_factory=list,
                                               description="A list of dictionaries capturing the "
                                                           "history of each step taken, including "
                                                           "intermediate questions, answers, and any "
                                                           "errors encountered.")
    current_subproblem: Optional[str] = Field(default=None,
                                              description="The subproblem currently being processed, if applicable.")
    current_subproblem_answer: Optional[str] = Field(default=None,
                                                      description="The answer to the current subproblem being processed, "
                                                                  "if applicable.")
    free_agent_idx: Optional[int] = Field(default=None,
                                            description="The index of the current free agent processing the subproblem, "
                                                        "if applicable.")
    computed_answer: Optional[float] = Field(default=None,
                                             description="The computed answer from the python code generated from the "
                                                         "subproblems' answers, if applicable.")
    agent_attempts: Optional[Dict[str, int]] = Field(default_factory=dict,
                                                     description="A dictionary tracking the number of attempts for each "
                                                                 "agent type.")

    @field_serializer('step_history')
    def serialize_step_history(self, sh):
        if sh is None:
            return None

        serialized_history = []
        for entry in sh:
            serialized_history.append(_serialize_entry(entry))

    @field_serializer('errors')
    def serialize_errors(self, e):
        if e is None:
            return None

        if isinstance(e, Mapping):
            return {
                str(k): (
                    _serialize_exception(v)
                    if isinstance(v, BaseException)
                       or (isinstance(v, type) and issubclass(v, BaseException))
                    else v  # keep non-exception values as-is
                )
                for k, v in e.items()
            }
            # If someone passed a single exception or something else by mistake:
        return _serialize_exception(e)


    @field_validator('step_history', mode='before')
    @classmethod
    def serialize_step_history(cls, sh):
        if sh is None:
            return None

        serialized_history = []
        for entry in sh:
            serialized_history.append(_serialize_entry(entry))

    @field_validator('errors', mode='before')
    @classmethod
    def serialize_errors(cls, e):
        if e is None:
            return None

        if isinstance(e, Mapping):
            return {
                str(k): (
                    _serialize_exception(v)
                    if isinstance(v, BaseException)
                       or (isinstance(v, type) and issubclass(v, BaseException))
                    else v  # keep non-exception values as-is
                )
                for k, v in e.items()
            }
            # If someone passed a single exception or something else by mistake:
        return _serialize_exception(e)

    def __str__(self):
        return self.model_dump_json(indent=2, exclude={'vector_db'})