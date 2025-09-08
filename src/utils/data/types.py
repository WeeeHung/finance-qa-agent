from typing import Dict, List, Union

from pydantic import BaseModel, Field, field_validator, ConfigDict
import pandas as pd

pd.set_option('display.max_columns', None)


class Document(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pre_text: str = Field(description="The text before the table in the document")
    post_text: str = Field(description="The text after the table in the document")
    table: pd.DataFrame = Field(
        description="The table of the document as a dictionary "
    )

    @property
    def last_pre_text_line(self) -> str:
        return [line for line in self.pre_text.split('\n') if line.strip()][-1]

    @property
    def first_post_text_line(self) -> str:
        return [line for line in self.post_text.split('\n') if line.strip()][0]

    @property
    def table_context(self) -> str:
        return f"{self.last_pre_text_line}\n{self.first_post_text_line}"

    @property
    def combined_text(self) -> str:
        return f"{self.pre_text}\n{self.post_text}"

    @field_validator('table', mode='before')
    @classmethod
    def convert_table(cls, table: dict[str, dict[str, float | str | int]]) -> pd.DataFrame:
        return pd.DataFrame.from_dict(table)


class Dialogue(BaseModel):
    conv_questions: list[str] = Field(
        description="The questions in the conversation dialogue, originally called 'dialogue_break'"
    )
    conv_answers: list[str] = Field(
        description="The answers to each question turn, derived from 'answer_list' and original FinQA answers"
    )
    turn_program: list[str] = Field(
        description="The DSL turn program for each question turn"
    )
    executed_answers: list[float | str] = Field(
        description="The golden program execution results for each question turn"
    )
    qa_split: list[bool] = Field(
        description="This field indicates the source of each question turn - 0 if from the decomposition of the first FinQA question, 1 if from the second. For the Type I simple conversations, this field is all 0s."
    )


class Features(BaseModel):
    num_dialogue_turns: int = Field(
        description="The number of turns in the dialogue, calculated from the length of conv_questions"
    )
    has_type2_question: bool = Field(
        description="Whether the dialogue has a type 2 question, calculated if qa_split contains a 1 this will return true"
    )
    has_duplicate_columns: bool = Field(
        description="Whether the table has duplicate column names not fully addressed during cleaning. We suffix the duplicate column headers with a number if there was no algorithmic fix. e.g. 'Revenue (1)' or 'Revenue (2) "
    )
    has_non_numeric_values: bool = Field(
        description="Whether the table has non-numeric values"
    )


class ConvFinQARecord(BaseModel):
    id: str = Field(description="The id of the record")
    doc: Document = Field(description="The document")
    dialogue: Dialogue = Field(description="The conversational dialogue")
    features: Features = Field(description="The features of the record, created by Tomoro to help you understand the data")

    @property
    def file_id(self) -> str:
        return self.id.replace('/', '-')


class RetrievedTextMetadata(BaseModel):
    type: str = Field(description="The type of the retrieved context, either 'text' or 'table_cell'")
    record_id: str = Field(description="The id of the ConvFinQARecord the text was retrieved from")


class RetrievedTableMetadata(BaseModel):
    type: str = Field(description="The type of the retrieved context, either 'text' or 'table_cell'")
    record_id: str = Field(description="The id of the ConvFinQARecord the table was retrieved from")
    column: str = Field(description="Column name of table cell")
    value: str | float | int = Field(description="Value of table cell")
    context: str = Field(description="The table context consists of the last line of the pre-table text and the first line of the post-table text.")


class RetrievedItem(BaseModel):
    context: str = Field(description="The retrieved context from the vector database")
    metadata: Dict = Field(description="The metadata associated with the retrieved context")
