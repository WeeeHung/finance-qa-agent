import os
from typing import Tuple, List, Dict, Union

import pandas as pd
from vectordb import Memory

from src.utils.data.read_dataset import DatasetDict
from global_utils.filepaths import vector_db_dir, dataset_fpath
from src.utils.data.types import ConvFinQARecord


class VectorDB:
    def __init__(self, record: ConvFinQARecord):
        if not os.path.exists(vector_db_dir):
            os.makedirs(vector_db_dir)
        mem_fpath = f'{vector_db_dir}/{record.file_id}.mem'
        self._mem = Memory(memory_file=mem_fpath)
        self._record = record

        if not os.path.isfile(mem_fpath):
            self._build()

    def _serialize_table(self, table: pd.DataFrame) -> Tuple[List[str], List[Dict[str, str]]]:
        elements = []
        metadatas = []
        # table string = row_name + col_name
        table = table.transpose() # so that when we iterate over rows we are iterating over the og cols
        for row_name, row in table.iterrows():
            _row_elements = []
            for col_name, value in row.items():
                try:
                    _row_elements.append(f"{row_name} {col_name} is {float(value):.2f}")
                except (ValueError, TypeError):
                    continue
            _row_elements_str = "; ".join(_row_elements)
            elements.append(_row_elements_str)
            metadatas.append({
                "type": "table_cell",
                "record_id": self._record.id,
                "column": row_name,
                "value": value,
                "context": self._record.doc.table_context
            })
        return elements, metadatas

    def _build(self):
        elements_to_add = []
        metadata_to_add = []

        # Add the combined text of the document
        elements_to_add.append(self._record.doc.combined_text)
        metadata_to_add.append({
            "type": "text",
            "record_id": self._record.id,
        })

        # Add the table cells
        elements, metadatas =  self._serialize_table(self._record.doc.table)
        elements_to_add.extend(elements)
        metadata_to_add.extend(metadatas)

        self._mem.save(elements_to_add, metadata_to_add)

    @staticmethod
    def _sort_chunk_by_type(results: List[Dict[str, Union[str, dict]]]) -> tuple[
        list[dict[str, str | dict]], list[dict[str, str | dict]]]:
        #print(results)
        text_chunks = [res for res in results if res['metadata']['type'] == 'text']
        table_chunks = [res for res in results if res['metadata']['type'] == 'table_cell']
        return text_chunks, table_chunks

    def _serialize_results(self, results: List[Dict[str, Union[str, dict]]]) -> str:
        text_chunks, table_chunks = self._sort_chunk_by_type(results)
        # Prioritize table chunks first, then text chunks
        if not table_chunks and not text_chunks:
            return "No relevant information found."

        return_str = f"File Title: {self._record.id}\n"
        if table_chunks:
            return_str += f"\nRelevant table information:\n" + "\n".join([f"- {chunk['chunk']}" for chunk in table_chunks]) + \
                    f"\nColumn name: {table_chunks[0]['metadata']['column']}, Value: {float(table_chunks[0]['metadata']['value']):.2f}" + \
                    f"\nPre and post lines around table: {table_chunks[0]['metadata']['context']}"
                    # no worry about value error since it is already filtered in _serialize_table
        if text_chunks:
            return_str += f"\nRelevant textual information:\n" + "\n".join([f"- {chunk['chunk']}" for chunk in text_chunks])

        return return_str

    def query(self, query: str, top_n: int = 5) -> str:
        results = self._mem.search(query, top_n=top_n)
        return self._serialize_results(results)


if __name__ == "__main__":
    dataset = DatasetDict(dataset_fpath)
    record = dataset.get_record('Double_ADBE/2018/page_86.pdf', subset='dev')
    vdb = VectorDB(record)
    retrieved = vdb.query(record.dialogue.conv_questions[0], top_n=5)
    print(retrieved)


