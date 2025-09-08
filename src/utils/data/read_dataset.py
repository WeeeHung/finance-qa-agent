import json
from typing import List, Dict, Any
from typing_extensions import Literal
from src.utils.data.types import ConvFinQARecord


class Dataset:
    def __init__(self, raw_ds: List[Dict[str, Any]]):
        self._data_lst: List[ConvFinQARecord] = self._process_data(raw_ds)
        self._data_idx: Dict[str, ConvFinQARecord] = self._build_data_idx()

    def __len__(self):
        return len(self._data_lst)

    @staticmethod
    def _process_data(raw_data: List[Dict[str, Any]]) -> List[ConvFinQARecord]:
        return [ConvFinQARecord(**record) for record in raw_data]

    def _build_data_idx(self) -> Dict[str, ConvFinQARecord]:
        return {record.id: record for record in self._data_lst}

    def get_records(self) -> List[ConvFinQARecord]:
        return self._data_lst

    def get_record(self, record_id: str) -> ConvFinQARecord:
        return self._data_idx[record_id]


class DatasetDict:
    def __init__(self, fpath: str):
        self._fpath: str = fpath
        self._raw_data: Dict[str, Any] = self._read_data()
        self._data_dict: Dict[str, Dataset] = {}
        self._process_data()

    def __len__(self):
        return sum(len(ds) for ds in self._data_dict.values())

    def _read_data(self) -> Dict[str, List[ConvFinQARecord]]:
        with open(self._fpath, 'r') as f:
            dataset = json.load(f)
        return dataset

    def _process_data(self):
        for subset in self._raw_data.keys():
            self._data_dict[subset] = Dataset(self._raw_data[subset])

    def get_records(self, subset: Literal['train', 'dev'] = 'train') -> List[ConvFinQARecord]:
        return self._data_dict[subset].get_records()

    def get_record(self, idx: str, subset: Literal['train', 'dev'] = 'train') -> ConvFinQARecord:
        return self._data_dict[subset].get_record(idx)

    def get_subset(self, subset: Literal['train', 'dev']) -> Dataset:
        return self._data_dict[subset]




if __name__ == "__main__":
    from src.utils.filepaths import dataset_fpath
    #print(len(Dataset(dataset_fpath)))
    #doc = Dataset(dataset_fpath).get_record('Single_JKHY/2009/page_28.pdf-3').doc
    #print(doc.table)
    ds = DatasetDict(dataset_fpath)
    doc = ds.get_record('Single_JKHY/2009/page_28.pdf-3').doc
    print(doc.table)
    print(len(ds))


