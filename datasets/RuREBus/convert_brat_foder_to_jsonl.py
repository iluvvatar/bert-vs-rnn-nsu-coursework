import os
import json
import re
from pathlib import Path
from typing import Iterator, Iterable

from mlpipeline.datasets.nlp.brat import BratDoc
from mlpipeline.utils import PathLike


def ann_files(data_folder_path: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(data_folder_path):
        for filename in filenames:
            file = Path(dirpath) / filename
            if file.suffix == '.ann':
                yield file


def brat_folder_to_jsonl(brat_folder_path: PathLike,
                         jsonl_file_path: PathLike,
                         del_keys: Iterable = ()):
    brat_folder_path = Path(brat_folder_path)
    jsonl_file_path = Path(jsonl_file_path)
    with open(jsonl_file_path, 'w', encoding='utf-8') as f_jsonl:
        for i, file in enumerate(ann_files(brat_folder_path)):
            doc = {'id': i}
            doc.update(BratDoc.load_brat(file).to_dict())
            for k in del_keys:
                del doc[k]
            spaces_pattern = r'[\u00A0\u1680\u180e\u2000-\u200b\u202f\u205f\u3000\ufeff\xa0]'
            json_line = json.dumps(doc, ensure_ascii=False)
            json_line = re.sub(spaces_pattern, ' ', json_line)
            print(json_line, file=f_jsonl)


if __name__ == '__main__':
    home_dir = Path(os.getenv('HOME'))
    dataset_dir = home_dir / 'datasets' / 'RuREBus'
    brat_folder_to_jsonl(
        dataset_dir / 'data/train_data',
        dataset_dir / 'HuggingFaceHub/data/train.jsonl',
        del_keys=('links',)
    )
    brat_folder_to_jsonl(
        dataset_dir / 'data/test_data',
        dataset_dir / 'HuggingFaceHub/data/test.jsonl',
        del_keys=('links',)
    )
