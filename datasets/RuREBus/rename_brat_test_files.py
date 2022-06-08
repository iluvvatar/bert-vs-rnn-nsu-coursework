import os
import json
import re
from pathlib import Path
from typing import Iterator, Iterable

from mlpipeline.datasets.nlp.brat import BratDoc


def ann_files(data_folder_path: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(data_folder_path):
        for filename in filenames:
            file = Path(dirpath) / filename
            if file.suffix == '.ann':
                yield file


home_dir = Path(os.getenv('HOME'))
dataset_dir = home_dir / 'datasets' / 'RuREBus'
dir_in = dataset_dir / 'data' / 'test_data'
dir_out = dataset_dir / 'real_test_docs_brat'
for i, file in enumerate(ann_files(dir_in)):
    new_name = f'{i}.ann'
    BratDoc.load_brat(file).save_brat(dir_out / new_name, save_text=False, save_relations=False, save_links=False)
