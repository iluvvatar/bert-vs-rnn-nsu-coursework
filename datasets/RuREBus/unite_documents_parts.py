# =============================================================================
# Unite several parts of https://github.com/dialogue-evaluation/RuREBus dataset
# =============================================================================
import os
from pathlib import Path
import re
from collections import defaultdict
from typing import Iterator

from mlpipeline.utils import PathLike


def ann_files(data_folder_path: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(data_folder_path):
        for filename in filenames:
            file = Path(dirpath) / filename
            if file.suffix == '.ann':
                yield file


def unite_rurebus_documents_parts(data_folder_path: PathLike,
                                  to_folder_path: PathLike):
    data_folder_path = Path(data_folder_path)
    to_folder_path = Path(to_folder_path)
    new_paths = defaultdict(list)
    for file in ann_files(data_folder_path):
        path = Path(file)
        parent = path.relative_to(data_folder_path).parent
        if path.match('*_part[_0-9]*.ann'):
            new_name = re.sub(r'_part[_0-9]*$', '', path.stem)
            new_path = (to_folder_path / parent / new_name).with_suffix('.ann')
        else:
            new_path = to_folder_path / parent / path.name
        new_paths[new_path].append(path)
        if not new_path.parent.exists():
            new_path.parent.mkdir(parents=True)
    for new_path, paths in new_paths.items():
        paths.sort(key=lambda p: int(p.stem.split('part')[-1].replace('_', '')))
        with open(new_path, 'w', encoding='utf-8') as f_out_ann, \
                open(new_path.with_suffix('.txt'), 'w', encoding='utf-8') as f_out_txt:
            for path in paths:
                with open(path, encoding='utf-8') as f_in_ann:
                    print(f_in_ann.read(), file=f_out_ann)
                with open(path.with_suffix('.txt'), encoding='utf-8') as f_in_txt:
                    print(f_in_txt.read(), file=f_out_txt)


if __name__ == '__main__':
    home_dir = Path(os.getenv('HOME'))
    dataset_dir = home_dir / 'datasets' / 'RuREBus'
    unite_rurebus_documents_parts(
        dataset_dir / 'data' / 'train_data',
        dataset_dir / 'data' / 'train_data_united'
    )
    unite_rurebus_documents_parts(
        dataset_dir / 'data' / 'test_data',
        dataset_dir / 'data' / 'test_data_united'
    )

