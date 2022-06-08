import os
from pathlib import Path

from mlpipeline.utils import PathLike
from mlpipeline.datasets.nlp.rurebus import RuREBus


def download(raw_txt, folder_path: PathLike):
    folder_path = Path(folder_path)
    dataset = RuREBus.load_from_hub(raw_txt)
    dataset.save(folder_path)


if __name__ == '__main__':
    home_dir = Path(os.getenv('HOME'))
    dataset_dir = home_dir / 'datasets' / 'RuREBus'
    download(False, dataset_dir / 'HuggingFaceLocal')
    download(True, dataset_dir / 'HuggingFaceLocalRawTxt')
