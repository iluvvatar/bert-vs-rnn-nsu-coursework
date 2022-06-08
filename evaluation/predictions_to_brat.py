from pathlib import Path
import os
from mlpipeline.datasets.nlp import RuREBus

home_dir = Path(os.getenv('HOME'))
dataset_dir = home_dir / 'datasets' / 'RuREBus'
dir_in = dataset_dir / 'models_predictions_united_docs'
dir_out = dataset_dir / 'models_predictions_united_docs_brat'

for ds_name in dir_in.iterdir():
    if ds_name.is_dir():
        RuREBus.load(dir_in / ds_name).save_brat(dir_out / ds_name, entities_column='predicted_entities')
