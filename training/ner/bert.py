import torch
from pathlib import Path
import os

from mlpipeline.models.nlp.ner import BertForNER
from training.ner.pipeline import ner_pipeline


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert_path = Path(os.getenv('HOME')) / 'models' / 'DeepPavlov-rubert-base-cased'
    model = BertForNER(bert_name_or_path=bert_path,
                       n_ent_types=8,
                       n_classes=5,
                       n_head_layers=2,
                       dropout=0.1).to(device)
    ner_pipeline(model, device, experiment_name='v0.0.1_dce_0.9_adj', is_bert=True)


if __name__ == '__main__':
    main()
