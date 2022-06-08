import torch
from pathlib import Path
import os

from mlpipeline.models.nlp.ner import BertForNER
from training.siam.pipeline import siam_pipeline


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert_path = Path(os.getenv('HOME')) / 'models' / 'DeepPavlov-rubert-base-cased'
    model = BertForNER(bert_name_or_path=bert_path,
                       n_ent_types=8,
                       n_classes=5,
                       return_hidden_states=True).to(device)
    siam_pipeline(model, device, experiment_name='v0.0.1', one_stage=False,
                  checkpoint_name='BertForNER_epoch=20_f1-macro=0.0')


if __name__ == '__main__':
    main()
