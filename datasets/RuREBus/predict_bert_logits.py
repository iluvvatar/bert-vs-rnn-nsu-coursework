from pathlib import Path
from datasets.arrow_dataset import Batch
import os
import torch

from mlpipeline.datasets.nlp import RuREBus
from mlpipeline.models.nlp.ner import BertForNER
from mlpipeline.processors.nlp.collators import MaxLenSplitCollator, PaddingCollator


def main():
    home_dir = Path(os.getenv('HOME'))
    datasets_dir = home_dir / 'datasets'

    rurebus_path = datasets_dir / 'RuREBus' / 'PreprocessedRawTxt100000'
    hf_bert_path = home_dir / 'models' / 'DeepPavlov-rubert-base-cased'
    bert_state_dict_path = Path(os.getenv('SCRATCH_DIR')) \
                           / 'models' / 'BertForNER' / 'rurebus' \
                           / 'v0.0.2_2layers_dce0.9' / 'checkpoints' \
                           / 'BertForNER_epoch=14_f1-macro=0.56576' \
                           / 'model.pt'
    output_dir = datasets_dir / 'RuREBus' / 'RawTxt100000BertPredictions'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BertForNER(bert_name_or_path=hf_bert_path,
                       n_ent_types=8,
                       n_classes=5,
                       n_head_layers=2,
                       dropout=0.1).to(device)
    model.load(bert_state_dict_path)

    rurebus = RuREBus.load(rurebus_path)

    collator = MaxLenSplitCollator(
        collate_columns=['tokens', 'spans', 'tokens_ids', 'attention_mask', 'token_type_ids'],
        pk_columns=['doc_id', 'start'],
        unite_columns=['logits'],
        max_len=128
    )

    padder = PaddingCollator(pad_value=0,
                             padding_type='max_length',
                             max_len=128,
                             collate_columns=['tokens_ids', 'attention_mask'])

    def predict_logits_map_fn(batch: Batch):
        lengths = [len(ex) for ex in batch['tokens_ids']]
        batch = padder.collate(batch)
        with torch.no_grad():
            output = model(tokens_ids=batch['tokens_ids'].to(device),
                           attention_mask=batch['attention_mask'].to(device))
        logits = [out[:lengths[i]] for i, out in enumerate(output)]
        return {'logits': logits}

    rurebus = collator.preprocess(rurebus)
    rurebus = rurebus.map(predict_logits_map_fn, batched=True, batch_size=128)
    rurebus = collator.postprocess(rurebus)

    rurebus.save(output_dir)


if __name__ == '__main__':
    main()
