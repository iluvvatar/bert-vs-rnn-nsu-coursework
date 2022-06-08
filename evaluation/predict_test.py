from pathlib import Path
import numpy as np
from datasets.arrow_dataset import Batch
import os
import json
import torch

from mlpipeline.datasets.nlp import RuREBus
from mlpipeline.datasets.nlp.units import Entity
from mlpipeline.models.nlp.ner import BertForNER, ConvLSTM, ConvSRU, ConvSRUpp
from mlpipeline.processors.nlp.collators import MaxLenSplitCollator, PaddingCollator
from mlpipeline.processors.nlp.labelizers import BILOULabelizer
from mlpipeline.processors.nlp.prediction_postprocessors import Viterbi


def main():
    home_dir = Path(os.getenv('HOME'))
    rurebus_path = home_dir / 'datasets' / 'RuREBus' / 'Preprocessed'
    hf_bert_path = home_dir / 'models' / 'DeepPavlov-rubert-base-cased'

    scratch_dir = Path(os.getenv('SCRATCH_DIR'))
    datasets_dir = scratch_dir / 'datasets'
    models_dir = scratch_dir / 'models'
    models = [
        (BertForNER(bert_name_or_path=hf_bert_path,
                    n_ent_types=8,
                    n_classes=5,
                    n_head_layers=2,
                    dropout=0.1),
         [('rurebus/v0.0.2_2layers_dce0.9/checkpoints/BertForNER_epoch=14_f1-macro=0.56576/model.pt',
           'rurebus/BertForNER_ner_preds')]),
        (ConvLSTM(n_ent_types=8,
                  n_classes=5,
                  cnn_layers=2,
                  cnn_kernels=[1, 3, 5],
                  rnn_layers=4,
                  hid_size=768 // 2,
                  head_layers=2,
                  dropout=0.1),
         [('rurebus/v1.0.0/checkpoints/ConvLSTM_epoch=100_f1-macro=0.41578/model.pt',
           'rurebus/ConvLSTM_ner_preds'),
          ('distil/v0.0.1/checkpoints/ConvLSTM_epoch=19_f1-macro=0.57962/model.pt',
           'rurebus/ConvLSTM_dist_preds'),
          ('fine_tuning_ditil_rurebus/v1.0.0/checkpoints/ConvLSTM_epoch=40_f1-macro=0.54448/model.pt',
           'rurebus/ConvLSTM_dist_ner_preds')]),
        (ConvSRU(n_ent_types=8,
                 n_classes=5,
                 cnn_layers=2,
                 cnn_kernels=[1, 3, 5],
                 rnn_layers=4,
                 hid_size=768 // 2,
                 head_layers=2,
                 dropout=0.1),
         [('rurebus/v1.0.0/checkpoints/ConvSRU_epoch=86_f1-macro=0.43208/model.pt',
           'rurebus/ConvSRU_ner_preds'),
          ('distil/v0.0.1/checkpoints/ConvSRU_epoch=14_f1-macro=0.56631/model.pt',
           'rurebus/ConvSRU_dist_preds'),
          ('fine_tuning_ditil_rurebus/v1.0.0/checkpoints/ConvSRU_epoch=40_f1-macro=0.53861/model.pt',
           'rurebus/ConvSRU_dist_ner_preds')]),
        (ConvSRUpp(n_ent_types=8,
                   n_classes=5,
                   cnn_layers=2,
                   cnn_kernels=[1, 3, 5],
                   rnn_layers=4,
                   hid_size=768 // 2,
                   head_layers=2,
                   dropout=0.1),
         [('rurebus/v1.0.0/checkpoints/ConvSRUpp_epoch=100_f1-macro=0.42305/model.pt',
           'rurebus/ConvSRUpp_ner_preds'),
          ('distil/v0.0.1/checkpoints/ConvSRUpp_epoch=16_f1-macro=0.5802/model.pt',
           'rurebus/ConvSRUpp_dist_preds'),
          ('fine_tuning_ditil_rurebus/v1.0.0/checkpoints/ConvSRUpp_epoch=40_f1-macro=0.53547/model.pt',
           'rurebus/ConvSRUpp_dist_ner_preds')])
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    rurebus = RuREBus.load(rurebus_path)['test']

    collator = MaxLenSplitCollator(
        collate_columns=['tokens', 'spans', 'tokens_ids', 'attention_mask', 'token_type_ids'],
        pk_columns=['doc_id', 'start'],
        unite_columns=['logits'],
        max_len=128
    )

    padder = PaddingCollator(pad_value=0,
                             padding_type='longest',
                             collate_columns=['tokens_ids', 'attention_mask'])

    viterbi = Viterbi(logits_column='logits',
                      word_tokens_indices_column='word_tokens_indices',
                      out_predicted_labels_ids_column='predicted_labels_ids',
                      first_subword_transition_probs=BILOULabelizer.first_subword_transition_probs,
                      middle_subword_transition_probs=BILOULabelizer.middle_subword_transition_probs,
                      last_subword_transition_probs=BILOULabelizer.last_subword_transition_probs,
                      word_transition_probs=BILOULabelizer.word_transition_probs,
                      initial_state=0,
                      pad_label_id=0)

    labelizer = BILOULabelizer(text_column='text',
                               tokens_column='tokens',
                               tokens_spans_column='spans',
                               entities_column='entities',
                               out_labels_column='labels',
                               out_labels_ids_column='labels_ids',
                               predicted_labels_ids_column='predicted_labels_ids',
                               out_predicted_entities_column='predicted_entities',
                               entity_types=rurebus.entity_types,
                               entities_deserialize_fn=Entity.from_brat,
                               entities_serialize_fn=Entity.to_brat)

    def predict_logits_map_fn(model):
        def f(batch: Batch):
            lengths = [len(ex) for ex in batch['tokens_ids']]
            batch = padder.collate(batch)
            with torch.no_grad():
                output = model(tokens_ids=batch['tokens_ids'].to(device),
                               attention_mask=batch['attention_mask'].to(device))
            logits = [out[:lengths[i]] for i, out in enumerate(output)]
            return {'logits': logits}
        return f

    for model, paths in models:
        model.to(device)
        for model_path, output_path in paths:
            model.load(models_dir / model.name / model_path)
            ds = rurebus
            if isinstance(model, BertForNER):
                ds = collator.preprocess(ds)
            ds = ds.map(predict_logits_map_fn(model), batched=True, batch_size=128)
            if isinstance(model, BertForNER):
                ds = collator.postprocess(ds)
            ds = viterbi.postprocess(ds)
            ds = labelizer.postprocess(ds)
            ds.save(datasets_dir / output_path)


if __name__ == '__main__':
    main()
