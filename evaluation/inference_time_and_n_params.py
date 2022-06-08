import torch
from pathlib import Path
import os
from time import time
import json
from torch.utils.data import DataLoader

from mlpipeline.datasets.nlp import RuREBus
from mlpipeline.processors.nlp.collators import PaddingCollator
from mlpipeline.models.nlp.ner import BertForNER, ConvLSTM, ConvSRU, ConvSRUpp
from mlpipeline.evaluators.ner import NerRnnTimeEvaluator

from tqdm import tqdm


def main():
    bert_path = Path(os.getenv('HOME')) / 'models' / 'DeepPavlov-rubert-base-cased'
    models = [
        BertForNER(bert_name_or_path=bert_path,
                   n_ent_types=8,
                   n_classes=5,
                   n_head_layers=2,
                   dropout=0.1),
        ConvLSTM(n_ent_types=8,
                 n_classes=5,
                 cnn_layers=2,
                 cnn_kernels=[1, 3, 5],
                 rnn_layers=4,
                 hid_size=768//2,
                 head_layers=2,
                 dropout=0.1),
        ConvSRU(n_ent_types=8,
                n_classes=5,
                cnn_layers=2,
                cnn_kernels=[1, 3, 5],
                rnn_layers=4,
                hid_size=768//2,
                head_layers=2,
                dropout=0.1),
        ConvSRUpp(n_ent_types=8,
                  n_classes=5,
                  cnn_layers=2,
                  cnn_kernels=[1, 3, 5],
                  rnn_layers=4,
                  hid_size=768//2,
                  head_layers=2,
                  dropout=0.1)
    ]
    results = [{'name': model.name} for model in models]

    rurebus_path = Path(os.getenv('HOME')) / 'datasets' / 'RuREBus' / 'Preprocessed'
    rurebus_bert_path = Path(os.getenv('HOME')) / 'datasets' / 'RuREBus' / 'PreprocessedForBert'
    output_file_path = Path(os.getenv('SCRATCH_DIR')) / 'models' / 'coursework_inference_time2.jsonl'
    if not output_file_path.parent.exists():
        output_file_path.mkdir(parents=True)
    rurebus_test = RuREBus.load(rurebus_path)['test']
    rurebus_bert_test = RuREBus.load(rurebus_bert_path)['test']
    collator = PaddingCollator(
        collate_columns=['tokens_ids', 'attention_mask', 'labels_ids'],
        pad_value=0,
        padding_type='longest'
    )
    evaluator = NerRnnTimeEvaluator(
        batch_size=1,
        collate_fn=collator.collate,
        tokens_ids_column='tokens_ids',
        attention_mask_column='attention_mask',
        verbose=True
    )

    for i, model in enumerate(models):
        model.freeze_embeddings()
        ds = rurebus_bert_test if 'bert' in model.name.lower() else rurebus_test
        results[i].update(evaluator.evaluate(model, ds, devices=['cpu', 'cuda']))
        results[i]['parameters_number'] = sum(p.numel()
                                              for p in model.parameters()
                                              if p.requires_grad)
        # loader = DataLoader(ds, batch_size=1, collate_fn=collator.collate)
        # with torch.no_grad():
        #     for device in devices:
        #         model.to(device)
        #         total_time = 0
        #         for batch in tqdm(loader, desc=f'{model.name}-{device}'):
        #             tokens_ids = batch['tokens_ids'].to(device)
        #             attention_mask = batch['attention_mask'].to(device)
        #             start_time = time()
        #             logits = model(tokens_ids=tokens_ids,
        #                            attention_mask=attention_mask)
        #             total_time += time() - start_time
        #         results[i][f'{device}-time-per-sample'] = total_time / len(rurebus_test)
        print(results)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for result in results:
                print(json.dumps(result), file=f)


if __name__ == '__main__':
    main()
