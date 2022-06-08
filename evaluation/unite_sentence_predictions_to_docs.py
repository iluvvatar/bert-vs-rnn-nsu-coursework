from pathlib import Path
import numpy as np
from datasets.arrow_dataset import Batch
import os
import json
import torch
from collections import defaultdict

from mlpipeline.datasets.nlp import RuREBus
from mlpipeline.datasets.nlp.units import Entity
from mlpipeline.processors.nlp.collators import MaxLenSplitCollator, PaddingCollator
from mlpipeline.processors.nlp.labelizers import BILOULabelizer
from mlpipeline.processors.nlp.sentenizers import NatashaSentenizer
from mlpipeline.processors.nlp.prediction_postprocessors import Viterbi


def main():
    rurebus_path = Path('D:/Research/RNNvsBERT/datasets/RuREBus/HuggingFaceLocal')
    rurebus = RuREBus.load(rurebus_path)['test'].sort('id')
    datasets_dir = Path('D:/Research/RNNvsBERT/datasets/RuREBus/models_predictions')

    sentenizer = NatashaSentenizer(text_column='text',
                                   doc_id_column='id',
                                   out_text_column='text',
                                   out_start_column='start',
                                   out_stop_column='stop',
                                   out_doc_id_column='doc_id',
                                   pred_entities_column='predicted_entities',
                                   postprocess_remove_columns=['start', 'stop', 'doc_id'],
                                   entities_deserialize_fn=Entity.from_brat,
                                   entities_serialize_fn=Entity.to_brat)
    for ds_name in os.listdir(datasets_dir):
        ds = RuREBus.load(datasets_dir / ds_name)
        rm_columns = set(ds.column_names) - {'text', 'predicted_entities', 'start', 'stop', 'doc_id'}
        ds = ds.remove_columns(list(rm_columns))
        # print(ds)
        ds = sentenizer.postprocess(ds, use_cached=False)
        # print(ds)
        ds = ds.sort('id')
        # if 'entities' in ds:
        #     raise Exception(ds['entities'])
        ds = ds.add_column('entities', rurebus['entities'])
        # for i, example in enumerate(ds):
        #     if i == 3:
        #         break
        #     # print(example['text'])
        #     # original_entities = [Entity.from_brat(e) for e in example['original_entities']]
        #     entities = [Entity.from_brat(e) for e in example['entities']]
        #     predicted_entities = [Entity.from_brat(e) for e in example['predicted_entities']]
        #     # original_entities.sort(key=lambda e: e.start)
        #     entities.sort(key=lambda e: e.start)
        #     predicted_entities.sort(key=lambda e: e.start)
        #     # print(original_entities)
        #     # print(entities)
        #     # print(predicted_entities)
        #     # original_entities = {(e.type, e.start, e.stop) for e in original_entities}
        #     entities = {(e.type, e.start, e.stop) for e in entities}
        #     predicted_entities = {(e.type, e.start, e.stop) for e in predicted_entities}
        #     print(len(entities & predicted_entities), entities & predicted_entities)
        #     print(len(entities - predicted_entities), entities - predicted_entities)
        #     print(len(predicted_entities - entities), predicted_entities - entities)
        #     print('=================================================')
        ds.save(datasets_dir / '..' / 'models_predictions_united_docs' / ds_name)


if __name__ == '__main__':
    main()
