from pathlib import Path
import json
from pprint import pprint
import numpy as np

from mlpipeline.datasets.nlp import RuREBus
from mlpipeline.datasets.nlp.samplers import UnbalancedEntitiesSampler
from mlpipeline.processors.nlp.labelizers import BILOULabelizer
from mlpipeline.datasets.nlp.units import Entity

from tqdm import tqdm


bilou_counts = {'ACT': {'O': 1237541, 'B': 18565, 'L': 18566, 'I': 73477, 'U': 3502}, 'BIN': {'O': 1292794, 'B': 6196, 'L': 6197, 'I': 2116, 'U': 44348}, 'CMP': {'O': 1326399, 'B': 2117, 'L': 2118, 'I': 1126, 'U': 19891}, 'ECO': {'O': 1204391, 'B': 35556, 'L': 35557, 'I': 68276, 'U': 7871}, 'INST': {'O': 1289164, 'B': 12551, 'L': 12552, 'I': 35488, 'U': 1896}, 'MET': {'O': 1236907, 'B': 22339, 'L': 22340, 'I': 67316, 'U': 2749}, 'QUA': {'O': 1328020, 'B': 5358, 'L': 5359, 'I': 2041, 'U': 10873}, 'SOC': {'O': 1286557, 'B': 16515, 'L': 16516, 'I': 27218, 'U': 4845}}
for ent_type in bilou_counts:
    bilou_counts[ent_type] = np.array([bilou_counts[ent_type][label] for label in BILOULabelizer.int2bilou])


def main():
    home_dir = Path('D:/Research/RNNvsBERT')
    rurebus_path = home_dir / 'datasets' / 'RuREBus' / 'Preprocessed'

    rurebus = RuREBus.load(rurebus_path)
    output_file = rurebus_path / 'class_weights_log_sampler.json'

    labelizer = BILOULabelizer(text_column='text',
                               tokens_column='tokens',
                               tokens_spans_column='spans',
                               entities_column='entities',
                               predicted_labels_ids_column='predicted_labels_ids',
                               out_labels_column='labels',
                               out_labels_ids_column='labels_ids',
                               out_predicted_entities_column='predicted_entities',
                               entity_types=rurebus.entity_types,
                               entities_deserialize_fn=Entity.from_brat,
                               entities_serialize_fn=Entity.to_brat)

    weights = {ent_type: {
        label: i for i, label in enumerate(BILOULabelizer.int2bilou)
    } for ent_type in rurebus.entity_types}
    # weights = {split: {ent_type: {
    #     label: i for i, label in enumerate(BILOULabelizer.int2bilou)
    # } for ent_type in rurebus.entity_types} for split in rurebus}
    for split in rurebus:
        dataset = rurebus[split]
        sampler = UnbalancedEntitiesSampler(dataset,
                                            entities_deserialize_fn=Entity.from_brat,
                                            entities_column='entities',
                                            tokens_spans_column='spans',
                                            entity_types_shares='log',
                                            size=10_000)
        for index in tqdm(sampler):
            example = dataset[index]
            labels_ids = example['labels_ids']
            for token_labels_ids in labels_ids:
                for ent_type_id, label_id in enumerate(token_labels_ids):
                    ent_type = labelizer.int2ent_type[ent_type_id]
                    label = BILOULabelizer.int2bilou[label_id]
                    # weights[split][ent_type][label] += 1
                    weights[ent_type][label] += 1
    pprint(weights)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(weights, f)


if __name__ == '__main__':
    main()
