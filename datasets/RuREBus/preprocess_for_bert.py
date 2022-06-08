import os
from pathlib import Path

from mlpipeline.datasets.nlp import RuREBus
from mlpipeline.processors.nlp.collators import MaxLenSplitCollator


def main():
    home_dir = Path(os.getenv('HOME'))
    dataset_dir = home_dir / 'datasets' / 'RuREBus'
    dataset_path = dataset_dir / 'Preprocessed'
    save_path = dataset_dir / 'PreprocessedForBert'

    dataset = RuREBus.load(dataset_path)

    collator = MaxLenSplitCollator(
        collate_columns=['tokens', 'spans', 'tokens_ids', 'attention_mask',
                         'token_type_ids', 'labels', 'labels_ids'],
        pk_columns=['doc_id', 'start'],
        unite_columns=['logits'],
        max_len=128
    )

    dataset = collator.preprocess(dataset, use_cached=True)

    dataset.save(save_path)


if __name__ == '__main__':
    main()
