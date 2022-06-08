import os
from pathlib import Path
import numpy as np

from mlpipeline.datasets.nlp import RuREBus
from mlpipeline.datasets.nlp.units import Entity
from mlpipeline.datasets.nlp.clean_text import clean_text

from mlpipeline.processors.numerator import Numerator
from mlpipeline.processors.nlp.sentenizers import NatashaSentenizer
from mlpipeline.processors.nlp.tokenizers import RuBERTTokenizer


def main():
    subset_size = 100_000
    home_dir = Path(os.getenv('HOME'))
    dataset_dir = home_dir / 'datasets' / 'RuREBus'
    dataset_path = dataset_dir / 'HuggingFaceLocalRawTxt'
    save_path = dataset_dir / f'PreprocessedRawTxt{subset_size}'
    tokenizer_path = 'DeepPavlov/rubert-base-cased'

    dataset = RuREBus.load(dataset_path)['raw_txt']

    numerator = Numerator(out_id_column='id')
    sentenizer = NatashaSentenizer(text_column='text',
                                   doc_id_column='id',
                                   # entities_column='entities',
                                   # remove_columns=['id', 'relations'],
                                   remove_columns=['id', 'region', 'district', 'title'],
                                   out_text_column='text',
                                   out_start_column='start',
                                   out_stop_column='stop',
                                   out_doc_id_column='doc_id',
                                   # out_entities_column='entities',
                                   entities_deserialize_fn=Entity.from_brat,
                                   entities_serialize_fn=Entity.to_brat)
    tokenizer = RuBERTTokenizer(text_column='text',
                                pretrained_tokenizer_path=tokenizer_path,
                                out_tokens_column='tokens',
                                out_spans_column='spans',
                                out_tokens_ids_column='tokens_ids',
                                out_word_tokens_indices_column='word_tokens_indices',
                                out_attention_mask_column='attention_mask',
                                out_token_type_ids_column='token_type_ids')
    # labelizer = BILOULabelizer(text_column='text',
    #                            tokens_column='tokens',
    #                            tokens_spans_column='spans',
    #                            entities_column='entities',
    #                            predicted_labels_ids_column='predicted_labels_ids',
    #                            out_labels_column='labels',
    #                            out_labels_ids_column='labels_ids',
    #                            out_predicted_entities_column='predicted_entities',
    #                            entity_types=dataset.entity_types,
    #                            special_tokens=tokenizer.special_tokens,
    #                            entities_deserialize_fn=Entity.from_brat,
    #                            entities_serialize_fn=Entity.to_brat)
    # collator = MaxLenSplitCollator(
    #     collate_columns=['tokens', 'spans', 'tokens_ids', 'attention_mask',
    #                      'token_type_ids', 'labels', 'labels_ids'],
    #     pk_columns=['doc_id', 'start'],
    #     unite_columns=['logits'],
    #     max_len=128
    # )

    dataset = dataset.map(lambda example: {'text': clean_text(example['text'])}, batched=False)
    dataset = numerator.preprocess(dataset, use_cached=True)
    dataset = sentenizer.preprocess(dataset, use_cached=True)
    dataset = dataset.filter(lambda example: len(example['text']) <= 512, batched=False)
    indices = np.arange(len(dataset))
    indices = np.random.choice(indices, size=subset_size)
    dataset = dataset.select(indices)
    dataset = tokenizer.preprocess(dataset, use_cached=True)
    # dataset = labelizer.preprocess(dataset, use_cached=False)
    # dataset = collator.preprocess(dataset, use_cached=True)

    dataset.save(save_path)


if __name__ == '__main__':
    main()
