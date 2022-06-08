from pathlib import Path
import os

from torch.utils.data import DataLoader
from torch.optim import AdamW

from mlpipeline.datasets.nlp import RuREBus
from mlpipeline.datasets.nlp.units import Entity
from mlpipeline.datasets.nlp.siameese_dataset_wrapper import TripleSimaeeseDatasetWrapper

from mlpipeline.processors.nlp.prediction_postprocessors import Viterbi
from mlpipeline.processors.nlp.labelizers import BILOULabelizer
from mlpipeline.processors.nlp.collators import PaddingCollator, SiameesePaddingCollator, MaxLenSplitCollator

from mlpipeline.trainers.utils.losses import LossList, TripleSiameeseCosineDistanceLoss, CrossEntropyLoss
from mlpipeline.trainers.utils.optimizers import Lookahead
from mlpipeline.trainers.utils.metrics import F1MacroScoreNER, LossListAlphaMetric, LRMetric
from mlpipeline.trainers.utils.callbacks import CheckpointCallback, LoggingCallback, LossListAlphaStepCallback
from mlpipeline.trainers.utils.lr_schedulers import NoneScheduler
from mlpipeline.trainers import NerTripleSiameeseTrainer


def siam_pipeline(model, device, experiment_name, checkpoint_name=None, one_stage=True):
    epochs = 60
    batch_size = 64

    home_dir = Path(os.getenv('HOME'))
    datasets_dir = home_dir / 'datasets'
    rurebus_path = datasets_dir / 'RuREBus' / 'PreprocessedForBert'
    if one_stage:
        dir_name = 'siam_rurebus_one_stage'
    else:
        dir_name = 'siam_rurebus_two_stage'
    output_dir = Path(os.getenv('SCRATCH_DIR')) / 'models' / model.name / dir_name / experiment_name

    rurebus = RuREBus.load(rurebus_path)

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

    train_ds = TripleSimaeeseDatasetWrapper(
        rurebus['train'],
        labelizer=labelizer,
        entity_types_shares='log',
        out_entity_mask_column='siam_entity_mask',
        out_entity_type_id_column='siam_entity_type_id',
        size=10_000
    )
    val_ds = rurebus['test']

    splitter = MaxLenSplitCollator(
        collate_columns=['tokens', 'spans', 'tokens_ids', 'attention_mask',
                         'token_type_ids', 'labels', 'labels_ids'],
        pk_columns=['doc_id', 'start'],
        unite_columns=['logits'],
        max_len=128
    )

    def val_postprocess_batch_fn(batch):
        batch = splitter.unite(batch)
        for example in RuREBus.batch_samples(batch):
            tokens = example['tokens']
            assert tokens[0] == '[CLS]'
            assert tokens[-1] == '[SEP]'
        return batch

    train_collator = SiameesePaddingCollator(
        collate_columns=['tokens_ids', 'attention_mask', 'labels_ids', 'siam_entity_mask'],
        pad_value=0,
        padding_type='longest',
        max_len=128
    )

    val_collator = PaddingCollator(
        collate_columns=['tokens_ids', 'attention_mask', 'labels_ids'],
        pad_value=0,
        padding_type='longest',
        max_len=128
    )
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              collate_fn=train_collator.collate)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            collate_fn=val_collator.collate)

    viterbi = Viterbi(logits_column='logits',
                      word_tokens_indices_column='word_tokens_indices',
                      out_predicted_labels_ids_column='predicted_labels_ids',
                      first_subword_transition_probs=BILOULabelizer.first_subword_transition_probs,
                      middle_subword_transition_probs=BILOULabelizer.middle_subword_transition_probs,
                      last_subword_transition_probs=BILOULabelizer.last_subword_transition_probs,
                      word_transition_probs=BILOULabelizer.word_transition_probs,
                      initial_state=0,
                      pad_label_id=0)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total trainable parameters:', total_params)

    adam = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    optimizer = Lookahead(adam)
    scheduler = NoneScheduler(optimizer)

    siam_loss = TripleSiameeseCosineDistanceLoss()
    ce_loss = CrossEntropyLoss()
    ce_losses = LossList([ce_loss] * len(rurebus.entity_types))
    ce_losses = LossList([ce_losses] * 3)
    if one_stage:
        alpha = 0.99
    else:
        alpha = 0
    loss_fn = LossList([siam_loss, ce_losses], ratios=[alpha, 1 - alpha]).to(device)

    callbacks = [CheckpointCallback(output_dir / 'checkpoints', 20, save_optimizer=False),
                 LoggingCallback(print_=True)]
                 # EarlyStoppingCallback(patience=40)

    metrics = [F1MacroScoreNER(entity_types=rurebus.entity_types,
                               entities_column='entities',
                               predicted_entities_column='predicted_entities',
                               entities_deserialize_fn=Entity.from_brat,
                               name='f1-macro',
                               return_detailed=True),
               LRMetric()]
    if one_stage:
        callbacks.append(LossListAlphaStepCallback(
            factor=0.5, alpha_min=0.01)
        )
        metrics.append(LossListAlphaMetric())

    key_metric_name = 'f1-macro'

    trainer = NerTripleSiameeseTrainer(model=model,
                                       optimizer=optimizer,
                                       loss_fn=loss_fn,
                                       lr_scheduler=scheduler,
                                       train_loader=train_loader,
                                       val_loader=val_loader,
                                       val_postprocess_batch_fn=val_postprocess_batch_fn,
                                       tokens_ids_column='tokens_ids',
                                       attention_mask_column='attention_mask',
                                       labels_ids_column='labels_ids',
                                       siam_entity_mask_column='siam_entity_mask',
                                       siam_entity_type_id_column='siam_entity_type_id',
                                       logits_column='logits',
                                       hidden_states_column='hidden_states',
                                       callbacks=callbacks,
                                       metrics=metrics,
                                       key_metric_name=key_metric_name,
                                       viterbi_decoder=viterbi,
                                       labelizer=labelizer,
                                       device=device,
                                       output_dir=output_dir,
                                       verbose=False)

    if checkpoint_name is not None:
        checkpoint_path = output_dir / 'checkpoints' / checkpoint_name
        trainer.load_checkpoint(checkpoint_path)

    trainer.train(epochs)
