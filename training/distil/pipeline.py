# =============================================================================
#
# =============================================================================
from pathlib import Path
import os

from torch.optim import AdamW
from torch.utils.data import DataLoader

from mlpipeline.datasets.nlp import RuREBus
from mlpipeline.datasets.nlp.units import Entity

from mlpipeline.processors.nlp.prediction_postprocessors import Viterbi
from mlpipeline.processors.nlp.labelizers import BILOULabelizer
from mlpipeline.processors.nlp.collators import PaddingCollator

from mlpipeline.trainers.utils.losses import MSELoss
from mlpipeline.trainers.utils.metrics import F1MacroScoreNER
from mlpipeline.trainers.utils.callbacks import CheckpointCallback, LoggingCallback, FreezeEmbeddingsCallback
from mlpipeline.trainers.utils.lr_schedulers import NoneScheduler
from mlpipeline.trainers import NerDistilTrainer


def distil_pipeline(model, device, experiment_name, checkpoint_name=None):
    epochs = 30
    batch_size = 128

    home_dir = Path(os.getenv('HOME'))
    datasets_dir = home_dir / 'datasets'
    models_dir = home_dir / 'models'

    raw_txt_path = datasets_dir / 'RuREBus' / 'RawTxt100000BertPredictions'
    rurebus_path = datasets_dir / 'RuREBus' / 'Preprocessed'
    embeddings_path = models_dir / 'DeepPavlov-rubert-base-cased_input_embeddings_layer' / 'state_dict.pt'
    output_dir = Path(os.getenv('SCRATCH_DIR')) / 'models' / model.name / 'distil' / experiment_name

    model.load_pretrained_embeddings(embeddings_path)
    model.freeze_embeddings(requires_grad=False)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total trainable parameters:', total_params)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = NoneScheduler(optimizer)
    loss_fn = MSELoss().to(device)

    train_ds = RuREBus.load(raw_txt_path)
    test_ds = RuREBus.load(rurebus_path)['test']

    train_collator = PaddingCollator(
        collate_columns=['tokens_ids', 'attention_mask', 'logits'],
        pad_value=0,
        padding_type='longest')

    val_collator = PaddingCollator(
        collate_columns=['tokens_ids', 'attention_mask'],
        pad_value=0,
        padding_type='longest')

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              collate_fn=train_collator.collate,
                              shuffle=True)
    val_loader = DataLoader(test_ds,
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

    labelizer = BILOULabelizer(text_column='text',
                               tokens_column='tokens',
                               tokens_spans_column='spans',
                               entities_column='entities',
                               out_labels_column='labels',
                               out_labels_ids_column='labels_ids',
                               predicted_labels_ids_column='predicted_labels_ids',
                               out_predicted_entities_column='predicted_entities',
                               entity_types=test_ds.entity_types,
                               entities_deserialize_fn=Entity.from_brat,
                               entities_serialize_fn=Entity.to_brat)

    callbacks = [FreezeEmbeddingsCallback(30),
                 CheckpointCallback(output_dir / 'checkpoints', 10, save_optimizer=False),
                 LoggingCallback(print_=True)]
                 # EarlyStoppingCallback(patience=10)
    metrics = [F1MacroScoreNER(entity_types=test_ds.entity_types,
                               entities_column='entities',
                               predicted_entities_column='predicted_entities',
                               entities_deserialize_fn=Entity.from_brat,
                               name='f1-macro',
                               return_detailed=True)]
    key_metric_name = 'f1-macro'

    trainer = NerDistilTrainer(model=model,
                               optimizer=optimizer,
                               loss_fn=loss_fn,
                               lr_scheduler=scheduler,
                               train_loader=train_loader,
                               val_loader=val_loader,
                               tokens_ids_column='tokens_ids',
                               attention_mask_column='attention_mask',
                               logits_column='logits',
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
