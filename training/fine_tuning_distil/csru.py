import torch

from mlpipeline.models.nlp.ner import ConvSRU
from training.fine_tuning_distil.pipeline import ner_pipeline


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConvSRU(n_ent_types=8,
                    n_classes=5,
                    cnn_layers=2,
                    cnn_kernels=[1, 3, 5],
                    rnn_layers=4,
                    hid_size=768//2,
                    head_layers=2,
                    dropout=0.1).to(device)
    checkpoint = '../../../distil/v0.0.1/checkpoints/ConvSRU_epoch=14_f1-macro=0.56631'
    ner_pipeline(model, device, experiment_name='v1.0.0',
                 checkpoint_name=checkpoint)


if __name__ == '__main__':
    main()
