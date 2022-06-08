import torch

from mlpipeline.models.nlp.ner import ConvLSTM
from training.distil.pipeline import distil_pipeline


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConvLSTM(n_ent_types=8,
                     n_classes=5,
                     cnn_layers=2,
                     cnn_kernels=[1, 3, 5],
                     rnn_layers=4,
                     hid_size=768//2,
                     head_layers=2,
                     dropout=0.1).to(device)
    distil_pipeline(model, device, experiment_name='v0.0.1')


if __name__ == '__main__':
    main()
