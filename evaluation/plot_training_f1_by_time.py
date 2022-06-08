import os
from matplotlib import pyplot as plt
from pathlib import Path

from mlpipeline.trainers.utils.history import History


def get_time_array(history: History, start_time: float = 0, start_epoch: int = 0):
    time = [start_time]
    for start, end in zip(history.start_timestamp[start_epoch+1:],
                          history.end_timestamp[start_epoch+1:]):
        duration = (end - start).total_seconds() / 3600
        time.append(time[-1] + duration)
    return time


def main():
    plt.rcParams.update({'font.size': 12})
    figsize = (10, 4)
    home_dir = Path(os.getenv('HOME'))
    root = home_dir / 'bert_vs_rnn_nsu_coursework' / 'models'

    out_dir = root.parent / 'results'
    ner_path = root / 'rurebus'
    dist_path = root / 'distil'
    ft_path = root / 'fine_tuning'
    ner_key = 'ner'
    dist_key = 'dist'
    ft_key = 'dist+ner'
    history_paths = {
        'CLSTM': {ft_key: ft_path / 'ConvLSTM_epoch=40_f1-macro=0.54448' / 'history.json',
                  ner_key: ner_path / 'ConvLSTM_epoch=100_f1-macro=0.41578' / 'history.json',
                  dist_key: dist_path / 'ConvLSTM_epoch=19_f1-macro=0.57962' / 'history.json'},
        'CSRU': {ft_key: ft_path / 'ConvSRU_epoch=40_f1-macro=0.53861' / 'history.json',
                 ner_key: ner_path / 'ConvSRU_epoch=86_f1-macro=0.43208' / 'history.json',
                 dist_key: dist_path / 'ConvSRU_epoch=14_f1-macro=0.56631' / 'history.json'},
        'CSRU++': {ft_key: ft_path / 'ConvSRUpp_epoch=40_f1-macro=0.53547' / 'history.json',
                   ner_key: ner_path / 'ConvSRUpp_epoch=100_f1-macro=0.42305' / 'history.json',
                   dist_key: dist_path / 'ConvSRUpp_epoch=16_f1-macro=0.5802' / 'history.json'},
    }
    colors = {ft_key: 'green',
              dist_key: 'orange',
              ner_key: 'royalblue'}

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (model, paths) in zip(axes, history_paths.items()):
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Время обучения, часов')
        ax.set_ylabel('F1-macro')
        ax.set_title(model)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(y=0.56576, color='grey', label='Bert F1-macro', linestyle='--')

        # ner
        history = History.load(paths[ner_key])
        f1 = history.metrics['f1-macro']
        time = get_time_array(history, 0)
        ax.plot(time, f1, label=ner_key, color=colors[ner_key])

        # dist
        history = History.load(paths[dist_key])
        dist_epochs = history.epoch
        f1 = history.metrics['f1-macro']
        time = get_time_array(history, 0)
        # if model == 'CSRU':
        #     start_idx = 5
        #     f1 = f1[start_idx:]
        #     time = get_time_array(history, )[start_idx:]
        #     start = time[0]
        #     for i in range(len(time)):
        #         time[i] -= start
        ax.plot(time, f1, label=dist_key, color=colors[dist_key])

        # fine-tuning
        history = History.load(paths[ft_key])
        f1 = history.metrics['f1-macro'][dist_epochs:]
        time = get_time_array(history, time[-1], dist_epochs)
        ax.plot(time, f1, label=ft_key, color=colors[ft_key])
        ax.legend(loc='upper right')

    fig.show()
    fig.savefig(out_dir / 'training.png')


if __name__ == '__main__':
    main()
