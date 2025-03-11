import datetime
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch 
from scipy.stats import spearmanr
from scipy.stats import pearsonr 
import sys


from src.configuration import Config

def get_date_time() -> str:
    return datetime.datetime.now().strftime("%Y_%d_%m_%H%M")


def get_checkpoint_path(config: Config) -> str:
    """ Returns path to .ckpt model checkpoint file. """

    date_time = get_date_time()
    config.date_time = date_time
    config.checkpoint_path = f'{config.checkpoint_path}/ckpt_{date_time}_{config.name}'



def create_paths(config: Config) -> None:
    """ Creates paths defined in config if they don't exist already. """

    Path(config.checkpoint_path).mkdir(parents=True, exist_ok=True)
    Path(config.data_path).mkdir(parents=True, exist_ok=True)
    Path(config.tensorboard_path).mkdir(parents=True, exist_ok=True)


def show_config(path: str) -> None:
    import json
    config = torch.load(path)['hyper_parameters']
    print(json.dumps(config, sort_keys=True, indent=4))


def show_samples(results: np.ndarray, num_samples: int=5, **kwargs) -> None:
    plt.figure(figsize=(10,4))
    for i in range(num_samples):
        plt.subplot(1,num_samples,i+1)
        plt.imshow(results[np.random.randint(len(results))], **kwargs)
    plt.show()


def steps_to_time(steps, num_steps):
    assert steps <= num_steps
    return steps/num_steps


def time_to_steps(time, num_steps):
    assert time >= 0 and time <= num_steps
    return int(time*num_steps)


def compute_correlations(a, b):
    """Computes the Spearman correlation between two arrays a and b."""

    assert(len(a) == len(b))

    corr = np.zeros(len(a))

    for i in range(len(a)):
        corr[i] = pearsonr(a[i].flatten(), b[i].flatten()).statistic

    return corr


def compute_filtered_correlations(a, b, filter):
    """Computes the Spearman correlation between two arrays a and b that are low-pass filtered."""

    filtered_a = filter(a)
    filtered_b = filter(b)

    correlations = compute_correlations(filtered_a, filtered_b)

    return correlations


class MyProgressBar(pl.callbacks.TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar