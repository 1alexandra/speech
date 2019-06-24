import h5py
from IPython.display import Audio
import numpy as np

from hyperparameters import *


def show(audio):
    display(Audio(audio, rate=SAMPLE_RATE))

    
def random_crop(x, crop_size=96):
    i = np.random.randint(0, x.shape[1] - crop_size)
    return x[:, i:i+crop_size]


def flatten(x):
    for k, v in x.items():
        if isinstance(v, h5py.Dataset):
            yield v
        else:
            for v in flatten(v):
                yield v


def get_X(dataset):
    X, y = list(zip(*[(X.numpy(), y) for X, y in dataset]))
    return np.stack(X), np.array(y)
