import librosa
import scipy
import numpy as np

from hyperparameters import *

mel_basis = librosa.filters.mel(SAMPLE_RATE, n_fft, n_mels=num_mels, fmin=min_frequency, fmax=max_frequency)


def pre_emphasis(x):
    return scipy.signal.lfilter([1, -PREEMPHASIS], [1], x)


def stft(y):
    return librosa.stft(y, n_fft=n_fft,
                        hop_length=hop_length, 
                        win_length=win_length)


def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def normalize(S):
    return np.clip((S - min_db) / -min_db, 0, 1)


def melspectrogram(y):
    S = np.abs(stft(pre_emphasis(y)))
    M = amp_to_db(np.dot(mel_basis, S)) - ref_db
    return normalize(M)

