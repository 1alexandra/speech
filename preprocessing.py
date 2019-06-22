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


def extract_mfcc(sound, sampling_rate=SAMPLE_RATE, shift=32., L=128., mel_coefs=120, mfcc_coefs=12, alpha=0.9, eps=1e-9):
    mfcc = librosa.feature.mfcc(y=sound, sr=sampling_rate, n_mfcc=mfcc_coefs)
    energy = librosa.feature.rms(sound)
    mfcc_energy = np.vstack((mfcc, energy))
    dx = librosa.feature.delta(mfcc_energy, order=1, width=3)
    d2x = librosa.feature.delta(mfcc_energy, order=2, width=3)
    res_features = np.vstack((mfcc_energy, dx, d2x)).T
    return res_features.astype(np.float32)
