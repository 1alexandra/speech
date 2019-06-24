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


def downsample_whiten(batch, rate=1, rms=0.038021):
    """This function whitens a batch so each sample has 0 mean and the same root mean square amplitude i.e. volume."""
    if len(batch.shape) != 3:
        raise(ValueError, 'Input must be a 3D array of shape (batch_size, n_timesteps, 1).')
    
    batch = batch[:, ::rate, :]
    
    # Subtract mean
    sample_wise_mean = batch.mean(axis=1)
    whitened_batch = batch - np.tile(sample_wise_mean, (1, 1, batch.shape[1])).transpose((1, 2, 0))

    # Divide through
    sample_wise_rescaling = rms / np.sqrt(np.power(batch, 2).mean(axis=1))
    whitened_batch = whitened_batch * np.tile(sample_wise_rescaling, (1, 1, batch.shape[1])).transpose((1, 2, 0))

    return whitened_batch
