from torch.utils.data import Dataset
import librosa
import torch
import numpy as np

import os
import glob

from hyperparameters import *


class VoxCelebDataset(Dataset):
    def __init__(self, root, sample_rate=SAMPLE_RATE, transform=None):
        super(VoxCelebDataset, self).__init__()
        
        self.sample_rate = sample_rate
        self.transform = transform
        
        self.person_ids = os.listdir(root)
        self.id2label = {person_id: label for label, person_id in enumerate(self.person_ids)}
        
        pattern = os.path.join(root, '*', '*', '*.wav')
        self.filenames = glob.glob(pattern)
        
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        person_label = self.id2label[filename.split(os.path.sep)[-3]]
        
        record = librosa.load(filename, sr=self.sample_rate)[0]
        if self.transform is not None:
            record = self.transform(record)
            
        return record, person_label
        
    def __len__(self):
        return len(self.filenames)


class MelCelebDataset(Dataset):
    def __init__(self, root, transform=None):
        super(MelCelebDataset, self).__init__()

        self.transform = transform

        self.person_ids = os.listdir(root)
        self.id2label = {person_id: label for label, person_id in enumerate(self.person_ids)}

        pattern = os.path.join(root, '*', '*.npy')
        self.filenames = glob.glob(pattern)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        person_label = self.id2label[filename.split(os.path.sep)[-2]]

        spectrogram = torch.load(filename)
        if self.transform is not None:
            spectrogram = self.transform(spectrogram)

        return spectrogram, person_label

    def __len__(self):
        return len(self.filenames)
