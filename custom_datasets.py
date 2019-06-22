from torch.utils.data import Dataset
import librosa
import torch
import numpy as np

import os
import glob
import re

from hyperparameters import *


def match(regexp, s):
    return re.match(regexp, s) is not None


# class VoxCelebDataset(Dataset):
#     def __init__(self, root, sample_rate=SAMPLE_RATE, transform=None):
#         super(VoxCelebDataset, self).__init__()
        
#         self.sample_rate = sample_rate
#         self.transform = transform
        
#         self.person_ids = os.listdir(root)
#         self.id2label = {person_id: label for label, person_id in enumerate(self.person_ids)}
        
#         pattern = os.path.join(root, '*', '*', '*.wav')
#         self.filenames = glob.glob(pattern)
        
#     def __getitem__(self, idx):
#         filename = self.filenames[idx]
#         person_label = self.id2label[filename.split(os.path.sep)[-3]]
        
#         record = librosa.load(filename, sr=self.sample_rate)[0]
#         if self.transform is not None:
#             record = self.transform(record)
            
#         return record, person_label
        
#     def __len__(self):
#         return len(self.filenames)


class MelCelebDataset(Dataset):
    def __init__(self, root, type, transform=None, user_regexp=None):
        super(MelCelebDataset, self).__init__()
        
        assert type in ('train', 'test', 'val', 'test_new')

        self.transform = transform

        self.person_ids = os.listdir(os.path.join(root, type))
        self.id2label = {person_id: int(person_id) for person_id in self.person_ids}

        pattern = os.path.join(root, type, '*', '*.npy')
        self.filenames = glob.glob(pattern)
        if user_regexp is not None:
            self.filenames = list(filter(lambda x: match(os.path.join(root, type, user_regexp, ''), x), self.filenames))

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        person_label = self.id2label[filename.split(os.path.sep)[-2]]

        spectrogram = torch.load(filename)
        if self.transform is not None:
            spectrogram = self.transform(record)

        return spectrogram, person_label

    def __len__(self):
        return len(self.filenames)
