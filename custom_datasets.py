from torch.utils.data import Dataset
import librosa
import torch
import numpy as np
from preprocessing import extract_mfcc

import os
import glob
import re
import random
from tqdm import tqdm

from hyperparameters import *


def match(regexp, s):
    return re.match(regexp, s) is not None


class MelCelebDataset(Dataset):
    def __init__(self, root, type, transform=None, user_regexp=None):
        super(MelCelebDataset, self).__init__()
        
        assert type in ('train', 'test', 'val')

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
            spectrogram = self.transform(spectrogram)

        return spectrogram, person_label

    def __len__(self):
        return len(self.filenames)


class EmbDataset(Dataset):
    def __init__(self, root, type, transform=None, user_regexp=None):
        super(EmbDataset, self).__init__()
        
        assert type in ('train1', 'train2', 'test', 'val')

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

        emb = torch.load(filename)
        if self.transform is not None:
            emb = self.transform(spectrogram)

        return emb, person_label

    def __len__(self):
        return len(self.filenames)

    
class WavDataset(Dataset):
    def __init__(self, root, type, transform=None, user_regexp=None):
        super(WavDataset, self).__init__()
        
        assert type in ('train', 'test', 'val')

        self.transform = transform

        self.person_ids = os.listdir(os.path.join(root, type))
        self.id2label = {person_id: int(person_id) for person_id in self.person_ids}
        pattern = os.path.join(root, type, '*', '*.wav')
        self.filenames = glob.glob(pattern)
        if user_regexp is not None:
            self.filenames = list(filter(lambda x: match(os.path.join(root, type, user_regexp, ''), x), self.filenames))

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        person_label = self.id2label[filename.split(os.path.sep)[-2]]

        wav = librosa.load(filename, sr=SAMPLE_RATE)[0]
        if self.transform is not None:
            wav = self.transform(wav)

        return wav, person_label

    def __len__(self):
        return len(self.filenames)
