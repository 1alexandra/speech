from hyperparameters import *
from custom_datasets import VoxCelebDataset
from preprocessing import melspectrogram
from tqdm import tqdm
import torch
import os

data = VoxCelebDataset(RAW_DATA_PATH, transform=melspectrogram)

last_label = None
i = 0

for spectrogram, label in tqdm(data):
    if last_label != label:
        last_label = label
        i = 0
    else:
        i += 1
    
    path = [MEL_DATA_PATH, data.person_ids[label], '{:0>3}.npy'.format(i)]
    
    if not os.path.isdir(os.path.join(*path[:2])):
        os.mkdir(os.path.join(*path[:2]))
        
    path = os.path.join(*path)
    
    torch.save(spectrogram, path)