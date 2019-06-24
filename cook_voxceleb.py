from hyperparameters import *
from tqdm import tqdm
import os
from glob import glob
from sklearn.model_selection import train_test_split as split
import librosa
import pathlib
from preprocessing import melspectrogram
import torch
import random
import numpy as np


def get_n_records(user_id, videos):
    n_records = 0
    for video_id in videos:
        pattern = os.path.join(RAW_DATA_PATH, user_id, video_id, '*.wav')
        cur_l = 0
        for file in glob(pattern):
            record = librosa.load(file, sr=SAMPLE_RATE)[0]
            cur_l += len(record)
        n_records += cur_l // NUM_FRAMES
    return n_records


def get_fat_users():
    n_records = []
    for user_id in tqdm(os.listdir(RAW_DATA_PATH)):
        videos = os.listdir(os.path.join(RAW_DATA_PATH, user_id))
        n_records.append((get_n_records(user_id, videos), user_id))
    return [x[1] for x in sorted(n_records, key=lambda x: x[0])[-N_USERS - 50:-50]]


def cut_wavs(real_user_id, user_id, videos, type):
    # autoinc
    file_id = 0
    for video_id in videos:
        pattern = os.path.join(RAW_DATA_PATH, real_user_id, video_id, '*.wav')
        record = None
        for file in glob(pattern):
            cur_record = librosa.load(file, sr=SAMPLE_RATE)[0]
            if record is None:
                record = cur_record
            else:
                record = np.concatenate((record, cur_record))
        
        for i in range(0, len(record), NUM_FRAMES):
            wav = record[i:i+NUM_FRAMES]
            # skip last part with len > NUM_FRAMES
            if len(wav) == NUM_FRAMES:
                # save wav
                write_path = os.path.join(NEAT_DATA_PATH, type, f'{user_id:03}')
                pathlib.Path(write_path).mkdir(parents=True, exist_ok=True)
                librosa.output.write_wav(os.path.join(write_path, f'{file_id:03}.wav'), wav, sr=SAMPLE_RATE)
                
                # save mel
                mel_write_path = os.path.join(MEL_DATA_PATH, type, f'{user_id:03}')
                pathlib.Path(mel_write_path).mkdir(parents=True, exist_ok=True)
                mel = torch.Tensor(melspectrogram(wav))
                torch.save(mel, os.path.join(mel_write_path, f'{file_id:03}.npy'))
                
                file_id += 1


random.seed(123)
fat_users = get_fat_users()
random.shuffle(fat_users)
for uid, real_user_id in tqdm(enumerate(fat_users)):
    videos = os.listdir(os.path.join(RAW_DATA_PATH, real_user_id))
    train_videos, val_test = split(videos, train_size=TRAIN_SIZE, random_state=123)
    val_videos, test_videos = split(val_test, test_size=0.5, random_state=123)
    cut_wavs(real_user_id, uid, train_videos, 'train')
    cut_wavs(real_user_id, uid, val_videos, 'val')
    cut_wavs(real_user_id, uid, test_videos, 'test')
