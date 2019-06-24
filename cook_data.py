from hyperparameters import *
from tqdm import tqdm
import os
from glob import glob
from sklearn.model_selection import train_test_split as split
import librosa
import pathlib
from preprocessing import melspectrogram, downsample_whiten
import torch
import random
import numpy as np
import soundfile as sf
from keras.models import load_model


def get_fat_users(root):
    n_books = []
    for user_id in tqdm(os.listdir(root)):
        books = os.listdir(os.path.join(root, user_id))
        n_books.append((len(books), user_id))
    return [x[1] for x in sorted(n_books, key=lambda x: x[0])[-N_USERS:]]


def cut_wavs(encoder, user_id, records, type):
    record = None
    file_id = 0
    for file in records:
        cur_record, sr = sf.read(file)
        assert sr == SAMPLE_RATE
        if record is None:
            record = cur_record
        else:
            record = np.concatenate((record, cur_record))
    record = record[:(len(record) // NUM_FRAMES) * NUM_FRAMES].reshape(-1, NUM_FRAMES, 1)
    record = downsample_whiten(record, rate=4)
    emb = torch.Tensor(encoder.predict(record))
    for file_id in range(len(emb)):
        write_path = os.path.join(EMB_DATA_PATH, type, f'{user_id:03}')
        pathlib.Path(write_path).mkdir(parents=True, exist_ok=True)
        torch.save(emb[file_id], os.path.join(write_path, f'{file_id:03}.npy'))
        # librosa.output.write_wav(os.path.join(write_path, f'{file_id:03}.wav'), wav, sr=SAMPLE_RATE)


encoder = load_model('models/encoder.h5')
random.seed(123)

fat_users_train = get_fat_users(TRAIN_DATA_PATH)
random.shuffle(fat_users_train)
for uid, real_user_id in tqdm(enumerate(fat_users_train)):
    pattern = os.path.join(TRAIN_DATA_PATH, real_user_id, '*', '*.flac')
    records = glob(pattern)
    train_records, test_records = split(records, train_size=TRAIN_SIZE, random_state=123)
    cut_wavs(encoder, uid, train_records, 'train1')
    cut_wavs(encoder, uid, test_records, 'val')

fat_users_test = get_fat_users(TEST_DATA_PATH)
random.shuffle(fat_users_test)
for uid, real_user_id in tqdm(enumerate(fat_users_test)):
    pattern = os.path.join(TEST_DATA_PATH, real_user_id, '*', '*.flac')
    records = glob(pattern)
    train_records, test_records = split(records, train_size=TRAIN_SIZE, random_state=123)
    cut_wavs(encoder, uid, train_records, 'train2')
    cut_wavs(encoder, uid, test_records, 'test')
