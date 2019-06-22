from hyperparameters import *
from tqdm import tqdm
import os
from glob import glob
from sklearn.model_selection import train_test_split as split
import librosa
import pathlib
from preprocessing import melspectrogram
import torch


def cut_wavs(real_user_id, user_id, videos, type):
    filenames = []
    for video_id in videos:
        pattern = os.path.join(RAW_DATA_PATH, real_user_id, video_id, '*.wav')
        filenames.extend(glob(pattern))
    
    # autoinc
    file_id = 0
    for file in filenames:
        record = librosa.load(file, sr=SAMPLE_RATE)[0]
        for i in range(0, len(record), NUM_FRAMES):
            wav = record[i:i+NUM_FRAMES]
            # skip last part with len > NUM_FRAMES
            if len(wav) == NUM_FRAMES:
                write_path = os.path.join(MEL_DATA_PATH, type, f'{user_id}')
                pathlib.Path(write_path).mkdir(parents=True, exist_ok=True)
                
                mel = torch.Tensor(melspectrogram(wav))
                torch.save(mel, os.path.join(write_path, f'{file_id:03}.npy'))
                # librosa.output.write_wav(os.path.join(write_path, f'{file_id}.wav'), wav, sr=SAMPLE_RATE)
                file_id += 1


users, new_users = split(list(enumerate(os.listdir(RAW_DATA_PATH))), test_size=0.2, random_state=123)
for uid, real_user_id in tqdm(users):
    videos = os.listdir(os.path.join(RAW_DATA_PATH, real_user_id))
    train_videos, val_test = split(videos, train_size=TRAIN_SIZE, random_state=123)
    val_videos, test_videos = split(val_test, test_size=0.5, random_state=123)
    cut_wavs(real_user_id, uid, train_videos, 'train')
    cut_wavs(real_user_id, uid, val_videos, 'val')
    cut_wavs(real_user_id, uid, test_videos, 'test')

for uid, real_user_id in tqdm(new_users):
    videos = os.listdir(os.path.join(RAW_DATA_PATH, real_user_id))
    cut_wavs(real_user_id, uid, videos, 'test_new')
