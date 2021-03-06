TRAIN_DATA_PATH = './data/LibriSpeech/train-clean-100' # './data/wav/'
TEST_DATA_PATH = './data/LibriSpeech/test-clean'
NEAT_DATA_PATH = './data/neat/'
MEL_DATA_PATH = './data/mel/'
EMB_DATA_PATH = './data/emb/'

SAMPLE_RATE = 16000

INPUT_SIZE = 39
NUM_LAYERS = 1
NUM_FRAMES = 48000
TRAIN_SIZE = 0.8
BATCH_SIZE = 32
N_USERS = 100

num_frequencies = 1025  # количество частот, которые будем извлекать
frame_length = 0.025     # длина окна STFT (в секундах)
frame_shift = 0.01    # сдвиг окна STFT (в секундах)
num_mels = 64           # размерность mel-пространства признаков
min_frequency = 125      # минимальная анализируемая частота (в Гц)
max_frequency = 7500   # максимальная анализируемая частота (в Гц)

hop_length = int(SAMPLE_RATE * frame_shift)
win_length = int(SAMPLE_RATE * frame_length)
n_fft = (num_frequencies - 1) * 2

ref_db = 20
min_db = -100
PREEMPHASIS = 0.97
POWER = 1.2
