import numpy as np
import librosa
from keras.models import load_model
import os
import glob
import tensorflow_addons as tfa
import warnings
from sklearn.metrics import f1_score
import tensorflow as tf

warnings.filterwarnings("ignore")
SAMPLE_RATE = 22050
BLOCK_SIZE = 1024
HOP_SIZE = 512
MEL_BANDS = 128
DURATION = 1.0


def preprocess_audio(audio_file, sr=SAMPLE_RATE):
    audio, sr = librosa.load(audio_file, sr=sr, mono=True)
    audio = audio / np.max(np.abs(audio))

    segment_length = int(sr * DURATION)
    audio_segments = [audio[i:i + segment_length]
                      for i in range(0, len(audio), segment_length)]

    # Discard the last segment if its length is shorter than the expected duration
    if len(audio_segments[-1]) < segment_length:
        audio_segments = audio_segments[:-1]

    log_mel_spectrograms = []

    for segment in audio_segments:
        stft = librosa.stft(segment, n_fft=BLOCK_SIZE, hop_length=HOP_SIZE)
        mel_spectrogram = librosa.feature.melspectrogram(
            S=np.abs(stft), sr=sr, n_mels=MEL_BANDS)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        log_mel_spectrograms.append(log_mel_spectrogram)

    return np.array(log_mel_spectrograms)


def predict_windows(model, windows):
    return model.predict(windows)


def aggregate_predictions(predictions):
    summed = np.sum(predictions, axis=0)
    return summed / np.sum(summed)


def read_labels_from_file(file_path, valid_instruments):
    with open(file_path, 'r') as file:
        labels = []
        for line in file:
            labels.extend([label.strip() for label in line.split(',')])

    label_array = np.zeros(len(valid_instruments), dtype=int)
    for label in labels:
        if label in valid_instruments:
            label_array[valid_instruments.index(label)] = 1

    return label_array


def test_model(model, test_data_path):
    y_true = []
    X_test = []

    valid_instruments = ['cel', 'cla', 'flu', 'gac',
                         'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']

    audio_files = glob.glob(test_data_path + '/*.wav')
    i = 0
    text_files = glob.glob(test_data_path + '/*.txt')
    for text_file in text_files:
        y_true.append(read_labels_from_file(text_file, valid_instruments))
    np.save("define_path_testY.npy", y_true)
    print(y_true[0])
    i = 0
    for audio_file in audio_files:
        i += 1
        print(i)
        log_mel_spectrograms = preprocess_audio(audio_file)
        X_test.append(log_mel_spectrograms)

    np.save("define_path_testX.npy", X_test)

    return


model = load_model(
    'load_h5_model')
test_data_path = 'path_to_IRMAS_validation_data'
results = test_model(model, test_data_path)
