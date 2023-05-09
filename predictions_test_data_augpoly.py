import os
import numpy as np
import librosa
from tqdm import tqdm
import random
import re
import audioread
import glob
import json
import tensorflow as tf
from tensorflow_addons.metrics import F1Score

# Constants
SAMPLE_RATE = 22050
BLOCK_SIZE = 1024
HOP_SIZE = 512
MEL_BANDS = 128
DURATION = 1.0

# constants to get the shape of 96,86
# SAMPLE_RATE = 22050
# BLOCK_SIZE = round(46.4 * SAMPLE_RATE / 1000)
# HOP_SIZE = round(259.41)
# MEL_BANDS = 96
# DURATION = 1.0


class_names = ['cel', 'cla', 'flu', 'gac', 'gel',
               'org', 'pia', 'sax', 'tru', 'vio', 'voi']


def spectral_contrast(audio_path):
    # Load and preprocess the audio
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=False)
    audio = librosa.to_mono(audio)
    audio = audio / np.max(np.abs(audio))

    # Segment audio into one-second intervals
    segment_length = int(SAMPLE_RATE * DURATION)
    audio_segments = [audio[i:i + segment_length]
                      for i in range(0, len(audio), segment_length)]
    audio_segments = [segment for segment in audio_segments if len(
        segment) == segment_length]

    # Calculate the spectral contrast features for each segment
    spectral_contrasts = []

    for segment in audio_segments:
        stft = librosa.stft(segment, n_fft=BLOCK_SIZE, hop_length=512)
        spectral_contrast = librosa.feature.spectral_contrast(
            S=np.abs(stft), sr=SAMPLE_RATE, n_bands=6)
        spectral_contrasts.append(spectral_contrast)

    return spectral_contrasts


def chromagram(audio_path):
    tin_normalize = 50

    # Load and preprocess the audio
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=False)
    audio = librosa.to_mono(audio)
    audio = audio / np.max(np.abs(audio))

    # Segment audio into one-second intervals
    segment_length = int(SAMPLE_RATE * DURATION)
    audio_segments = [audio[i:i + segment_length]
                      for i in range(0, len(audio), segment_length)]
    audio_segments = [segment for segment in audio_segments if len(
        segment) == segment_length]

    # Calculate the chromagram features for each segment
    chromagrams = []

    for segment in audio_segments:
        stft = librosa.stft(segment, n_fft=BLOCK_SIZE, hop_length=HOP_SIZE)
        chromagram = librosa.feature.chroma_stft(
            S=np.abs(stft), sr=SAMPLE_RATE, n_chroma=24)
        chromagrams.append(chromagram * tin_normalize)

    return chromagrams


def parse_instruments(file_name):
    # Extract the instrument abbreviations from the file name
    instruments = re.findall(r'\[([a-z]{3})\]', file_name)
    return [instr for instr in instruments if instr in class_names]


def check_unique_instruments(file1, file2):
    # Check if the two files contain any common instruments
    instruments1 = parse_instruments(file1)
    instruments2 = parse_instruments(file2)
    return not bool(set(instruments1) & set(instruments2))


def mel_spectrogram(audio_path):
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=False)
    audio = librosa.to_mono(audio)
    audio = audio / np.max(np.abs(audio))

    # Segment audio into one-second intervals
    segment_length = int(SAMPLE_RATE * DURATION)
    audio_segments = [audio[i:i + segment_length]
                      for i in range(0, len(audio), segment_length)]
    audio_segments = [segment for segment in audio_segments if len(
        segment) == segment_length]

    log_mel_spectrograms = []
    for segment in audio_segments:
        stft = librosa.stft(segment, n_fft=BLOCK_SIZE, hop_length=HOP_SIZE)
        mel_spectrogram = librosa.feature.melspectrogram(
            S=np.abs(stft), sr=SAMPLE_RATE, n_mels=MEL_BANDS)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        log_mel_spectrograms.append(log_mel_spectrogram)

    return log_mel_spectrograms


def preprocess_audio_file(audio_file):
    log_mel_spectrograms = mel_spectrogram(audio_file)
    chromagrams = chromagram(audio_file)
    spectral_contrasts = spectral_contrast(audio_file)
    spectograms = np.mean(np.concatenate(
        (log_mel_spectrograms, chromagrams, spectral_contrasts), axis=1), axis=0, keepdims=True)

    X = np.array([spectograms], dtype=np.float32)
    return X


def one_hot_encode(labels, class_names):
    # One-hot encode the given labels
    encoding = [0] * len(class_names)
    for label in labels:
        encoding[class_names.index(label)] = 1
    return encoding


def hamming_accuracy(y_true, y_pred):
    y_true = K.round(K.clip(y_true, 0, 1))
    y_pred = K.round(K.clip(y_pred, 0, 1))
    equal_elements = K.cast(K.equal(y_true, y_pred), K.floatx())
    return K.mean(equal_elements)


def process_audio_files(input_dir, num_iterations=100):
    X = []
    y = []

    for _ in tqdm(range(num_iterations), desc="Processing audio files"):
        folder_paths = [os.path.join(input_dir, folder) for folder in os.listdir(
            input_dir) if os.path.isdir(os.path.join(input_dir, folder))]

        n = random.randint(1, 5)
        if n > len(folder_paths):
            n = len(folder_paths)

        selected_folders = random.sample(folder_paths, n)

        audio_segments = []
        unique_instruments = []

        for folder in selected_folders:
            folder_path = os.path.join(input_dir, folder)
            audio_files = [file for file in os.listdir(
                folder_path) if file.endswith(('.wav', '.ogg'))]
            random.shuffle(audio_files)

            for audio_file in audio_files:
                if not unique_instruments or all([check_unique_instruments(audio_file, instrument_file) for instrument_file in unique_instruments]):
                    try:
                        audio_path = os.path.join(folder_path, audio_file)
                        log_mel_spectrograms = mel_spectrogram(audio_path)
                        chromagrams = chromagram(audio_path)
                        spectral_contrasts = spectral_contrast(audio_path)
                        spectograms = np.mean(np.concatenate(
                            (log_mel_spectrograms, chromagrams, spectral_contrasts), axis=1), axis=0, keepdims=True)

                        audio_segments.append(spectograms)
                        unique_instruments.append(audio_file)
                        break
                    except audioread.exceptions.NoBackendError:
                        print(
                            f"Error: Unable to process {audio_file}. Skipping.")
                        continue

        combined_audio = np.sum(audio_segments, axis=0)
        X.append(combined_audio)
        instrument_labels = [
            instr for file in unique_instruments for instr in parse_instruments(file)]
        y.append(one_hot_encode(instrument_labels, class_names))
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)

    return X, y


def test_model(model, audio_file):
    X = preprocess_audio_file(audio_file)
    predictions = model.predict(X)
    one_hot_predictions = (predictions > 0.5).astype(int)
    return {audio_file: dict(zip(class_names, one_hot_predictions[0]))}


if __name__ == '__main__':
    model = tf.keras.models.load_model(
        'C:/AudioClassification/h5_models/augpoly78.h5',
        custom_objects={
            'F1Score': F1Score,
            'hamming_accuracy': hamming_accuracy
        })

    audio_folder = 'C:\\Polyphonic_audio_classification\\Datasets\\test_dataset-20230502T133016Z-001\\test_dataset'

    for file in os.listdir(audio_folder):
        if file.endswith(('.wav', '.ogg')):
            audio_file = os.path.join(audio_folder, file)
            results = test_model(model, audio_file)
            with open("C:\\AudioClassification\\testing_json_files\\{}_augpoly78_test.json".format(file), "w") as outfile:
                json.dump(results, outfile)
