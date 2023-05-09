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


# constants for models: 78kratimenos.h5 and augpoly78.h5
SAMPLE_RATE = 22050
BLOCK_SIZE = 1024
HOP_SIZE = 512
MEL_BANDS = 128
DURATION = 1.0


class_names = ['cel', 'cla', 'flu', 'gac', 'gel',
               'org', 'pia', 'sax', 'tru', 'vio', 'voi']


def parse_label(audio_path):
    """
    Parse the label from the audio file path and create a binary representation of the label.
    """
    valid_instruments = ['cel', 'cla', 'flu', 'gac',
                         'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
    binary_labels = [
        1 if f'[{inst}]' in audio_path else 0 for inst in valid_instruments]
    return binary_labels


def preprocess_audio(audio_path):
    """
    Load an audio file, convert it to mono, normalize it, segment it into fixed-length segments,
    and convert each segment into a set of waveform frames.
    """
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


def process_audio_files(input_dir):
    """
    process all the audio files in a given directory
    """
    X = []
    y = []

    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing files in {root}"):
            if file.endswith('.wav') or file.endswith('.ogg'):
                audio_path = os.path.join(root, file)
                binary_labels = parse_label(audio_path)
                try:
                    log_mel_spectrograms = preprocess_audio(audio_path)
                    for spectrogram in log_mel_spectrograms:
                        X.append(spectrogram)
                        y.append(binary_labels)

                except Exception as e:
                    print(
                        f"Failed to process audio file {audio_path}. Error: {type(e).__name__}: {str(e)}")

    X = np.array(X, dtype=object).astype(np.float32)
    y = np.array(y, dtype=object).astype(np.float32)
    return X, y


def convert_np_int32_to_int(dict_obj):
    for key, value in dict_obj.items():
        if isinstance(value, dict):
            convert_np_int32_to_int(value)
        elif isinstance(value, np.int32):
            dict_obj[key] = int(value)
    return dict_obj


def hamming_accuracy(y_true, y_pred):
    y_true = K.round(K.clip(y_true, 0, 1))
    y_pred = K.round(K.clip(y_pred, 0, 1))
    equal_elements = K.cast(K.equal(y_true, y_pred), K.floatx())
    return K.mean(equal_elements)


def predict_and_export(model_path, audio_folder, output_json_path):
    # Load the saved model
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'F1Score': F1Score,
            'hamming_accuracy': hamming_accuracy
        })

    # Process audio files and make predictions
    X, y = process_audio_files(audio_folder)
    predictions = model.predict(X)

    # Prepare the results in the desired format
    results = {}
    valid_instruments = ['cel', 'cla', 'flu', 'gac',
                         'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
    threshold = 0.3

    for i, file in enumerate(os.listdir(audio_folder)):
        binary_labels = (predictions[i] > threshold).astype(int)
        track_results = {inst: binary_labels[j]
                         for j, inst in enumerate(valid_instruments)}
        results[file] = track_results

    # Export the results to a JSON file
    with open("C:\\AudioClassification\\testing_json_files\\kretimenos_test1.json", "w") as outfile:
        results = convert_np_int32_to_int(results)
        json.dump(results, outfile)


if __name__ == '__main__':
    model_path = 'C:/AudioClassification/h5_models/78kratimenos.h5'
    audio_folder = 'C:\\Polyphonic_audio_classification\\Datasets\\test_dataset-20230502T133016Z-001\\test_dataset'
    output_json_path = 'predictions.json'

    predict_and_export(model_path, audio_folder, output_json_path)
