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


def preprocess_audio_cqt(audio_path):
    """
    Load an audio file, convert it to mono, normalize it, segment it into fixed-length segments,
    and compute the CQT spectrograms for each segment.
    """
    # Load and preprocess audio
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=False)
    audio = librosa.to_mono(audio)
    audio = audio / np.max(np.abs(audio))

    # Segment audio into fixed-length segments
    segment_length = int(SAMPLE_RATE * DURATION)
    audio_segments = [audio[i:i + segment_length]
                      for i in range(0, len(audio), segment_length)]
    audio_segments = [segment for segment in audio_segments if len(
        segment) == segment_length]

    # Compute CQT spectrograms
    cqt_spectrograms = []
    for segment in audio_segments:
        cqt = np.abs(librosa.cqt(segment, sr=SAMPLE_RATE, hop_length=HOP_SIZE))
        log_cqt_spectrogram = librosa.amplitude_to_db(cqt, ref=np.max)
        cqt_spectrograms.append(log_cqt_spectrogram)

    return cqt_spectrograms


def preprocess_audio_wav(audio_path):
    """
    Load an audio file, convert it to mono, normalize it, segment it into fixed-length segments,
    and convert each segment into a set of waveform frames.
    """
    # Load and preprocess audio
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=False)
    audio = librosa.to_mono(audio)
    audio = audio / np.max(np.abs(audio))

    # Segment audio into fixed-length segments
    segment_length = int(SAMPLE_RATE * DURATION)
    audio_segments = [audio[i:i + segment_length]
                      for i in range(0, len(audio), segment_length)]
    audio_segments = [segment for segment in audio_segments if len(
        segment) == segment_length]

    # Convert segments into waveform frames
    waveform_frames = []
    for segment in audio_segments:
        frames = waveform_to_frames(segment, BLOCK_SIZE, HOP_SIZE)
        waveform_frames.append(frames)

    return waveform_frames


def preprocess_audio_chromagram(audio_path):
    """
    Load an audio file, convert it to mono, normalize it, segment it into fixed-length segments,
    and compute the chromagrams for each segment.
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

    chromagrams = []

    for segment in audio_segments:
        stft = librosa.stft(segment, n_fft=BLOCK_SIZE, hop_length=HOP_SIZE)
        chromagram = librosa.feature.chroma_stft(
            S=np.abs(stft), sr=SAMPLE_RATE, n_chroma=24)
        chromagrams.append(chromagram * 50)

    return chromagrams


def preprocess_audio(audio_path):
    """
    Load an audio file, convert it to mono, normalize it, segment it into fixed-length segments,
    and compute the log-mel spectrograms for each segment.
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


def waveform_to_frames(waveform, frame_size, hop_size):
    """
    Convert a waveform into a set of overlapping frames.
    """
    num_frames = 1 + (len(waveform) - frame_size) // hop_size
    frames = np.zeros((num_frames, frame_size))
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frames[i, :] = waveform[start:end]
    return frames


def preprocess_spectral_contrast(audio_path):
    """
    Load an audio file, convert it to mono, normalize it, segment it into fixed-length segments,
    and compute the spectral contrast features for each segment.
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

    spectral_contrasts = []

    for segment in audio_segments:
        stft = librosa.stft(segment, n_fft=BLOCK_SIZE, hop_length=HOP_SIZE)
        spectral_contrast = librosa.feature.spectral_contrast(
            S=np.abs(stft), sr=SAMPLE_RATE, n_bands=6)
        spectral_contrasts.append(spectral_contrast)

    return spectral_contrasts


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

                log_mel_spectrograms = preprocess_audio(audio_path)
                chromagrams = preprocess_audio_chromagram(audio_path)
                spectral_contrasts = preprocess_spectral_contrast(audio_path)
                spectograms = np.concatenate(
                    (log_mel_spectrograms, chromagrams, spectral_contrasts), axis=1)
                for spectrogram in spectograms:
                    X.append(spectrogram)
                    y.append(binary_labels)
    X = np.array(X, dtype=object).astype(np.float32)
   #  X = np.expand_dims(X, axis=1).astype(np.float32)
    X = np.reshape(X, (*X.shape, 1)).astype(np.float32)
   #  X = np.squeeze(X, axis=1)
   #  X = np.expand_dims(X, axis=-1)
    X = np.transpose(X, (0, 3, 1, 2))
    y = np.array(y, dtype=object).astype(np.float32)
    return X, y


def hamming_accuracy(y_true, y_pred):
    y_true = K.round(K.clip(y_true, 0, 1))
    y_pred = K.round(K.clip(y_pred, 0, 1))
    equal_elements = K.cast(K.equal(y_true, y_pred), K.floatx())
    return K.mean(equal_elements)


def convert_np_int32_to_int(dict_obj):
    for key, value in dict_obj.items():
        if isinstance(value, dict):
            convert_np_int32_to_int(value)
        elif isinstance(value, np.int32):
            dict_obj[key] = int(value)
    return dict_obj


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
    threshold = 0.5

    for i, file in enumerate(os.listdir(audio_folder)):
        binary_labels = (predictions[i] > threshold).astype(int)
        track_results = {inst: binary_labels[j]
                         for j, inst in enumerate(valid_instruments)}
        results[file] = track_results

    # Export the results to a JSON file
    with open("C:\\AudioClassification\\testing_json_files\\augpoly78_test1.json", "w") as outfile:
        results = convert_np_int32_to_int(results)
        json.dump(results, outfile)


if __name__ == '__main__':
    model_path = 'C:/AudioClassification/h5_models/augpoly78.h5'
    audio_folder = 'C:\\Polyphonic_audio_classification\\Datasets\\test_dataset-20230502T133016Z-001\\test_dataset'
    output_json_path = 'predictions.json'

    predict_and_export(model_path, audio_folder, output_json_path)
