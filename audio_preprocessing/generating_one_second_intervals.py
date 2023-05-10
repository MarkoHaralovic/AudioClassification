from tqdm import tqdm
import os
import librosa
import numpy as np
import pandas as pd
from scipy.io.wavfile import write

# Define constants
SAMPLE_RATE = 22050  # 22000 Hz
DURATION = 1.0  # duration of audio segments in seconds
NUM_MFCC = 13  # number of Mel frequency cepstral coefficients to extract
NUM_CLASSES = 11


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean


def extract_mfcc(audio, sr, n_mfcc=NUM_MFCC):
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)


def extract_mel_spectrogram(audio, sr, n_mels=128):
    return librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)


def segment_audio_files(audio_path, output_dir, threshold=0.01):
    try:
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        # Downsample audio to 22000 Hz
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        # Downmix to mono
        audio = librosa.to_mono(audio)

        # Normalizing the data
        r = librosa.feature.rms(y=audio)
        a = np.sqrt((len(audio) * r**2) / np.sum(audio**2))
        audio = audio * a

        # Segment audio into one-second intervals
        for segment in range(int(len(audio) / (SAMPLE_RATE * DURATION))):
            start = int(segment * SAMPLE_RATE * DURATION)
            end = int(start + SAMPLE_RATE * DURATION)
            # Extract Mel spectrogram and MFCC
            mfccs = extract_mfcc(audio[start:end], SAMPLE_RATE, NUM_MFCC)
            mel_spectrogram = extract_mel_spectrogram(
                audio[start:end], SAMPLE_RATE)
            # Create output path for segmented audio
            file_name, file_ext = os.path.splitext(
                os.path.basename(audio_path))
            output_file_name = f"{file_name}_{segment + 1}{file_ext}"
            output_file_path = os.path.join(output_dir, output_file_name)
            # Save segmented audio to output directory
            write(output_file_path, SAMPLE_RATE, audio[start:end])
    except Exception as e:
        print(f"Failed to segment audio file: {audio_path}. Error: {e}")


def process_audio_files(input_dir, output_base_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing files in {root}"):
            if file.endswith('.wav') or file.endswith('.ogg'):
                audio_path = os.path.join(root, file)
                # Create output directory if not exists
                output_dir = os.path.join(
                    output_base_dir, os.path.relpath(root, input_dir))
                os.makedirs(output_dir, exist_ok=True)
                # Segment and save audio files
                segment_audio_files(audio_path, output_dir)


if __name__ == '__main__':
    process_audio_files('path_to_original_dataset',
                        'path_for_new_dataset')
