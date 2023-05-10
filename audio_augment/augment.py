import os
import numpy as np
import librosa
from tqdm import tqdm
import random
import re
import audioread

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
    """
    Compute spectral contrast features for a given audio file.

    Args:
    audio_path (str): Path to the audio file.

    Returns:
    list: Spectral contrast features.
    """
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
    """
    Compute chromagram features for a given audio file.

    Args:
    audio_path (str): Path to the audio file.

    Returns:
    list: Chromagram features.
    """

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
    """
    Extract the instrument abbreviations from the file name.

    Args:
    file_name (str): File name to extract instruments from.

    Returns:
    list: List of extracted instruments.
    """
    # Extract the instrument abbreviations from the file name
    instruments = re.findall(r'\[([a-z]{3})\]', file_name)
    return [instr for instr in instruments if instr in class_names]


def check_unique_instruments(file1, file2):
    """
    Check if the two files contain any common instruments.

    Args:
    file1 (str): File name 1.
    file2 (str):file2 (str): File name 2.

    Returns :
    bool True if there are no common instruments, False otherwise
    """
    # Check if the two files contain any common instruments
    instruments1 = parse_instruments(file1)
    instruments2 = parse_instruments(file2)
    return not bool(set(instruments1) & set(instruments2))


def mel_spectrogram(audio_path):
    """
    Compute log Mel spectrogram features for a given audio file.
    Args:
    audio_path (str): Path to the audio file.

    Returns:
    list: Log Mel spectrogram features.
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


def one_hot_encode(labels, class_names):
    """
    One-hot encode the given labels.
    Args:
    labels (list): List of labels to be encoded.
    class_names (list): List of unique class names.

    Returns:
    list: One-hot encoded labels.
    """

    # One-hot encode the given labels
    encoding = [0] * len(class_names)
    for label in labels:
        encoding[class_names.index(label)] = 1
    return encoding


def process_audio_files(input_dir, num_iterations=100):
    """
    Process audio files and extract features.
    Args:
    input_dir (str): Directory containing audio files.
    num_iterations (int, optional): Number of iterations to run. Defaults to 100.

    Returns:
    tuple: X and y arrays containing the combined audio features and corresponding labels.
    """
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
                        spectograms = np.concatenate(
                            (log_mel_spectrograms, chromagrams, spectral_contrasts), axis=1)

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
    X = np.array(X, dtype=object)
    y = np.array(y, dtype=object)

    return X, y


if __name__ == '__main__':
    X, y = process_audio_files(
        'path_to_one_second_interval_data || path_to_IRMAS_training_data', num_iterations=15000)
    z = np.sum(y, axis=1)
    np.save('C:\\AudioClassification\\npy_data\\X\\[set_name].npy', X)
    np.save('C:\\AudioClassification\\npy_data\\y\\[set_name].npy', y)
    np.save('C:\\AudioClassification\\npy_data\\Z\\[set_name].npy', z)
