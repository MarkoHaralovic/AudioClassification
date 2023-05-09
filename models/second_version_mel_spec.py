import os
import numpy as np
import librosa
from tqdm import tqdm

# Define constants
SAMPLE_RATE = 22050  # 22000 Hz
DURATION = 1.0  # duration of audio segments in seconds
HOP_SIZE = round(11.6 * SAMPLE_RATE / 1000)
BLOCK_SIZE = round(46.4 * SAMPLE_RATE / 1000)
MEL_BANDS = 96  # number of Mel bands to generate


def extract_mel_spectrogram(audio, sr, n_mels=MEL_BANDS):
    hop_length = int(SAMPLE_RATE * DURATION / (86-1))
    return librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)


def parse_label(audio_path):
    valid_instruments = ['cel', 'cla', 'flu', 'gac',
                         'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
    binary_labels = [
        1 if f'[{inst}]' in os.path.dirname(audio_path) else 0 for inst in valid_instruments]
    return binary_labels


def segment_audio_files(audio_path, threshold=0.01):
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    # Downsample audio to 22000 Hz
    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Downmix to mono
    audio = librosa.to_mono(audio)

    # Normalizing the data
    r = librosa.feature.rms(y=audio)
    a = np.sqrt((len(audio) * r**2) / np.sum(audio**2))
    a = np.mean(a)
    audio = audio * a

    X = []

    for segment in range(int(len(audio) / (SAMPLE_RATE * DURATION))):
        start = int(segment * SAMPLE_RATE * DURATION)
        end = int(start + SAMPLE_RATE * DURATION)
        hop_length = int(SAMPLE_RATE * DURATION / (86-1))

        # Extract Mel spectrogram
        mel_spectrogram = extract_mel_spectrogram(
            audio[start:end], sr)

        # Converting mel spectrogram to decibels
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        X.append(mel_spectrogram_db)

    # Convert data to numpy arrays
    X = np.array(X, dtype=object)
    return X


def process_audio_files(input_dir):
    X = []
    y = []

    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing files in {root}"):
            if file.endswith('.wav') or file.endswith('.ogg'):
                audio_path = os.path.join(root, file)
                try:
                    X_temp = segment_audio_files(audio_path)
                    X.extend(X_temp)

                    binary_labels = parse_label(audio_path)
                    y_temp = [binary_labels for _ in range(len(X_temp))]
                    y.extend(y_temp)

                except Exception as e:
                    print(
                        f"Failed to segment audio file {audio_path}. Error: {type(e).__name__}: {str(e)}")

    X = np.array(X, dtype=object)
    y = np.array(y, dtype=object)

    return X, y


if __name__ == '__main__':
    X, y = process_audio_files(
        'pat_to_irmas_trai')
    np.save(
        'C:\\AudioClassification\\npy_data\\X\\mfccx.npy', X)
    np.save(
        'C:\\AudioClassification\\npy_data\\y\\single_y.npy', y)
