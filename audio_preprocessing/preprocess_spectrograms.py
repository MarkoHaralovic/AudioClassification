import os
import numpy as np
import librosa
from tqdm import tqdm

# Constants
SAMPLE_RATE = 22050
BLOCK_SIZE = 1024
HOP_SIZE = 512
MEL_BANDS = 128
DURATION = 1.0


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

    X = np.array(X, dtype=object)
    y = np.array(y, dtype=object)
    return X, y


if __name__ == '__main__':
    X, y = process_audio_files(
        'path_to_IRMAS_Training_Data')
    np.save('C:\\AudioClassification\\npy_data\\X\\[set_name].npy', X)
    np.save(
        'C:\\AudioClassification\\npy_data\\y\\[set_name].npy', y)
