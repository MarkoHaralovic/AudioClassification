from tqdm import tqdm
import os
import librosa
import numpy as np
from scipy.io.wavfile import write

# Define constants
SAMPLE_RATE = 22050  # 22000 Hz
DURATION = 1.0  # duration of audio segments in seconds
NUM_MFCC = 13  # number of Mel frequency cepstral coefficients to extract
IRMAS_CLASSES = 11  # number of classes in IRMAS dataset (all unique)
# number of classes in OpenMIC dataset that are common/shared with IRMAS dataset
OPENMIC_CLASSES = 9


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
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    # Downsample audio to 22000 Hz
    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    # Apply envelope function to audio file
    # mask, env = envelope(audio, sr, threshold)
    # audio = audio[mask]
    # Segment audio into one-second intervals

    # downmix to mono
    audio = audio.to_mono(audio)

    # normalizing the  data
    r = librosa.feature.rms(y=audio)
    a = np.sqrt((len(audio) * r**2) / np.sum(audio**2))
    audio = audio * a

    for segment in range(int(len(audio) / (SAMPLE_RATE * DURATION))):
        start = int(segment * SAMPLE_RATE * DURATION)
        end = int(start + SAMPLE_RATE * DURATION)
        # Extract Mel spectrogram and MFCC
        mfccs = librosa.feature.mfcc(
            y=audio, sr=SAMPLE_RATE, n_mfcc=NUM_MFCC)
        mel_spectrogram = extract_mel_spectrogram(
            audio, SAMPLE_RATE)
        # Create output path for segmented audio
        file_name, file_ext = os.path.splitext(os.path.basename(audio_path))
        output_file_name = f"{file_name}_{segment + 1}{file_ext}"
        output_file_path = os.path.join(output_dir, output_file_name)
        # Save segmented audio to output directory
        write(output_file_path, SAMPLE_RATE, audio[start:end])


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
                try:
                    segment_audio_files(audio_path, output_dir)
                except Exception as e:
                    print("Failed to segment audio file")


if __name__ == '__main__':
    process_audio_files('path_to_original_dataset',
                        'path_for_new_dataset')
