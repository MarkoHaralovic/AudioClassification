import os
import numpy as np
import librosa
from tqdm import tqdm

# Constants
SAMPLE_RATE = 22050  # Sample rate for loading audio files
DURATION = 1.0  # Duration (in seconds) of each audio segment


def parse_label(folder_name):
    """
    Parse the label from the folder name and create a binary representation of the label.
    """
    valid_instruments = ['cel', 'cla', 'flu', 'gac',
                         'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
    binary_labels = [1 if inst ==
                     folder_name else 0 for inst in valid_instruments]
    return binary_labels


def segment_audio(audio_path):
    """
    Load an audio file, convert it to mono, and segment it into fixed-length segments.
    """
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=False)
    audio = librosa.to_mono(audio)

    # Calculate segment length in samples
    segment_length = int(SAMPLE_RATE * DURATION)

    # Slice audio into segments
    audio_segments = [audio[i:i + segment_length]
                      for i in range(0, len(audio), segment_length)]
    audio_segments = [segment for segment in audio_segments if len(
        segment) == segment_length]

    return audio_segments


def process_audio_files(input_dir):
    """
    Process all audio files in the specified directory, segmenting them and extracting their labels.
    """
    y = []

    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing files in {root}"):
            if file.endswith('.wav') or file.endswith('.ogg'):
                audio_path = os.path.join(root, file)
                folder_name = os.path.basename(root)
                label = parse_label(folder_name)
                try:
                    audio_segments = segment_audio(audio_path)
                    for _ in audio_segments:
                        y.append(label)

                except Exception as e:
                    print(
                        f"Failed to process audio file {audio_path}. Error: {type(e).__name__}: {str(e)}")

    y = np.array(y, dtype=object)
    return y


if __name__ == '__main__':
    y = process_audio_files(
        'path_to_IRMAS_Training_Data')
    np.save('define_path_to_singleY.npy', y)
