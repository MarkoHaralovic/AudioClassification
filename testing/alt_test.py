import numpy as np
import librosa
import os
import glob
from keras.models import load_model
from scipy.ndimage import zoom
from tqdm import tqdm
import tensorflow_addons as tfa
import json
# # Constants for model mel_spec_irmas_singleton.h5
# SAMPLE_RATE = 22050
# BLOCK_SIZE = round(46.4 * SAMPLE_RATE / 1000)
# HOP_SIZE = round(259.41)
# MEL_BANDS = 96
# DURATION = 1.0

# constants for models: 78kratimenos.h5 and augpoly78.h5
SAMPLE_RATE = 22050
BLOCK_SIZE = 1024
HOP_SIZE = 512
MEL_BANDS = 128
DURATION = 1.0


def preprocess_audio(audio_path):
    """
    Preprocess audio file by converting it to mono, normalizing it, and
    segmenting it into one-second intervals. Then, compute the log-mel
    spectrograms for each audio segment.

    :param audio_path: str, path to the audio file
    :return: list of log-mel spectrograms for each audio segment
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


def validation_accuracy(file_path_txt, predictions, threshold=0.5):
    """
    Calculate the validation accuracy based on the ground truth from
    the input file and the predicted values.

    :param file_path_txt: str, path to the ground truth file
    :param predictions: dict, predicted values for each instrument
    :param threshold: float, threshold for prediction acceptance
    :return: float, validation accuracy
    """
    correct = 0
    total = 0

    with open(file_path_txt, 'r') as f:
        ground_truth = [line.strip() for line in f.readlines()]

    for instrument, value in predictions.items():
        if value > threshold:
            total += 1
            if instrument in ground_truth:
                correct += 1

    return correct / len(ground_truth)


def process_audio_files(input_dir):
    """
    Process all audio files in the input directory and generate log-mel
    spectrograms for each file.

    :param input_dir: str, path to the input directory
    :return: np.array, log-mel spectrograms for all processed files
    """
    X = []

    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing files in {root}"):
            if file.endswith('.wav') or file.endswith('.ogg'):
                audio_path = os.path.join(root, file)
                try:
                    log_mel_spectrograms = preprocess_audio(audio_path)
                    X.append(log_mel_spectrograms)
                    for spectrogram in log_mel_spectrograms:
                        X.append(spectrogram)

                except Exception as e:
                    print(
                        f"Failed to process audio file {audio_path}. Error: {type(e).__name__}: {str(e)}")

    X = np.array(X, dtype=object)
    return X


def predict_windows(model, log_mel_spectrogram):
    """
    Make predictions for an input log-mel spectrogram using the given model.
    :param model: Keras model, trained model for predictions
    :param log_mel_spectrogram: np.array, input log-mel spectrogram
    :return: np.array, model predictions
    """
    return model.predict(np.array([log_mel_spectrogram]))


def aggregate_predictions(predictions):
    """
    Aggregate predictions from multiple windows by summing them and normalizing.
    :param predictions: list of np.arrays, predictions for each window
    :return: np.array, aggregated predictions
    """
    summed = np.sum(predictions, axis=0)
    return summed / np.sum(summed)


def test_model(model, test_data_path, threshold=0.5):
    """
    Test the given model on the data from the test_data_path and calculate
    the accuracy of the predictions.
    :param model: Keras model, trained model for predictions
    :param test_data_path: str, path to the test data directory
    :param threshold: float, threshold for prediction acceptance
    :return: dict, results for each test file
    """
    results = {}
    accuracies = []

    instruments = ['cel', 'cla', 'flu', 'gac', 'gel',
                   'org', 'pia', 'sax', 'tru', 'vio', 'voi']

    audio_files = glob.glob(test_data_path + '/*.wav')
    for audio_file in audio_files:
        log_mel_spectrograms = preprocess_audio(audio_file)
        predictions = []

        for spectrogram in log_mel_spectrograms:
            prediction = predict_windows(model, spectrogram)
            predictions.append(prediction[0])

        aggregated_predictions = aggregate_predictions(predictions)
        instrument_dict = {instrument: 0 for instrument in instruments}

        for i, value in enumerate(aggregated_predictions):
            if value > threshold:
                instrument_dict[instruments[i]] = 1

        file_path_txt = audio_file.replace('.wav', '.txt')
        acc = validation_accuracy(file_path_txt, instrument_dict, threshold)
        accuracies.append(acc)

        print(
            f"Predicted instruments for {audio_file}: {instrument_dict}, Accuracy: {acc}")
        results[os.path.basename(audio_file)] = instrument_dict

    mean_accuracy = np.mean(accuracies)
    print(f"Mean accuracy: {mean_accuracy}")

    return results


model = load_model(
    'path_to_h5_file')

test_data_path = 'path_to_IRMAS_validation_data'
results = test_model(model, test_data_path)
with open("78kratimenos_results.json", "w") as outfile:
    json.dump(results, outfile)
