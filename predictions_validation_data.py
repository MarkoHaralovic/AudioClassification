import numpy as np
import librosa
import os
import glob
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom
from tqdm import tqdm
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow_addons.metrics import F1Score

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
    Load an audio file, convert it to mono, normalize it, segment it into fixed-length segments,
    and convert each segment into a set of waveform frames.

    :param audio_path: str, path to the audio file
    :return: list, waveform frames for each segment of the audio file
    :raises InvalidAudioPathError: if the provided audio_path is not valid
    """
    if not os.path.isfile(audio_path):
        raise InvalidAudioPathError(f"Invalid audio path: {audio_path}")

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
    Calculate the validation accuracy based on the ground truth and the predicted instruments.

    :param file_path_txt: str, path to the ground truth text file
    :param predictions: dict, instrument predictions
    :param threshold: float, threshold value for predicting instruments
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
    :return: np.array, log-mel spectrograms
    """
    X = []

    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing files in {root}"):
            if file.endswith('.wav') or file.endswith('.ogg'):
                audio_path = os.path.join(root, file)
                try:
                    log_mel_spectrograms = preprocess_audio(audio_path)
                    for spectrogram in log_mel_spectrograms:
                        X.append(spectrogram)

                except Exception as e:
                    print(
                        f"Failed to process audio file {audio_path}. Error: {type(e).__name__}: {str(e)}")

    X = np.array(X, dtype=object)
    return X


def predict_windows(model, log_mel_spectrogram):
    """
    :param model: Keras model, trained model for predictions
    :param log_mel_spectrogram: np.array, input log-mel spectrogram
    :return: np.array, model predictions
    """
    # Make predictions for an input log-mel spectrogram using the given model.
    return model.predict(np.array([log_mel_spectrogram]))


def aggregate_predictions(predictions):
    """
    Aggregate predictions from multiple windows by summing them and normalizing.
    :param predictions: list of np.arrays, predictions for each window
    :return: np.array, aggregated predictions
    """
    summed = np.sum(predictions, axis=0)
    return summed / np.sum(summed)


def hamming_accuracy(y_true, y_pred):
    """
    Compute the hamming accuracy between the true labels and predicted labels.

    : param y_true: tensor, true labels
    : param y_pred: tensor, predicted labels
    : return: float, hamming accuracy
    """
    y_true = K.round(K.clip(y_true, 0, 1))
    y_pred = K.round(K.clip(y_pred, 0, 1))
    equal_elements = K.cast(K.equal(y_true, y_pred), K.floatx())
    return K.mean(equal_elements)


def test_model(model, test_data_path, threshold=0.31):
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


model = tf.keras.models.load_model(
    'C:/AudioClassification/h5_models/augpoly78.h5',
    custom_objects={
        'F1Score': F1Score,
        'hamming_accuracy': hamming_accuracy
    })

test_data_path = 'IRMAS_validation_data_path'
results = test_model(model, test_data_path)
with open("validation_json_files", "w") as outfile:
    json.dump(results, outfile)
