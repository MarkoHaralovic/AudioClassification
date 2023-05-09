import requests
from flask import Flask, request, jsonify, render_template
import os
import tempfile
from werkzeug.utils import secure_filename
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom
from tqdm import tqdm
import glob
import json

# SAMPLE_RATE = 22050
# BLOCK_SIZE = round(46.4 * SAMPLE_RATE / 1000)
# HOP_SIZE = round(259.41)
# MEL_BANDS = 96
# DURATION = 1.0

SAMPLE_RATE = 22050
BLOCK_SIZE = 1024
HOP_SIZE = 512
MEL_BANDS = 128
DURATION = 1.0


def preprocess_audio(audio_path):
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


def predict_windows(model, log_mel_spectrogram):
    # Add a channel dimension to the input
    log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=-1)

    payload = {"instances": [log_mel_spectrogram.tolist()]}
    response = requests.post(
        "http://localhost:8501/v1/models/2:predict", json=payload)
    try:
        pred = response.json()["predictions"][0]
    except KeyError:
        print("Error: predictions key not found in TensorFlow serving API response")
        print("Response:", response.json())
        return None
    return np.array([pred])


def aggregate_predictions(predictions):
    summed = np.sum(predictions, axis=0)
    return summed / np.sum(summed)


model = load_model(
    'C:\\Polyphonic_audio_classification\\second_model_mel_spec_irmas_singleton_lr_0.00001.h5')

app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file in request"}), 400

    audio_file = request.files['audio']
    filename = secure_filename(audio_file.filename)
    temp_file_path = os.path.join(tempfile.gettempdir(), filename)
    audio_file.save(temp_file_path)

    log_mel_spectrograms = preprocess_audio(temp_file_path)

    os.remove(temp_file_path)

    predictions = []
    for spectrogram in log_mel_spectrograms:
        prediction = predict_windows(model, spectrogram)
        if prediction is not None:
            predictions.append(prediction[0])
        else:
            return jsonify({"error": "Failed to get predictions from TensorFlow serving API"}), 500

    aggregated_predictions = aggregate_predictions(predictions)

    instruments = ['cel', 'cla', 'flu', 'gac', 'gel',
                   'org', 'pia', 'sax', 'tru', 'vio', 'voi']
    threshold = 0.3
    instrument_dict = {instrument: 0 for instrument in instruments}

    for i, value in enumerate(aggregated_predictions):
        if value > threshold:
            instrument_dict[instruments[i]] = 1
    # top_n = 2
    # # Get the indices of the top N predictions
    # top_n_indices = np.argsort(aggregated_predictions)[-top_n:]

    # Set the corresponding instruments to 1 in the instrument_dict
    # for idx in top_n_indices:
    #     instrument_dict[instruments[idx]] = 1
    return jsonify(instrument_dict)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
