# Audio Classification

This Git repository contains code for polyphonic audio classification using a Convolutional Neural Network (CNN) model. The audio is loaded, processed, and then fed into the CNN for polyphonic instrument detection, with learning and validation based on the IRMAS dataset. The model achieves state-of-the-art precision, with a Hamming accuracy of over 90%.

## How to Use This Repository

### Models
Original codes for each model are located in the `models` folder. Modify the relative paths to the IRMAS datasets as needed.

### Pre-trained Models
Models are saved as `.h5` files in the `h5_models` folder, which can be loaded for use.

### Testing Models
Test a model by loading it from the `.h5` file and calling `model.predict` on a preprocessed audio.

### Audio Augmentation
Audio augmentation code is in the `audio_augment` folder. Use `augment.py` and specify paths to the IRMAS training set. Save the processed data in the `npy_data` folder to be used as input to the `augpoly78.h5` model.

### Audio Preprocessing
Code for audio preprocessing is in the `audio_preprocessing` folder. It includes:
  - `generating_one_second_intervals.py` for generating one-second intervals.
  - `audio_proc_singular_labels` for generating single labels for files (used with `mel_spec_irmas_singleton.h5`).
  - `preprocess_spectrograms.py` for generating mel spectrograms (used with `78kratimenos.h5`).
  - `preprocess_mel_spec_chromagram_spec_contrast.py` for generating mel spectrograms, chromagrams, and spectral contrast (used with `augpoly78.h5`).

### Testing Code
The `testing` folder contains:
  - `F1_score.py`, a custom Python file to calculate the F1 score of a model.
  - `test.py` to obtain the aggregated result and precision of the model.
  - `test_predict.py` for the result, precision, and JSON output. The JSON output is saved in the `validation_json_files` folder.

### Images
The `IMAGES` folder contains useful images for this project.

### Technical Documentation
Located in the project directory, it provides technical specifications, descriptions, and a user manual.

### API
The `API` folder contains a `Dockerfile` for setting up the Flask app and TensorFlow Serving. 
`app.py` is a Flask application that can be run using the command `python app.py`.

### TensorFlow Serving
Pull the TensorFlow Serving Docker image and set up a new container as described in the documentation. The structure of the model directory should be:

/path/to/your/model_directory/
├── model_name
│ └── 1
│ ├── saved_model.pb
│ └── variables
│ ├── variables.data-00000-of-00001
│ └── variables.index


### Defined Constants
Constants for `augpoly78` and `78kratimenos`:
  - `SAMPLE_RATE = 22050`
  - `BLOCK_SIZE = 1024`
  - `HOP_SIZE = 512`
  - `MEL_BANDS = 128`
  - `DURATION = 1.0`

Constants for other models:
  - `SAMPLE_RATE = 22050`
  - `BLOCK_SIZE = round(46.4 * SAMPLE_RATE / 1000)`
  - `HOP_SIZE = round(259.41)`
  - `MEL_BANDS = 96`
  - `DURATION = 1.0`

### Loading the `augpoly` Model
Load the `augpoly` model with the `hamming_accuracy` function using the following command:

```python
model = tf.keras.models.load_model(
    'path_to_augpoly78.h5',
    custom_objects={
        'F1Score': F1Score,
        'hamming_accuracy': hamming_accuracy
    })
## Generating JSON File Predictions
Use `predictions_test_data_augpoly.py` to generate a JSON file with predictions for the main model on the test data. The file is saved in the `testing_json_file` folder.

## JSON Prediction Files
- `augpoly78_test.json` contains predictions with a threshold set at 0.5.
- `augpoly78_test1.json` contains predictions with a threshold set at 0.3.

## Main Prediction JSON File
The main prediction JSON file is `augpoly78_test.json` (threshold 0.5).

## Docker Compose
Use the `docker-compose.yml` file to run and manage the Flask app and TensorFlow Serving services in containers.

## Documentation
The repository contains both Technical and Project documentation.

## Model Names
- `78kratimenos.h5`
- `augpoly78.h5`
- `mel_spec_irmas_singleton.h5`

For detailed instructions on setting up and running the models, refer to the Technical Documentation.

## Contributions and Feedback
We welcome contributions and feedback on this project. Please follow the guidelines in the `CONTRIBUTING.md` file for contributions.

## License
This project is licensed under the MIT License.

## Acknowledgments
This work is based on research funded by [Your Institution/Organization Name].
