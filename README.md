# AudioClassification

This git repository consists of the code for polyphonic audio classification . The audio is loaded, processed, and then sent as an input to a  Convolutional Neural Network Model. The CNN is used for polyphonic instrument detection, with learning and validation based on the IRMAS dataset. Model rocords state of the art precision, with Hamming accuracy over 90%.


HOW TO USE THIS GIT:

1. in folder models, there are original codes for each model. Only thing to modify are relative paths to IRMAS datasets.
2. those models are saved as h5 files in h5_models folder, from where they can be loaded from. 
3. test a model by loading it from h5 file and calling model.predict on a preprocessed audio
4.code for audio augmentation is in folder audio_augment. Use augment.py. Specify path to IRMAS_training_set and set the names for labels and for X values (values for the model). The data will be saved to npy_data corresponding folder where can be loaded from and sent as an input to augpoly78.h5 model.
5.code for audio preprocessing is in folder audio_preprocessing. 
 - To generate one second intervals , use generating_one_second_intervals.py. 
 - to generate single labels for files (used for mel_spec_irmas_singleton.h5 model), use audio_proc_singular_labels
 - to generate mel spectrograms  (used for 78kratimenos.h5), use preprocess_spectrograms.py
 - to generate mel spectrograms, chromagrams and spectral contrast (used for augpoly78.h5) , use preprocess_mel_spec_chromagram_spec_contrast.py
6. code for testing is located inside testing folder
-F1_score.py is custom python file to calculate F! score of an model
- call test.py on the model you want to test to get aggregated result and precision of the model
- call test_predict.py to get the result, precision and json output (json output is saved based on h5 model name to validation_json_files, where predictions are saved, with accuracy calculated at each step and mean acc is calculated in the end)
7.IMAGES - folder with useful images for this project
8.Techinical documentation is located in the project directory, use it for techinical specifications, descriptions and user manual documentation for this project.
9.API folder contains Dockerfile which should be run ( described in the Technical documentation), with defined requirements in the project directory. There is a file named h5_to_pb_model.py , which needs to be run only in the beggining to get the pb file to be used for tfserving and connection with API. app.py is Flask application that can be run with command python app.py , upon setting up docker, tfserving and running command: 
docker run -p 8500:8500 -p 8501:8501 --name=tfserving --mount type=bind,source= path_to_API\\model,target=/models/[define_which_model_to_use] -e MODEL_NAME=[previously_defined_model_number] -t tensorflow/serving
MODEL NAMES:
1 - 78kratimenos.h5
2 - augpoly78.h5
3 - mel_spec_irmas_singleton.h5
(docker can be installed at this link:https://www.docker.com/products/docker-desktop/)

You will have to ull the TensorFlow Serving Docker image from Docker Hub:
docker pull tensorflow/serving

Start a new TensorFlow Serving container:

To start a new container with the TensorFlow Serving image, you'll need to provide a directory containing your TensorFlow model(s). This directory should have a structure like this:

/path/to/your/model_directory/
├── model_name
│   └── 1
│       ├── saved_model.pb
│       └── variables
│           ├── variables.data-00000-of-00001
│           └── variables.index

Then you run the command previously mentioned:
docker run -p 8500:8500 -p 8501:8501 --mount type=bind,source=/path/to/your/model_directory/,target=/models/model_name -e MODEL_NAME=model_name -t tensorflow/serving

->Replace /path/to/your/model_directory/ with the actual path to your model directory and model_name with the name of your model.

10. When using API, ypu will get the https:\\ where you can access the website. Website is defined in the API/templates folder, under the name index.html. You will be able to upload a file and get predictions from model you specify in the previous command.

11. Defined constants:
- for augpoly78 and 78kratimenos, use constants:
SAMPLE_RATE = 22050
BLOCK_SIZE = 1024
HOP_SIZE = 512
MEL_BANDS = 128
DURATION = 1.0

-for other models use constants:
SAMPLE_RATE = 22050
BLOCK_SIZE = round(46.4 * SAMPLE_RATE / 1000)
HOP_SIZE = round(259.41)
MEL_BANDS = 96
DURATION = 1.0 

12. model augpoly needs to have hamming_accuracy function defined in the folder and can be loaded with this command:

model = tf.keras.models.load_model(
    'path_to_augpoly78.h5',
    custom_objects={
        'F1Score': F1Score,
        'hamming_accuracy': hamming_accuracy
    })

13. file predictions_test_data_augpoly.py is used for generating json file predictions for main model that we have on the test data. The json file is saved to testing_json_file under name: augpoly78_test.json.

14.augpoly78_test.json is json file containing predictions of our main model on the test data with treshold set at 0.5,
and augpoly78_test1.json is json file containing predictions of our main model on the test data with treshold set at 0.3

15. Main prediction json file is : augpoly78_test.json ( treshold 0.5) 

16. docker-compose-xml : by using this docker-compose.yml file, you can easily run and manage the Flask app and TensorFlow Serving services together in their respective containers, ensuring an isolated and consistent environment for both services,file builds the Docker image from the ./API directory and builds the Docker image using the Dockerfile located in ./tfserving/Dockerfile

17. git contains Technical documentation and Project documentation as well.
