# Import necessary libraries
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import tensorflow_addons as tfa
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from sklearn.utils import shuffle

# Load input data (features and labels) from .npy files
X = np.load("path_to_X.npy", allow_pickle=True)
y = np.load("path_to_y.npy", allow_pickle=True)

# Load the pre-trained model
model = load_model('path_to_h5_file')

# Initialize true positive, false positive, and false negative counters
tp = 0
fp = 0
fn = 0

# Iterate through every 10th sample in the dataset
for i in range(0, len(X), 10):
    print(i, len(X))

    # Reshape the input data to match the model's input shape
    X[i] = np.reshape(X[i], X[i].shape + (1,)).astype(np.float32)

    # Initialize a non-binary label
    non_binary_label = np.zeros(11)

    # Accumulate model predictions for all frames in a sample
    for j in range(len(X[i])):
        non_binary_label1 = np.squeeze(model.predict(
            np.expand_dims(X[i][j], axis=0), verbose=0))
        non_binary_label = non_binary_label1 + non_binary_label

    # Normalize the non-binary label
    non_binary_label /= np.max(non_binary_label)

    # Convert the non-binary label to binary
    pred_binary_label = np.zeros(len(non_binary_label))
    for j in range(len(non_binary_label)):
        if (non_binary_label[j] >= 0.5):
            pred_binary_label[j] = 1

        # Update true positive, false positive, and false negative counters
        if (pred_binary_label[j] == 1):
            if (y[i][j] == 1):
                tp += 1
            if (y[i][j] == 0):
                fp += 1
        if (pred_binary_label[j] == 0):
            if (y[i][j] == 1):
                fn += 1

    # Calculate Precision, Recall, and F1 score
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F1 = 2 * P * R / (P + R)
    print(F1)
