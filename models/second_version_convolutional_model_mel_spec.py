import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.regularizers import l2
from tqdm import tqdm
import tensorflow as tf
from IPython.display import clear_output
from tqdm.keras import TqdmCallback
from sklearn.preprocessing import StandardScaler
import json
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Define constants
SAMPLE_RATE = 22000  # 22000 Hz
DURATION = 1.0  # duration of audio segments in seconds
NUM_MFCC = 13  # number of Mel frequency cepstral coefficients to extract
IRMAS_CLASSES = 11
OPENMIC_CLASSES = 9


##   this model was initially used, it is a model for predominant instrument predictions    ##

# custom callback implemented
class TqdmProgressCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self, 'progress_bar'):
            self.progress_bar.close()
        self.progress_bar = tqdm(
            total=self.params['steps'], desc=f'Epoch {epoch+1}/{self.epochs}', unit='step')
        clear_output(wait=True)

    def on_batch_end(self, batch, logs=None):
        self.progress_bar.update(1)

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.close()


def get_conv_model(input_shape, N_CLASSES=IRMAS_CLASSES):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
              padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((3, 3), padding='same'))
    model.add(Dropout(0.25))
    num_filters = 32
    while num_filters != 512:
        model.add(Conv2D(num_filters, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((3, 3), padding='same'))
        model.add(Dropout(0.25))
        num_filters *= 2

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(N_CLASSES, activation='softmax'))
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def load_saved_data(mfcc_data_path, labels_path):
    X_mfcc = np.load(mfcc_data_path, allow_pickle=True)
    y = np.load(labels_path, allow_pickle=True).astype(np.float32)

    X_mfcc = X_mfcc.astype(np.float32)
    # Normalize the MFCCs -> previosu to this, accuracy was 7% and loss wass too high
    X_mfcc_2d = X_mfcc.reshape(-1, X_mfcc.shape[-1])
    scaler = StandardScaler()
    X_mfcc_2d = scaler.fit_transform(X_mfcc_2d)
    X_mfcc = X_mfcc_2d.reshape(X_mfcc.shape)

    # Check for NaN values
    if np.isnan(X_mfcc).any():
        print("Warning: NaN values found in the data.")

    # Add the channel dimension
    X_mfcc = np.reshape(X_mfcc, (*X_mfcc.shape, 1))

    return X_mfcc, y


mel_spec_data_path_irmas = 'path_to_mfcc_extracted_data'
labels_path_irmas = 'path_to_single_y_labels'

X_mel_spec_irmas, y_irmas = load_saved_data(
    mel_spec_data_path_irmas, labels_path_irmas)

X_mel_spec_irmas, y_irmas = shuffle(X_mel_spec_irmas, y_irmas, random_state=42)


# Train the model using mel spectrograms  from IRMAS dataset
input_shape_irmas = X_mel_spec_irmas.shape[1:]
model_mel_spec_irmas = get_conv_model(
    input_shape=input_shape_irmas, N_CLASSES=IRMAS_CLASSES)
tqdm_callback_irmas = TqdmProgressCallback()
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
history_mel_spec_irmas = model_mel_spec_irmas.fit(
    X_mel_spec_irmas, y_irmas, validation_split=0.2, epochs=100, batch_size=128, verbose=1, callbacks=[early_stopping])
# callbacks=[TqdmCallback(verbose=1)])


model_mel_spec_irmas.save('save_model_as.h5')
