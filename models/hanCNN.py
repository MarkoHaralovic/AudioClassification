
import numpy as np
import tensorflow_addons as tfa
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.metrics import label_ranking_average_precision_score

X = np.load(
    "path_to_augmentedX.npy", allow_pickle=True)
Y = np.load("path_to_augmentedY.npy",
            allow_pickle=True).astype(np.float16)


# X = np.reshape(X, (*X.shape, 1)).astype(np.float32)
X = (X - X.mean()) / X.std()

X, Y = shuffle(X, Y, random_state=10)


def hanCNN(input_shape, N_CLASSES=11):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu',
               padding='same', input_shape=input_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(3, 3), strides=(1, 1)),
        Dropout(0.25),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(3, 3), strides=(1, 1)),
        Dropout(0.25),


        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(3, 3), strides=(1, 1)),
        Dropout(0.25),

        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        GlobalMaxPooling2D(),

        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(11, activation='sigmoid'),
    ])
    optimizer = Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=[tfa.metrics.F1Score(num_classes=11, average='micro')])
    model.summary()
    return model


checkpoint_filepath = 'C:\\AudioClassification\\checkpoint\\hanCNN.{epoch:02d}.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='auto',
    save_best_only=False,
    verbose=1,
    save_freq='epoch')

# load model
# loaded_model = tf.keras.models.load_model('path/to/your/checkpoint/directory/model.XX.h5')


input_shape = X.shape[1:]
# print("input shape ", input_shape)
model = hanCNN(input_shape=input_shape)
# model = load_model('path_to_h5_model')
lr_reduction = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=5, verbose=1)
early_stopping = EarlyStopping(
    monitor='val_loss', patience=7, verbose=1)  # increase to 7 myb
history = model.fit(X, Y, validation_split=0.2,
                    epochs=100, batch_size=16, verbose=1,
                    callbacks=[lr_reduction, early_stopping, model_checkpoint_callback])

model.save('set_name_for_h5_model')
