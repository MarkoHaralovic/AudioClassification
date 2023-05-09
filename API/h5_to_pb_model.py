import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_addons.metrics import F1Score
import numpy as np


def hamming_accuracy(y_true, y_pred):
    y_true = K.round(K.clip(y_true, 0, 1))
    y_pred = K.round(K.clip(y_pred, 0, 1))
    equal_elements = K.cast(K.equal(y_true, y_pred), K.floatx())
    return K.mean(equal_elements)

# # 1.
# model = load_model(
#     'C:\\AudioClassification\\h5_models\\78kratimenos.h5',
#     custom_objects={'Addons>F1Score': F1Score})

# tf.saved_model.save(
#     model, "C:\\AudioClassification\\API\\model\\1")


# 2.
model = load_model(
    'C:\\AudioClassification\\h5_models\\augpoly78.h5',
    custom_objects={
        'Addons>F1Score': F1Score,
        'hamming_accuracy': hamming_accuracy
    }
)
tf.saved_model.save(
    model, "C:\\AudioClassification\\API\\model\\2")

# # 3.
# model = load_model(
#     'C:\\AudioClassification\\h5_models\\mel_spec_irmas_singleton.h5',
#     custom_objects={'Addons>F1Score': F1Score})

# tf.saved_model.save(
#     model, "C:\\AudioClassification\\API\\model\\3")
