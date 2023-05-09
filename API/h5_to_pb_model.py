import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_addons.metrics import F1Score

# # 1.
# model = load_model(
#     'C:\\AudioClassification\\h5_models\\78kratimenos.h5',
#     custom_objects={'Addons>F1Score': F1Score})

# tf.saved_model.save(
#     model, "C:\\AudioClassification\\API\\model\\1")

# 2.
model = load_model(
    'C:\\AudioClassification\\h5_models\\augpoly78.h5',
    custom_objects={'Addons>F1Score': F1Score})

tf.saved_model.save(
    model, "C:\\AudioClassification\\API\\model\\2")

# # 3.
# model = load_model(
#     'C:\\AudioClassification\\h5_models\\mel_spec_irmas_singleton.h5',
#     custom_objects={'Addons>F1Score': F1Score})

# tf.saved_model.save(
#     model, "C:\\AudioClassification\\API\\model\\3")
