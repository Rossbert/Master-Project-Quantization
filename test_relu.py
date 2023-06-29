import tensorflow as tf
import tensorflow_model_optimization as tfmot
from keras.engine.functional import Functional
import numpy as np
import numpy.typing as npt
import Quantization

OUTPUTS_DIR = "./outputs/"
LOAD_PATH = "./model/model_final_01"
LOAD_PATH_Q_AWARE = "./model/model_q_aware_final_01"

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# Load model
model : Functional = tf.keras.models.load_model(LOAD_PATH)
with tfmot.quantization.keras.quantize_scope():
    q_aware_model : Functional = tf.keras.models.load_model(LOAD_PATH_Q_AWARE)

stop_index = 2
model_output = Quantization.model_begin_predict(model, test_images[:,:,:,np.newaxis], layer_index_stop = stop_index)
q_model_output = Quantization.model_begin_predict(q_aware_model, test_images[:,:,:,np.newaxis], layer_index_stop = stop_index + 1)
Quantization.garbage_collection()
print(f"Model output shape: {model_output.shape}")
print(f"Model output max: {np.max(model_output)}")
print(f"Model output min: {np.min(model_output)}")
print(f"Q model output shape: {q_model_output.shape}")
print(f"Q model output max: {np.max(q_model_output)}")
print(f"Q model output min: {np.min(q_model_output)}")
