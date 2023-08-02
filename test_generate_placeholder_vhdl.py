import csv
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import Quantization
from keras.engine.functional import Functional

# Number of partitions for batch analysis
N_PARTITIONS = 2
# First index of q_aware_model
SPLIT_INDEX = 3
SEPARATION_MODE = Quantization.SeparationMode.first_quantized_weights
# Set section
RANGE_START = 50
RANGE_END = 150

# LOAD_PATH_Q_AWARE = "./model/model_q_aware_final_01"
ID = "ep5_2023-07-02_16-50-58"
LOAD_PATH_Q_AWARE = "./model/model_q_aware_" + ID
SAVE_CONV_OUT_NPY = "./model/" + ID[-8:] + "_first_conv_out.npy"
SAVE_LABELS_NPY = "./model/" + ID[-8:] + "_labels.npy"

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

new_test_images = test_images[RANGE_START:RANGE_END]
new_test_labels = test_labels[RANGE_START:RANGE_END]

# Load Q Aware model
with tfmot.quantization.keras.quantize_scope():
    q_aware_model : Functional = tf.keras.models.load_model(LOAD_PATH_Q_AWARE)

q_model_info = Quantization.QuantizedModelInfo(q_aware_model)

# Preparing the quantized test set
quantized_test_images = np.round(new_test_images[:,:,:,np.newaxis]/q_model_info.output_scales[q_model_info.keys[0]]).astype(int)

# Generating the split models
model_1, model_2 = Quantization.split_model_mixed(q_aware_model, q_model_info, start_index = SPLIT_INDEX, first_part_mode = SEPARATION_MODE)

quantized_conv, test_loss, test_accuracy = Quantization.model_parts_predict_by_batches(
    data_input = quantized_test_images,
    test_labels = new_test_labels,
    n_partitions = N_PARTITIONS,
    q_model_info = q_model_info,
    model_1 = model_1,
    model_2 = model_2,
    evaluation_mode = Quantization.ModelEvaluationMode.manual_saturation,
    start_index = SPLIT_INDEX
    )

print(f"Model test accuracy: {test_accuracy:.2%}")
print(f"Model test loss: {test_loss:.6f}\n")

np.save(SAVE_CONV_OUT_NPY, quantized_conv)
np.save(SAVE_LABELS_NPY, new_test_labels)
