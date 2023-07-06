import numpy as np
import numpy.typing as npt
from typing import List
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import Quantization
from keras.engine.functional import Functional

# Number of partitions for batch analysis
N_PARTITIONS = 2
# Only for first convolution
SPLIT_INDEX = 3

LOAD_PATH_Q_AWARE = "./model/model_q_aware_ep5_2023-07-02_16-50-58"

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# Load Q Aware model
with tfmot.quantization.keras.quantize_scope():
    q_aware_model : Functional = tf.keras.models.load_model(LOAD_PATH_Q_AWARE)

q_model_info = Quantization.QuantizedModelInfo(q_aware_model)

# Preparing the quantized test set
quantized_test_images = np.round(test_images[:,:,:,np.newaxis]/q_model_info.output_scales[q_model_info.keys[0]]).astype(int)

# Generating the split models
m1_quantized, _ = Quantization.split_model_mixed(q_aware_model, q_model_info, start_index = SPLIT_INDEX, separation_mode = Quantization.SeparationMode.first_quantized_weights)
m1_nonquantized, _ = Quantization.split_model_mixed(q_aware_model, q_model_info, start_index = SPLIT_INDEX, separation_mode = Quantization.SeparationMode.first_floating_weights)

IMAGE = 0
CHANNEL = 10 # 10
kernel_pos = (slice(None), slice(None), 0, CHANNEL)
image_pos_section = (IMAGE, slice(16, 21), slice(16, 21))
output_pos_section = (IMAGE, slice(16, 21), slice(16, 21), CHANNEL)

i = 0
idx = q_model_info.layers_indexes[i]
key = q_model_info.keys[i]
dequantized_images = quantized_test_images*q_model_info.output_scales[key]
print(f"Original test image:\n{test_images[image_pos_section]}")
print(f"Quantized test image:\n{quantized_test_images[image_pos_section + (0,)]}")
print(f"Dequantized test image:\n{dequantized_images[image_pos_section + (0,)]}\n")

i = 1
idx = q_model_info.layers_indexes[i]
key = q_model_info.keys[i]
weight = q_aware_model.layers[idx].get_weights()
dequantized_kernel = q_model_info.quantized_weights[key]*q_model_info.kernel_scales[key]
dequantized_biases = q_model_info.quantized_biases[key]*q_model_info.bias_scales[key]

print(f"Original kernel weights:\n{weight[0][kernel_pos]}")
print(f"Quantized kernel weights:\n{q_model_info.quantized_weights[key][kernel_pos]}")
print(f"Dequantized kernel weights:\n{dequantized_kernel[kernel_pos]}\n")
print(f"Original biases:\n{weight[1]}")
print(f"Quantized biases:\n{q_model_info.quantized_biases[key]}")
print(f"Dequantized biases:\n{dequantized_biases}\n")

quantized_conv = Quantization.model_predict_by_batches(
    data_input = quantized_test_images,
    n_partitions = N_PARTITIONS,
    model = m1_quantized,
    ).astype(np.int32)

original_conv = Quantization.model_predict_by_batches(
    data_input = test_images[:,:,:,np.newaxis],
    n_partitions = N_PARTITIONS,
    model = m1_nonquantized,
    )

dequantized_convolution = quantized_conv*q_model_info.bias_scales[key]

print(f"Original convolution\n{original_conv[output_pos_section]}")
print(f"Quantized convolution\n{quantized_conv[output_pos_section]}")
print(f"Dequantized convolution\n{dequantized_convolution[output_pos_section]}\n")

kernel_eval = (slice(None), slice(None), slice(None), CHANNEL)
test_weights = weight[0][kernel_eval][:,:,:,np.newaxis]
test_output_pos_section = (0, slice(16, 21), slice(16, 21), 0)
test_out = tf.nn.conv2d(
    input = test_images[IMAGE,:,:][np.newaxis,:,:,np.newaxis],
    filters = test_weights,
    strides = 1,
    padding = "VALID",
).numpy()

print(f"Manual tensorflow convolution:\n{test_out[test_output_pos_section]}")