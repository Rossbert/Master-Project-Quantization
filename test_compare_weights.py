import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import Quantization
from keras.engine.functional import Functional

# LOAD_PATH_Q_AWARE = "./model/model_q_aware_final_02"
LOAD_PATH_Q_AWARE = "./model/model_q_aware_ep5_2023-07-02_16-50-58"

# Load Q Aware model
with tfmot.quantization.keras.quantize_scope():
    q_aware_model : Functional = tf.keras.models.load_model(LOAD_PATH_Q_AWARE)

q_model_info = Quantization.QuantizedModelInfo(q_aware_model)

# Weights extraction
i = 0
idx = 0
for j, layer in enumerate(q_aware_model.layers):
    idx = q_model_info.layers_indexes[i]
    key = q_model_info.keys[i]
    if j == idx:
        if 'quantize_layer' not in key:
            weight = layer.get_weights()
            # print(f"Original kernel:\n{weight[0]}\n")
            # print(q_model_info.quantized_weights[key])
            # print(f"Dequantized kernel:\n{q_model_info.quantized_weights[key]*q_model_info.kernel_scales[key]}\n")
            dequantized_kernel = q_model_info.quantized_weights[key]*q_model_info.kernel_scales[key]
            print(f"Kernel original max: {np.max(np.abs(weight[0]))}")
            print(f"Kernel original min: {np.min(np.abs(weight[0]))}")
            print(f"Quantized kernel max: {np.max(np.abs(dequantized_kernel))}")
            print(f"Quantized kernel min: {np.min(np.abs(dequantized_kernel))}")
            print(f"Kernel difference max:\n{np.max(np.unique(weight[0] - dequantized_kernel))}")
            print(f"Kernel difference min:\n{np.min(np.unique(weight[0] - dequantized_kernel))}\n")
            
            # print(f"Original biases:\n{weight[1]}\n")
            # print(q_model_info.quantized_biases[key])
            # print(f"Dequantized biases:\n{q_model_info.quantized_biases[key]*q_model_info.bias_scales[key]}\n")
            dequantized_biases = q_model_info.quantized_biases[key]*q_model_info.bias_scales[key]
            print(f"Biases original max: {np.max(np.abs(weight[1]))}")
            print(f"Biases original min: {np.min(np.abs(weight[1]))}")
            print(f"Quantized biases max: {np.max(np.abs(dequantized_biases))}")
            print(f"Quantized biases min: {np.min(np.abs(dequantized_biases))}")
            print(f"Biases difference max:\n{np.max(np.unique(weight[1] - dequantized_biases))}")
            print(f"Biases difference min:\n{np.min(np.unique(weight[1] - dequantized_biases))}\n")
        i += 1