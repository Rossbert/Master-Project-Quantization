import time
import datetime
import os
import csv
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import Quantization
from keras.engine.functional import Functional

""" Test to affect convolution on first layer. 
-
* All outputs on the test set will be affected
* The output channels are generated randomly
* The positions are also generated randomly
* The bit positions to flip belong to a fixed set and are the same per each simulation.

Parameters to be tuned:
- Regarding the output file name, if you don't update the name manually the previous file won't be deleted. New data will be appended to the end of the file instead.
- Number of simulations = repetitions.
- Limit of number of flips in total.
- Bit step that will be flipped in the 32 bit element.
"""

SAVE_FILE_NAME = f"Quant_Split_{datetime.datetime.now().strftime('%Y-%m-%d_%H')}.csv"
N_SIMULATIONS = 2                                       # Number of repetitions of everything
N_FLIPS_LIMIT = 4                                       # Maximum total number of flips per simulation
BIT_STEPS_PROB = 1                                      # Divisor of 32, from 1 to 32

MODELS_DIR = "./model/"
LOAD_PATH_Q_AWARE = MODELS_DIR + "model_q_aware_final_01"
LOAD_TFLITE_PATH = MODELS_DIR + 'tflite_final_01.tflite'
OUTPUTS_DIR = "./outputs/"
SAVE_DATA_PATH = OUTPUTS_DIR + SAVE_FILE_NAME

# Program related constants
# Quantification constants
BIAS_BIT_WIDTH = 32
# Number of partitions for batch analysis
N_PARTITIONS = 2
# Convolutional key
INDEX_KEY_CONV1 = 1

if not os.path.exists(OUTPUTS_DIR):
    os.mkdir(OUTPUTS_DIR)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# Load Q Aware model
with tfmot.quantization.keras.quantize_scope():
    q_aware_model : Functional = tf.keras.models.load_model(LOAD_PATH_Q_AWARE)

q_model_info = Quantization.QuantizedModelInfo(q_aware_model)

print(q_model_info.bias_scales[q_model_info.keys[1]])
print(q_model_info.output_scales[q_model_info.keys[1]])

print(np.round(np.round(q_model_info.output_max[q_model_info.keys[1]]/q_model_info.output_scales[q_model_info.keys[1]])*q_model_info.output_scales[q_model_info.keys[1]]/q_model_info.bias_scales[q_model_info.keys[1]]))
print(np.round(np.round(q_model_info.output_min[q_model_info.keys[1]]/q_model_info.output_scales[q_model_info.keys[1]])*q_model_info.output_scales[q_model_info.keys[1]]/q_model_info.bias_scales[q_model_info.keys[1]]))





