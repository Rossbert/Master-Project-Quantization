import os
import csv
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import Quantization
from keras.engine.functional import Functional

SPLIT_INDEX = 3
OUTPUTS_DIR = "./outputs/"
LOAD_PATH_Q_AWARE = "./model/model_q_aware_final_01"

if not os.path.exists(OUTPUTS_DIR):
    os.mkdir(OUTPUTS_DIR)

# Load Q Aware model
with tfmot.quantization.keras.quantize_scope():
    q_aware_model : Functional = tf.keras.models.load_model(LOAD_PATH_Q_AWARE)

q_model_info = Quantization.QuantizedModelInfo(q_aware_model)
idx_conv = q_model_info.layers_indexes.index(SPLIT_INDEX - 1)
key_conv = q_model_info.keys[idx_conv]

SAVE_PATH_NPY = OUTPUTS_DIR + key_conv + "_kernel.npy"
SAVE_PATH_TXT = OUTPUTS_DIR + key_conv + "_kernel.txt"
SAVE_PATH_TXT_RAW = OUTPUTS_DIR + key_conv + "_kernel_raw.txt"

np.save(SAVE_PATH_NPY, q_model_info.quantized_weights[key_conv])

with open(SAVE_PATH_TXT, 'w', newline = '') as main_file:
    main_file.write(str(q_model_info.quantized_weights[key_conv]))
    main_file.flush()

with open(SAVE_PATH_TXT_RAW, 'w', newline = '') as main_file:
    writer = csv.writer(main_file)
    for l in range(q_model_info.quantized_weights[key_conv].shape[3]):
        for k in range(q_model_info.quantized_weights[key_conv].shape[2]):
            for i in range(q_model_info.quantized_weights[key_conv].shape[0]):
                writer.writerow(q_model_info.quantized_weights[key_conv][i,:,k,l])
                main_file.flush()

test = np.load(SAVE_PATH_NPY)
print(test)