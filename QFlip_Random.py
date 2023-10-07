import time
import datetime
import os
import csv
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from typing import List
import Quantization
from keras.engine.functional import Functional

""" Parameters to be tuned:
- Output file name, if you don't update the name manually the previous file won't be deleted. New data will be appended to the end of the file instead.
- Flag that enables training data to be saved, a False flag will decrease running time significantly.
- Flag that enables laplacian related data to be saved.
- Number of simulations per layer.
- Total number of bits that will be flipped randomly from any weight in each layer.
"""
N_SIMULATIONS = 20
N_FLIPS_LIMIT = 4

BIT_WIDTH = 8
T_VARIABLES_KERNEL_INDEX = 0

OUTPUTS_DIR = "./outputs/"

# Load path
LOAD_PATH_Q_AWARE = "./model/model_q_aware_final_01"

# Save path
SAVE_FILE_NAME = f"QFlip_Rand_{LOAD_PATH_Q_AWARE[-8:]}_{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"
SAVE_DATA_PATH = OUTPUTS_DIR + SAVE_FILE_NAME

if not os.path.exists(OUTPUTS_DIR):
    os.mkdir(OUTPUTS_DIR)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

del train_labels
del train_images

# Load Q Aware model
with tfmot.quantization.keras.quantize_scope():
    q_aware_model : Functional = tf.keras.models.load_model(LOAD_PATH_Q_AWARE)

# Evaluate accuracy of both models in test set
q_aware_test_loss, q_aware_test_acc = q_aware_model.evaluate(test_images, test_labels)
print(f"Q Aware model test accuracy : {q_aware_test_acc:.2%}")
print(f"Q Aware model test loss: {q_aware_test_loss:.6f}")

# Quantification of values
q_model_info = Quantization.QuantizedModelInfo(q_aware_model)

total_time = time.time()
# File writing and simulation
FILE_EXISTS_FLAG = os.path.exists(SAVE_DATA_PATH)
# Bit positions to be disrupted
set_values_bits = np.arange(0, BIT_WIDTH, 1)

last_index = Quantization.recover_file_index(FILE_EXISTS_FLAG, SAVE_DATA_PATH)
with open(SAVE_DATA_PATH, 'a', newline = '') as main_file:
    main_writer = csv.writer(main_file)
    if not FILE_EXISTS_FLAG:
        main_writer.writerow([
            '',
            'reference',
            'layer_key',
            'kernel_index',
            'n_bits_flipped', 
            'bit_disrupted', 
            'q_aware_test_accuracy', 
            'q_aware_test_acc_degradation', 
            'q_aware_test_loss'
        ])
        main_writer.writerow([
            None,
            "original",
            None,
            None,
            None,
            None,
            q_aware_test_acc,
            None,
            q_aware_test_loss,
        ])
        main_file.flush()
        file_idx = 0
    else:
        file_idx = last_index + 1

    for simulation_number in range(N_SIMULATIONS):
        simulation_time = time.time()
        
        keys_list = q_model_info.keys.copy()
        keys_list.remove('quantize_layer')
        for key in keys_list:
            idx = q_model_info.keys.index(key)
            layer_idx = q_model_info.layers_indexes[idx]

            for n_flips in range(1, N_FLIPS_LIMIT + 1):
                iteration_time = time.time()
                # Load Q Aware model copy
                with tfmot.quantization.keras.quantize_scope():
                    q_aware_copy : Functional = tf.keras.models.load_model(LOAD_PATH_Q_AWARE)

                bit_position_list : List[int] = []
                for k in range(n_flips):
                    if "dense" not in key:
                        # It is a convolutional layer
                        kernel_row = np.random.randint(0, q_model_info.kernel_shapes[key][0])
                        kernel_column = np.random.randint(0, q_model_info.kernel_shapes[key][1])
                        in_channel = np.random.randint(0, q_model_info.kernel_shapes[key][2])
                        out_channel = np.random.randint(0, q_model_info.kernel_shapes[key][3])
                        position = (kernel_row, kernel_column, in_channel, out_channel)
                        kernel_scales = q_model_info.kernel_scales[key][out_channel]
                    else:
                        # It is a fully connected layer
                        kernel_row = None
                        kernel_column = None
                        in_channel = np.random.randint(0, q_model_info.kernel_shapes[key][0])
                        out_channel = np.random.randint(0, q_model_info.kernel_shapes[key][1])
                        position = (in_channel, out_channel)
                        kernel_scales = q_model_info.kernel_scales[key]

                    # Flip values calculation
                    bit_position = np.random.randint(0, 8)
                    flipped_int = Quantization.n_bit_flipper(int(q_model_info.quantized_weights[key][position]), BIT_WIDTH, bit_position)
                    update_kernel = q_aware_copy.layers[layer_idx].trainable_variables[T_VARIABLES_KERNEL_INDEX].numpy()
                    update_kernel[position] = flipped_int * kernel_scales
                    q_aware_copy.layers[layer_idx].trainable_variables[T_VARIABLES_KERNEL_INDEX].assign(update_kernel)

                    bit_position_list.append(bit_position)

                # Check new accuracy on test set
                new_test_loss, new_test_accuracy = q_aware_copy.evaluate(test_images, test_labels, verbose = 0)
                print(f"Disturbed model test accuracy: {new_test_accuracy:.2%}")
                print(f"Disturbed model test loss: {new_test_loss:.6f}")
                
                Quantization.garbage_collection()

                main_writer.writerow([
                    file_idx,
                    f"{datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}",
                    key,
                    layer_idx,
                    n_flips,
                    ''.join(str(v) + ' ' if i < len(bit_position_list) - 1 else str(v) for i, v in enumerate(bit_position_list)),
                    new_test_accuracy,
                    new_test_accuracy - q_aware_test_acc,
                    new_test_loss,
                ])
                main_file.flush()
                file_idx += 1
                print(f"Sim={simulation_number} Model={file_idx} layer={key} flips={n_flips} bit-pos={bit_position_list} iter-time={datetime.timedelta(seconds = time.time() - iteration_time)} time-now={datetime.timedelta(seconds = time.time() - total_time)}\n")

        print(f"Simulation={simulation_number} sim-time={datetime.timedelta(seconds = time.time() - simulation_time)}\n")

print(f"Finished total-time={datetime.timedelta(seconds = time.time() - total_time)}\n")