import time
import datetime
import os
import csv
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import Quantization
from keras.engine.functional import Functional

""" Test to affect convolution operation on first layer. 

* All the 10000 outputs on the test set will be affected
* The output channels are generated randomly
* The positions are also generated randomly
* The program iterates through all the bit positions for each 32-bit convolution result

Parameters to be tuned:
- Number of simulations = repetitions.
- Limit of number of flips in total.
- Bit step that will be flipped in the 32 bit element.
- Operation mode:
    1 = no_output_saturation : the data at the end of the first model won't be saturated.
    2 = manual_saturation : the data at the end of the first model is manually saturated after the activation.
    3 = multi_relu : the data at the end of the first model is saturated by an integer-manual-multichannel-relu activation function.
"""
N_SIMULATIONS = 50                                      # Number of repetitions of everything
N_FLIPS_LIMIT = 4                                       # Maximum total number of flips per simulation
BIT_STEPS_PROB = 1                                      # Divisor of 32, from 1 to 32
OPERATION_MODE = 1                                      # Modification of operation mode

# Quantification constants
BIAS_BIT_WIDTH = 32
# Number of partitions for batch analysis
N_PARTITIONS = 2
# First index of q_aware_model
SPLIT_INDEX = 3
operation_mode = Quantization.ModelEvaluationMode(OPERATION_MODE)
FIRST_SEPARATION = Quantization.SeparationMode.first_quantized_weights
SECOND_SEPARATION = Quantization.SeparationMode.second_tflite_model

OUTPUTS_DIR = "./outputs/"

# Load path
LOAD_PATH_Q_AWARE = "./model/model_q_aware_ep5_2023-07-02_16-50-58"
# LOAD_PATH_Q_AWARE = "./model/model_q_aware_final_01"

# Save path
SAVE_FILE_NAME = f"QSplit_{LOAD_PATH_Q_AWARE[-8:]}_{operation_mode.name}_{SECOND_SEPARATION.name[7:-4]}_{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"
# SAVE_FILE_NAME = f"QSplit_16-50-58_multi_relu_2023-07-12.csv"
# SAVE_FILE_NAME = f"QSplit_final_01_multi_relu_2023-06-18.csv"
SAVE_DATA_PATH = OUTPUTS_DIR + SAVE_FILE_NAME

if not os.path.exists(OUTPUTS_DIR):
    os.mkdir(OUTPUTS_DIR)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# Load Q Aware model
with tfmot.quantization.keras.quantize_scope():
    q_aware_model : Functional = tf.keras.models.load_model(LOAD_PATH_Q_AWARE)

# Evaluate accuracy of both models in test set
q_aware_test_loss, q_aware_test_acc = q_aware_model.evaluate(test_images, test_labels)
print(f"Q Aware model test accuracy : {q_aware_test_acc:.2%}")
print(f"Q Aware model test loss: {q_aware_test_loss:.6f}")

q_model_info = Quantization.QuantizedModelInfo(q_aware_model)
idx_conv = q_model_info.layers_indexes.index(SPLIT_INDEX - 1)
key_conv = q_model_info.keys[idx_conv]

# Preparing the quantized test set
quantized_test_images = np.round(test_images[:,:,:,np.newaxis]/q_model_info.output_scales[q_model_info.keys[0]]).astype(int)

# Generating the split models
model_1, model_2 = Quantization.split_model_mixed(q_aware_model, q_model_info, start_index = SPLIT_INDEX, first_part_mode = FIRST_SEPARATION, second_part_mode = SECOND_SEPARATION)

# Generating the quantized convolution output for the test set, as the test set is unique so is the quantized output of the convolution
quantized_conv, test_loss, test_accuracy = Quantization.model_parts_predict_by_batches(
    data_input = quantized_test_images,
    test_labels = test_labels,
    n_partitions = N_PARTITIONS,
    q_model_info = q_model_info,
    model_1 = model_1,
    model_2 = model_2,
    evaluation_mode = operation_mode,
    start_index = SPLIT_INDEX
    )

print(f"Model test accuracy: {test_accuracy:.2%}")
print(f"Model test loss: {test_loss:.6f}\n")

# Deletion of unsused variable to diminish RAM usage, highest memory value so far
del train_images # 358.8868 MBi
del test_images # 59.8145 MBi
del quantized_test_images # 29.9073 MBi
del model_1 # 48 bytes
Quantization.garbage_collection()

total_time = time.time()

# File writing and simulation
FILE_EXISTS_FLAG = os.path.exists(SAVE_DATA_PATH)
# Probabilities variables
set_values_bits = np.arange(BIT_STEPS_PROB - 1, BIAS_BIT_WIDTH, BIT_STEPS_PROB)

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
        main_writer.writerow([
            None,
            "new reference",
            None,
            None,
            None,
            None,
            test_accuracy,
            None,
            test_loss,
        ])
        main_file.flush()
        file_idx = 0
    else:
        file_idx = last_index + 1

    for simulation_number in range(N_SIMULATIONS):
        simulation_time = time.time()

        for bit_position in set_values_bits.tolist():
            for n_flips in range(1, N_FLIPS_LIMIT + 1):
                iteration_time = time.time()
                # Runs simulation for all images on the set
                quantized_conv_copy = np.copy(quantized_conv)
                for element_set in range(test_labels.shape[0]):
                    for k in range(n_flips):
                        
                        # Convolutional layer random position
                        kernel_row = np.random.randint(0, q_model_info.output_shapes[key_conv][1])
                        kernel_column = np.random.randint(0, q_model_info.output_shapes[key_conv][2])
                        channel_output = np.random.randint(0, q_model_info.output_shapes[key_conv][3])
                        position = (element_set, kernel_row, kernel_column, channel_output)
                        kernel_position = (element_set, slice(None), slice(None), channel_output)
                        
                        # Flipped values calculation
                        flipped_int = Quantization.n_bit_flipper(int(quantized_conv_copy[position]), BIAS_BIT_WIDTH, bit_position)
                        
                        # New int update value
                        quantized_conv_copy[position] = flipped_int
                
                _ , new_test_loss, new_test_accuracy = Quantization.model_parts_predict_by_batches(
                    data_input = quantized_conv_copy,
                    test_labels = test_labels,
                    n_partitions = N_PARTITIONS,
                    q_model_info = q_model_info,
                    model_1 = None,
                    model_2 = model_2,
                    evaluation_mode = operation_mode,
                    start_index = SPLIT_INDEX
                    )

                print(f"Disturbed model test accuracy: {new_test_accuracy:.2%}")
                print(f"Disturbed model test loss: {new_test_loss:.6f}")

                # Deletion of unsused memory
                del quantized_conv_copy # 703.12515 MBi
                Quantization.garbage_collection()

                main_writer.writerow([
                    file_idx,
                    f"{datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}",
                    key_conv,
                    idx_conv,
                    n_flips,
                    bit_position,
                    new_test_accuracy,
                    new_test_accuracy - test_accuracy,
                    new_test_loss,
                ])
                main_file.flush()
                file_idx += 1
                print(f"Sim={simulation_number} Model={file_idx} flips={n_flips} bit-pos={bit_position} iter-time={datetime.timedelta(seconds = time.time() - iteration_time)} time-now={datetime.timedelta(seconds = time.time() - total_time)}\n")

        print(f"Simulation={simulation_number} layer={key_conv} sim-time={datetime.timedelta(seconds = time.time() - simulation_time)}\n")

print(f"Finished total-time={datetime.timedelta(seconds = time.time() - total_time)}\n")