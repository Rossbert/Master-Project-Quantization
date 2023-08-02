import time
import datetime
import os
import csv
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import Quantization

""" Test to affect convolution on first layer.
-
Aggressive:
* All outputs on the test set will be affected
* All the channels of the output set will be affected
* The positions are generated randomly
* The bit positions to flip will be generated randomly and different for each flip.

Parameters to be tuned:
- Regarding the output file name, if you don't update the name manually the previous file won't be deleted. New data will be appended to the end of the file instead.
- Number of simulations = repetitions.
- Limit of number of flips per channel.
- Bit step that will be flipped in the 32 bit element.
"""
SAVE_FILE_NAME = 'Quantization_Split_Aggressive.csv'
N_SIMULATIONS = 10                                              # Number of repetitions of everything
N_FLIPS_PER_CHANNEL_LIMIT = 4                                   # Maximum value? = 24 x 24 = 576
BIT_STEPS_PROB = 8                                              # Divisor of 32, from 1 to 32

MODELS_DIR = "./model/"
LOAD_PATH_Q_AWARE = MODELS_DIR + "model_q_aware_final_01"
LOAD_TFLITE_PATH = MODELS_DIR + 'tflite_final_01.tflite'
OUTPUTS_DIR = "./outputs/"
SAVE_DATA_PATH = OUTPUTS_DIR + SAVE_FILE_NAME

if not os.path.exists(OUTPUTS_DIR):
    os.mkdir(OUTPUTS_DIR)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# Load Q Aware model
q_aware_model : tf.keras.Model
with tfmot.quantization.keras.quantize_scope():
    q_aware_model = tf.keras.models.load_model(LOAD_PATH_Q_AWARE)
# Load TFLite model
interpreter = tf.lite.Interpreter(LOAD_TFLITE_PATH)

# Evaluate accuracy of both models in test set
q_aware_test_loss, q_aware_test_acc = q_aware_model.evaluate(test_images, test_labels)
print('Q Aware model test accuracy : ', "{:0.2%}".format(q_aware_test_acc))
print('Q Aware model test loss: ', q_aware_test_loss)
interpreter.allocate_tensors()
tflite_test_loss, tflite_test_accuracy = Quantization.evaluate_tflite(interpreter, test_images, test_labels)
print('TFLite model test accuracy:', "{:0.2%}".format(tflite_test_accuracy))
print('TFLite model test loss: ', tflite_test_loss)

# Construction of indexes and keys for dictionaries
new_layer_index_list, new_keys_list = Quantization.get_layer_index_and_keys_list(q_aware_model)

# Quantification constants
BIT_WIDTH = 8
BIAS_BIT_WIDTH = 32
T_VARIABLES_KERNEL_INDEX = 0
T_VARIABLES_BIAS_INDEX = 1

# Model scales extraction
(kernel_max_vars, kernel_min_vars, input_scales, 
    kernel_scales, bias_scales, output_scales) = Quantization.get_model_scales(q_aware_model, new_layer_index_list, new_keys_list, BIT_WIDTH)

# Quantized values for weights and biases extraction
quantized_weights, quantized_bias = Quantization.get_quantized_weights_and_biases(q_aware_model, new_layer_index_list, new_keys_list, 
    kernel_max_vars, kernel_min_vars, kernel_scales, bias_scales, 
    BIT_WIDTH, T_VARIABLES_KERNEL_INDEX, T_VARIABLES_BIAS_INDEX)

# Extraction of shapes
_ , _ , _ , output_shapes = Quantization.get_layers_shapes(q_aware_model, T_VARIABLES_KERNEL_INDEX, T_VARIABLES_BIAS_INDEX)

# Preparing the quantized test set
semi_quantized_test_images = np.round(test_images[:,:,:,np.newaxis]/output_scales[new_keys_list[0]]).astype(int)

# Generating the split models
model_pt1_nq, model_pt2_q = Quantization.split_model_mixed(q_aware_model, quantized_weights, new_keys_list)

# Generating the quantized convolution output for the test set, as the test set is unique so is the quantized output of the convolution
INDEX_KEY_CONV1 = 1
KEY_CONV1 = new_keys_list[INDEX_KEY_CONV1]
out_part1 = model_pt1_nq.predict(semi_quantized_test_images)
dequantized_out_part1 = output_scales[KEY_CONV1] * np.round(bias_scales[KEY_CONV1] * tf.nn.relu(tf.nn.bias_add(out_part1, quantized_bias[KEY_CONV1])) / output_scales[KEY_CONV1])

# New base accuracy of split model
pt2_test_loss, pt2_test_acc = model_pt2_q.evaluate(dequantized_out_part1, test_labels, verbose = 0)
print('Split new base model test accuracy : ', "{:0.2%}".format(pt2_test_acc))
print('Split new base model test loss: ', pt2_test_loss)  

# # Comparing both outputs
# out_model_1 = model_pt2_q.predict(dequantized_out_part1)
# out_model_2 = q_aware_model.predict(test_images)
# # Test differences
# evaluate_separation_error(q_aware_model, test_images, out_model_2, dequantized_out_part1, out_model_1)
# check_differences(out_model_1, out_model_2)

# Original model variables
entry_keys = [
    '',
    'name', 
    'layer_key', 
    'kernel_index', 
    'layer_index', 
    'n_bits_flipped_per_channel', 
    'bit_disrupted', 
    'q_aware_test_accuracy', 
    'q_aware_test_acc_degradation', 
    'q_aware_test_loss'
]
original_entry = [
    None,
    q_aware_model.name + "-original",
    None,
    None,
    None,
    None,
    None,
    (q_aware_test_acc, tflite_test_accuracy),
    None,
    (q_aware_test_loss, tflite_test_loss),
]
base_entry = [
    None,
    model_pt1_nq.name + "-" + model_pt2_q.name,
    None,
    None,
    None,
    None,
    None,
    pt2_test_acc,
    None,
    pt2_test_loss,
]

print("Keys of layers", new_keys_list)
total_time = time.time()

# File writing and simulation
FILE_EXISTS_FLAG = os.path.exists(SAVE_DATA_PATH)
SET_SIZE = out_part1.shape[0]
FILE_DELIMITER = ";"
# Probabilities variables
set_values_bits = np.arange(BIT_STEPS_PROB - 1, BIAS_BIT_WIDTH, BIT_STEPS_PROB)
probabilities = np.ones(len(set_values_bits))/len(set_values_bits)

last_index = Quantization.recover_file_index(FILE_EXISTS_FLAG, SAVE_DATA_PATH, FILE_DELIMITER)
with open(SAVE_DATA_PATH, 'a') as main_file:
    main_writer = csv.writer(main_file, delimiter = FILE_DELIMITER, lineterminator = '\n')
    if not FILE_EXISTS_FLAG:
        main_writer.writerow(entry_keys)
        main_writer.writerow(original_entry)
        main_writer.writerow(base_entry)
        file_idx = 0
    else:
        file_idx = last_index + 1

    # JUST FOR FIRST LAYER
    layer_index = new_layer_index_list[INDEX_KEY_CONV1]
    for simulation_number in range(N_SIMULATIONS):
        simulation_time = time.time()

        for n_flips in range(1, N_FLIPS_PER_CHANNEL_LIMIT + 1):
            iteration_time = time.time()

            # Runs simulation for all images on the set
            out_part1_copy = np.copy(out_part1)
            print("Simulation number", simulation_number, KEY_CONV1, INDEX_KEY_CONV1)
            for element_set in range(SET_SIZE):
                # List definitions
                position_list = []
                bit_position_list = []
                for channel_output in range(out_part1.shape[-1]):
                    for k in range(n_flips):
                        
                        # Convolutional layer random position
                        # element_set = np.random.randint(0, SET_SIZE)
                        kernel_row = np.random.randint(0, output_shapes[KEY_CONV1][1])
                        kernel_column = np.random.randint(0, output_shapes[KEY_CONV1][2])
                        # channel_output = np.random.randint(0, output_shapes[KEY_CONV1][3])
                        position = (element_set, kernel_row, kernel_column, channel_output)
                        kernel_position = (element_set, slice(None), slice(None), channel_output)

                        # Flipped values calculation
                        bit_position = np.random.choice(set_values_bits, p = probabilities).item()
                        flipped_int = Quantization.n_bit_flipper(int(out_part1_copy[position]), BIAS_BIT_WIDTH, bit_position)
                        
                        # New int update value
                        out_part1_copy[position] = flipped_int
                        # Appending primordial data
                        position_list.append(position)
                        bit_position_list.append(bit_position)

            # Calculation of the dequantized output of part 1
            new_dequantized_out_part1 = output_scales[KEY_CONV1] * np.round(bias_scales[KEY_CONV1] * tf.nn.relu(tf.nn.bias_add(out_part1_copy, quantized_bias[KEY_CONV1])) / output_scales[KEY_CONV1])
            # new_out_part2 = model_pt2_q.predict(new_dequantized_out_part1)

            # Check new accuracy on test set
            new_pt2_test_loss, new_pt2_test_acc = model_pt2_q.evaluate(new_dequantized_out_part1, test_labels, verbose = 0)
            print('Split disturbed model test accuracy : ', "{:0.2%}".format(new_pt2_test_acc))
            print('Split disturbed model test loss: ', new_pt2_test_loss)  

            main_writer.writerow([
                file_idx,
                str(simulation_number) + "_" + datetime.datetime.now().strftime("%Y%m%d %H:%M:%S"),
                KEY_CONV1,
                INDEX_KEY_CONV1,
                layer_index,
                n_flips,
                "Random with " + str(BIT_STEPS_PROB) + " steps",
                new_pt2_test_acc,
                new_pt2_test_acc - pt2_test_acc,
                new_pt2_test_loss,
            ])
            file_idx += 1
            print("N flips", n_flips, "time", datetime.timedelta(seconds = time.time() - iteration_time), '\n')

        print("Simulation", simulation_number, KEY_CONV1, "time", datetime.timedelta(seconds = time.time() - simulation_time), '\n')

print("Total time", datetime.timedelta(seconds = time.time() - total_time), '\n')