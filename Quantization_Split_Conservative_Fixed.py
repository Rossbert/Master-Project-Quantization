import time
import datetime
import os
import csv
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from collections import OrderedDict
from typing import Tuple, List

def evaluate_model(interpreter: tf.lite.Interpreter, test_images : np.ndarray, test_labels : np.ndarray) -> Tuple[float, float]:
    """ Evaluate TFLite Model:
    -
    Receives the interpreter and returns a tuple of loss and accuracy.
    """
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    predictions = []
    for i, test_image in enumerate(test_images):
        # Pre-processing: add batch dimension and convert to float32 to match with the model's input data format.
        test_image = np.expand_dims(test_image, axis = 0).astype(np.float32)
        test_image = np.expand_dims(test_image, axis = 3).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest probability.
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        predictions.append(np.copy(output()[0]))
        prediction_digits.append(digit)

    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)
    predictions = np.array(predictions)
    scce = tf.keras.losses.SparseCategoricalCrossentropy()(test_labels, predictions)

    loss = scce.numpy()
    accuracy = (prediction_digits == test_labels).mean()
    return loss, accuracy

def mix_split_models_generator(q_aware_model : tf.keras.Model, quantized_weights: OrderedDict, keys_list: List):
    r""" Mixed model generation
    -
    Generates first model as non quantized and second model as quantized.
    - First non quantized model with quantized weights
    - Second quantized model with dequantized weights
    """
    INDEX_FIRST_LAYER_KEY_LIST = 1
    INDEX_FIRST_CONV_ORIGINAL_MODEL = 2
    INDEX_FIRST_CONV_PT1_MODEL = 1
    INDEX_QUANTIZE_LAYER_PT2_MODEL = 1
    PT2_LENGTH = len(q_aware_model.layers) - 3
    START_INDEX_ORIGINAL_MODEL_PT2 = 3
    START_INDEX_PT2_MODEL = 2

    input_layer = tf.keras.layers.Input(shape = (28, 28, 1))
    conv_1 = tf.keras.layers.Conv2D(32, 5, use_bias = False, activation = None)(input_layer)

    input_layer_2 = tf.keras.layers.Input(shape = (24, 24, 32))
    pool_1 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2)(input_layer_2)
    conv_2 = tf.keras.layers.Conv2D(64, 5, use_bias = True, activation = 'relu')(pool_1)
    pool_2 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2)(conv_2)
    conv_3 = tf.keras.layers.Conv2D(96, 3, use_bias = True, activation = 'relu')(pool_2)
    pool_3 = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2)(conv_3)
    flat_1 = tf.keras.layers.Flatten()(pool_3)
    dense_out = tf.keras.layers.Dense(10, activation = 'softmax', name = "dense_last")(flat_1)

    # First model: non-quantized with quantized weights
    nq_model_part1 = tf.keras.models.Model(inputs = input_layer, outputs = conv_1)
    # Second model: quantized
    nq_model_part2 = tf.keras.models.Model(inputs = input_layer_2, outputs = dense_out)
    q_model_part2 = tfmot.quantization.keras.quantize_model(nq_model_part2)

    # Assignation of values for the part 1 model
    key = keys_list[INDEX_FIRST_LAYER_KEY_LIST]
    nq_model_part1.layers[INDEX_FIRST_CONV_PT1_MODEL].set_weights([quantized_weights[key]])
    
    # Assignation of max and min values for quantization layer for part 2 model
    indexes_original_part2 = list(range(START_INDEX_ORIGINAL_MODEL_PT2, START_INDEX_ORIGINAL_MODEL_PT2 + PT2_LENGTH))
    indexes_new_part2 = list(range(START_INDEX_PT2_MODEL, START_INDEX_PT2_MODEL + PT2_LENGTH))
    quantize_layer_max_min = q_aware_model.layers[INDEX_FIRST_CONV_ORIGINAL_MODEL].get_weights()[-2:]
    quantize_layer_max_min.append(-1)
    q_model_part2.layers[INDEX_QUANTIZE_LAYER_PT2_MODEL].set_weights(quantize_layer_max_min)
    # Assignation of the rest of the values for the rest of the part 2 model
    weights_part2_model = [q_aware_model.layers[idx].get_weights() for idx in indexes_original_part2]
    for i, idx in enumerate(indexes_new_part2):
        q_model_part2.layers[idx].set_weights(weights_part2_model[i])

    q_model_part2.compile(optimizer = 'adam', 
        loss = 'sparse_categorical_crossentropy', 
        metrics = ['accuracy'])

    return nq_model_part1, q_model_part2

def n_bit_flipper(value : int, bit_width: int, bit_pos : int) -> int:
    """ Bit flipper n-bit length
    -
    Obtains a value and flips one bit at the specified position.
    - All values are in n bits
    - It is assumed value is a signed n bit number """
    # Negative 2 Complement conversion
    mask = 2**bit_width - 1
    if value < 0:
        value = (-value ^ mask) + 1
    flip_mask = 1 << bit_pos
    flipped_value = value ^ flip_mask
    # Negative back conversion 2 Complement
    if flipped_value >= 2**(bit_width - 1):
        flipped_value = -((flipped_value ^ mask) + 1)
    return flipped_value

def get_layer_index_and_keys_list(q_aware_model : tf.keras.Model) -> Tuple[List[int], List[str]]:
    """ Get the layer index and keys list
    -
    - Receives a quantized model as a parameter
    - Returns a tuple of lists
    - First created value is a list of every index of each layer that contains important parameters
    - Second created value is a list of all the keys for every layer that will be used later
    """
    new_layer_index_list = []
    new_keys_list = []
    for i, layer in enumerate(q_aware_model.layers):
        layer_flag = False
        for nt_variable in layer.non_trainable_variables:
            if ("quantize_layer" in nt_variable.name or "kernel" in nt_variable.name) and not layer_flag:
                new_layer_index_list.append(i)
                new_keys_list.append(layer.name)
                layer_flag = True
    return new_layer_index_list, new_keys_list

def get_layer_shapes(q_aware_model : tf.keras.Model, t_vars_kernel_index : int = 0, t_vars_bias_index : int = 1) -> Tuple[OrderedDict]:
    """ Get the layer shapes
    -
    - Receives a quantized model as a parameter
    - Returns a tuple of ordered dictionaries:
        * input_shapes
        * kernel_shapes
        * bias_shapes
        * output_shapes
    """
    input_shapes = OrderedDict()
    kernel_shapes = OrderedDict()
    bias_shapes = OrderedDict()
    output_shapes = OrderedDict()
    for i, layer in enumerate(q_aware_model.layers):
        layer_flag = False
        for nt_variable in layer.non_trainable_variables:
            if "kernel" in nt_variable.name and not layer_flag:
                input_shapes[layer.name] = layer.input_shape
                kernel_shapes[layer.name] = layer.trainable_variables[t_vars_kernel_index].numpy().shape
                bias_shapes[layer.name] = layer.trainable_variables[t_vars_bias_index].numpy().shape
                output_shapes[layer.name] = layer.output_shape
                layer_flag = True
    return input_shapes, kernel_shapes, bias_shapes, output_shapes

def get_model_scales(q_aware_model : tf.keras.Model, new_layer_index_list : List[int], new_keys_list : List[str], bit_width : int = 8) -> Tuple[OrderedDict]:
    """ Get scales for all the layers of the quantized model
    -
    - Inputs are the quantized model, a list of all the layers that contain scales and a list of all the keys that will be used to create the dictionaries.
    - Returns Ordered Dictionaries:
        * kernel_max_vars : max values for each convolutional layer
        * kernel_min_vars : min values for each convolutional layer
        * input_scales : input quantizing scale to each layer
        * kernel_scales : scales for the kernel of each convolutional and dense layer
        * bias_scales : scales for all the biases in each convolutional and dense layer
        * output_scales : scales for the output of each layer
    """
    input_scales = OrderedDict()
    kernel_scales = OrderedDict()
    bias_scales = OrderedDict()
    output_scales = OrderedDict()
    kernel_max_vars = OrderedDict()
    kernel_min_vars = OrderedDict()

    for i, index in enumerate(new_layer_index_list):
        key = new_keys_list[i]
        for nt_variable in q_aware_model.layers[index].non_trainable_variables:
            # First layer (Quantization Layer) does not have input scale
            if i == 0:
                if "quantize_layer" in nt_variable.name and "min" in nt_variable.name:
                    min_var = nt_variable
                if "quantize_layer" in nt_variable.name and "max" in nt_variable.name:
                    max_var = nt_variable
            else:
                if "activation_min" in nt_variable.name:
                    min_var = nt_variable
                if "activation_max" in nt_variable.name:
                    max_var = nt_variable
                if "kernel_min" in nt_variable.name:
                    kernel_min_var = nt_variable
                if "kernel_max" in nt_variable.name:
                    kernel_max_var = nt_variable
        if i != 0:
            kernel_max_vars[key] = kernel_max_var
            kernel_min_vars[key] = kernel_min_var
            input_scales[key] = output_scales[new_keys_list[i - 1]]
            kernel_scales[key] = (kernel_max_var - kernel_min_var).numpy()/(2**bit_width - 2)
            bias_scales[key] = input_scales[key]*kernel_scales[key]
        output_scales[key] = ((max_var - min_var).numpy()/(2**bit_width - 1)).astype(np.float32)

    return kernel_max_vars, kernel_min_vars, input_scales, kernel_scales, bias_scales, output_scales

def get_quantized_weights_and_biases(q_aware_model : tf.keras.Model, new_layer_index_list : List[int], new_keys_list : List[str], 
kernel_max_vars : OrderedDict, kernel_min_vars : OrderedDict, kernel_scales : OrderedDict, bias_scales : OrderedDict, 
bit_width : int = 8, t_vars_kernel_index : int = 0, t_vars_bias_index : int = 1) -> Tuple[OrderedDict]:
    """ Get the quantized int values of the kernels weights and biases
    -
    - Receives the quantized model and lists of indexes of important layers and the keys for the dictionaries
    - Receives the kernel max and min values, as well as the kernel scales and the biases scales
    - Returns 2 values:
        * quantized_weights : The quantized weights of every kernel of all the convolutional and dense layers
        * quantized_bias : The quantized biases of every convolutional and dense layers
    """
    quantized_bias = OrderedDict()
    quantized_and_dequantized_weights = OrderedDict()
    quantized_weights = OrderedDict()

    for i, index in enumerate(new_layer_index_list):
        key = new_keys_list[i]
        if len(q_aware_model.layers[index].trainable_variables) != 0:
            quantized_bias[key] = np.round(q_aware_model.layers[index].trainable_variables[t_vars_bias_index].numpy()/bias_scales[key]).astype(int)
            if "conv2d" in key:
                quantized_and_dequantized_weights[key] = tf.quantization.fake_quant_with_min_max_vars_per_channel(q_aware_model.layers[index].trainable_variables[t_vars_kernel_index], kernel_min_vars[key], kernel_max_vars[key], bit_width, narrow_range = True)
            elif "dense" in key:
                quantized_and_dequantized_weights[key] = tf.quantization.fake_quant_with_min_max_vars(q_aware_model.layers[index].trainable_variables[t_vars_kernel_index], kernel_min_vars[key], kernel_max_vars[key], bit_width, narrow_range = True)
            quantized_weights[key] = np.round(quantized_and_dequantized_weights[key] / kernel_scales[key]).astype(int)

    return quantized_weights, quantized_bias

def evaluate_separation_error(q_aware_model : tf.keras.Model, test_images: np.ndarray, 
out_q_aware: np.ndarray, dequantized_out_part1: np.ndarray, out_part2: np.ndarray) -> None:
    """ Test the errors generated by separating the model into 2 parts
    -
    """
    # Input layer
    input_layer_output = q_aware_model.layers[0](test_images[:,:,:,np.newaxis])
    # Quantize layer
    quantize_layer_output = q_aware_model.layers[1](input_layer_output)
    # Convolutional layer
    convolutional_layer_output = q_aware_model.layers[2](quantize_layer_output)

    diff_first_layer_output, first_layer_output_counts = np.unique(dequantized_out_part1 - convolutional_layer_output, return_counts = True)
    diff_real_output, real_output_counts = np.unique(out_q_aware - out_part2, return_counts = True)

    print("Difference first layer", diff_first_layer_output, first_layer_output_counts)
    print("Difference final real models", diff_real_output, real_output_counts)
    max_diff = np.max(diff_real_output)
    min_diff = np.min(diff_real_output)
    loc_max = np.where((out_q_aware - out_part2) == max_diff)
    loc_min = np.where((out_q_aware - out_part2) == min_diff)
    print("Max diff", max_diff, "Value original model", out_q_aware[loc_max], "Value part 2 model", out_part2[loc_max])
    print("Min diff", min_diff, "Value original model", out_q_aware[loc_min], "Value part 2 model", out_part2[loc_min])
    print("Difference max output", out_q_aware[loc_max[0]] - out_part2[loc_max[0]])
    print("Difference min output", out_q_aware[loc_min[0]] - out_part2[loc_min[0]])

def check_differences(out_model_1 : np.ndarray, out_model_2 : np.ndarray) -> None:
    """ Check differences between the output of 2 models predictions
    -
    - Receives 2 model outputs
    - Prints differences in console
    """
    
    prediction_digits_m1 = []
    prediction_digits_m2 = []
    predictions_m1 = []
    predictions_m2 = []
    for i in range(len(out_model_1)):
        digit = np.argmax(out_model_1[i])
        predictions_m1.append(out_model_1[i])
        prediction_digits_m1.append(digit)
        digit2 = np.argmax(out_model_2[i])
        predictions_m2.append(out_model_2[i])
        prediction_digits_m2.append(digit2)

    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits_m1 = np.array(prediction_digits_m1)
    predictions_m1 = np.array(predictions_m1)
    prediction_digits_m2 = np.array(prediction_digits_m2)
    predictions_m2 = np.array(predictions_m2)

    diff = prediction_digits_m1 == prediction_digits_m2
    loc_false = np.where(diff == False)
    print(loc_false)

def recover_file_last_index(file_exists_flag: bool, save_data_path : str, file_delimiter: str = ",") -> int:
    """ Verifies if file already exists
    -
    - Returns the last index of the file as a string
    """
    # Recover last number
    if file_exists_flag:
        with open(save_data_path, 'r') as file:
            if os.stat(save_data_path).st_size > 0:
                file.seek(0, os.SEEK_END)
                # Number of positions to jump out of the \r\n characters = 3
                file.seek(file.tell() - 3, os.SEEK_SET)
                pos = file.tell()
                # file.read() consumes the ammount of bytes read in the internal pointer
                # file.tell() will give the position of the file internal pointer
                # It has to be read byte by byte
                while file.read(1) != '\n':
                    pos -= 1
                    file.seek(pos, os.SEEK_SET)
                last_index = ''
                while file.read(1) != file_delimiter:
                    pos += 1
                    file.seek(pos, os.SEEK_SET)
                    last_index += file.read(1)
                if last_index == '':
                    last_index = '0'
                print("Last", last_index)
            else:
                file.seek(0, os.SEEK_SET)
                last_index = '-1'
            file.close()
    else:
        last_index = 0
    return int(last_index)

""" Test to affect convolution on first layer. 
-
Conservative: 
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
SAVE_FILE_NAME = 'Quantization_Split_Conservative.csv'
N_SIMULATIONS = 10                                      # Number of repetitions of everything
N_FLIPS_LIMIT = 20                                      # Maximum value? = 24 x 24 = 576
BIT_STEPS_PROB = 8                                      # Divisor of 32, from 1 to 32

MODELS_DIR = "./model/"
LOAD_PATH_Q_AWARE = MODELS_DIR + "model_q_aware_final_01"
LOAD_TFLITE_PATH = MODELS_DIR + 'tflite_final_01.tflite'
OUTPUTS_DIR = "./outputs/"
SAVE_DATA_PATH = OUTPUTS_DIR + SAVE_FILE_NAME
# SAVE_POSITIONS_FILE_NAME = 'PositionsLog_' + datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S") + '.csv'
# SAVE_POSITIONS_DATA_PATH = OUTPUTS_DIR + SAVE_POSITIONS_FILE_NAME

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
tflite_test_loss, tflite_test_accuracy = evaluate_model(interpreter, test_images, test_labels)
print('TFLite model test accuracy:', "{:0.2%}".format(tflite_test_accuracy))
print('TFLite model test loss: ', tflite_test_loss)

# Construction of indexes and keys for dictionaries
new_layer_index_list, new_keys_list = get_layer_index_and_keys_list(q_aware_model)

# Quantification constants
BIT_WIDTH = 8
BIAS_BIT_WIDTH = 32
T_VARIABLES_KERNEL_INDEX = 0
T_VARIABLES_BIAS_INDEX = 1

# Model scales extraction
(kernel_max_vars, kernel_min_vars, input_scales, 
    kernel_scales, bias_scales, output_scales) = get_model_scales(q_aware_model, new_layer_index_list, new_keys_list, BIT_WIDTH)

# Quantized values for weights and biases extraction
quantized_weights, quantized_bias = get_quantized_weights_and_biases(q_aware_model, new_layer_index_list, new_keys_list, 
    kernel_max_vars, kernel_min_vars, kernel_scales, bias_scales, 
    BIT_WIDTH, T_VARIABLES_KERNEL_INDEX, T_VARIABLES_BIAS_INDEX)

# Extraction of shapes
_ , _ , _ , output_shapes = get_layer_shapes(q_aware_model, T_VARIABLES_KERNEL_INDEX, T_VARIABLES_BIAS_INDEX)

# Preparing the quantized test set
semi_quantized_test_images = np.round(test_images[:,:,:,np.newaxis]/output_scales[new_keys_list[0]]).astype(int)

# Generating the split models
model_pt1_nq, model_pt2_q = mix_split_models_generator(q_aware_model, quantized_weights, new_keys_list)

# Generating the quantized convolution output for the test set, as the test set is unique so is the quantized output of the convolution
INDEX_KEY_CONV1 = 1
KEY_CONV1 = new_keys_list[INDEX_KEY_CONV1]


out_list = []
print(semi_quantized_test_images.shape)
for image in semi_quantized_test_images:
    val = model_pt1_nq(image[np.newaxis,:])
    out_list.append(val[0,:,:,:])
out_list = np.array(out_list)
print(out_list.shape)
out_part1 = model_pt1_nq.predict(semi_quantized_test_images, batch_size = 1000)
print(type(out_part1))
print(out_part1.shape)


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
    'name', 
    'layer_key', 
    'kernel_index', 
    'layer_index', 
    'n_bits_flipped', 
    'bit_disrupted', 
    'q_aware_test_accuracy', 
    'q_aware_test_acc_degradation', 
    'q_aware_test_loss'
]
original_entry = {
    entry_keys[0] : q_aware_model.name + "-original",
    entry_keys[1] : None,
    entry_keys[2] : None,
    entry_keys[3] : None,
    entry_keys[4] : None,
    entry_keys[5] : None,
    entry_keys[6] : (q_aware_test_acc, tflite_test_accuracy),
    entry_keys[7] : None,
    entry_keys[8] : (q_aware_test_loss, tflite_test_loss),
}
base_entry = {
    entry_keys[0] : model_pt1_nq.name + "-" + model_pt2_q.name,
    entry_keys[1] : None,
    entry_keys[2] : None,
    entry_keys[3] : None,
    entry_keys[4] : None,
    entry_keys[5] : None,
    entry_keys[6] : pt2_test_acc,
    entry_keys[7] : None,
    entry_keys[8] : pt2_test_loss,
}

print("Keys of layers", new_keys_list)
total_time = time.time()

# File writing and simulation
FILE_EXISTS_FLAG = os.path.exists(SAVE_DATA_PATH)
SET_SIZE = out_part1.shape[0]
CHANNELS_OUTPUT_SIZE = out_part1.shape[-1]
FILE_DELIMITER = ";"
# Probabilities variables
set_values_bits = np.arange(BIT_STEPS_PROB - 1, BIAS_BIT_WIDTH, BIT_STEPS_PROB)
probabilities = np.ones(len(set_values_bits))/len(set_values_bits)

last_index = recover_file_last_index(FILE_EXISTS_FLAG, SAVE_DATA_PATH, FILE_DELIMITER)
with open(SAVE_DATA_PATH, 'a') as main_file:
    main_writer = csv.writer(main_file, delimiter = FILE_DELIMITER, lineterminator = '\n')
    if not FILE_EXISTS_FLAG:
        main_writer.writerow([''] + entry_keys)
        main_writer.writerow([''] + list(original_entry.values()))
        main_writer.writerow([''] + list(base_entry.values()))
        file_idx = 0
    else:
        file_idx = last_index + 1

    # # Save output positions
    # with open(SAVE_POSITIONS_DATA_PATH, 'w') as position_file:
    #     position_writer = csv.writer(position_file, delimiter = FILE_DELIMITER, lineterminator = '\n')
    #     sub_entry_keys = ['simulation_number', 'set_element', 'n_bits_flipped', 'bit_disrupted', 'position_disrupted']
    #     position_writer.writerow(sub_entry_keys)

    # JUST FOR FIRST LAYER
    layer_index = new_layer_index_list[INDEX_KEY_CONV1]
    for simulation_number in range(N_SIMULATIONS):
        simulation_time = time.time()

        for bit_position in set_values_bits.tolist():
            for n_flips in range(1, N_FLIPS_LIMIT + 1):
                iteration_time = time.time()

                # Runs simulation for all images on the set
                out_part1_copy = np.copy(out_part1)
                print("Simulation number", simulation_number, KEY_CONV1, INDEX_KEY_CONV1)
                for element_set in range(SET_SIZE):
                    # List definitions
                    position_list = []
                    for k in range(n_flips):
                        
                        # Convolutional layer random position
                        # element_set = np.random.randint(0, SET_SIZE)
                        kernel_row = np.random.randint(0, output_shapes[KEY_CONV1][1])
                        kernel_column = np.random.randint(0, output_shapes[KEY_CONV1][2])
                        channel_output = np.random.randint(0, output_shapes[KEY_CONV1][3])
                        position = (element_set, kernel_row, kernel_column, channel_output)
                        kernel_position = (element_set, slice(None), slice(None), channel_output)

                        # Flipped values calculation
                        # bit_position = np.random.choice(set_values_bits, p = probabilities).item()
                        flipped_int = n_bit_flipper(int(out_part1_copy[position]), BIAS_BIT_WIDTH, bit_position)
                        
                        # New int update value
                        out_part1_copy[position] = flipped_int
                        # Appending primordial data
                        position_list.append(position)

                    # # Save subentries for each value flipped
                    # sub_entry = {
                    #     sub_entry_keys[0] : simulation_number,
                    #     sub_entry_keys[1] : element_set,
                    #     sub_entry_keys[2] : n_flips,
                    #     sub_entry_keys[3] : bit_position,
                    #     sub_entry_keys[4] : position_list,
                    # }
                    # position_writer.writerow(list(sub_entry.values()))
                
                # Calculation of the dequantized output of part 1
                new_dequantized_out_part1 = output_scales[KEY_CONV1] * np.round(bias_scales[KEY_CONV1] * tf.nn.relu(tf.nn.bias_add(out_part1_copy, quantized_bias[KEY_CONV1])) / output_scales[KEY_CONV1])
                # new_out_part2 = model_pt2_q.predict(new_dequantized_out_part1)

                # Check new accuracy on test set
                new_pt2_test_loss, new_pt2_test_acc = model_pt2_q.evaluate(new_dequantized_out_part1, test_labels, verbose = 0)
                print('Split disturbed model test accuracy : ', "{:0.2%}".format(new_pt2_test_acc))
                print('Split disturbed model test loss: ', new_pt2_test_loss)  

                entry = {
                    entry_keys[0] : str(simulation_number) + "_" + datetime.datetime.now().strftime("%Y%m%d %H:%M:%S"),
                    entry_keys[1] : KEY_CONV1,
                    entry_keys[2] : INDEX_KEY_CONV1,
                    entry_keys[3] : layer_index,
                    entry_keys[4] : n_flips,
                    entry_keys[5] : bit_position,
                    entry_keys[6] : new_pt2_test_acc,
                    entry_keys[7] : new_pt2_test_acc - pt2_test_acc,
                    entry_keys[8] : new_pt2_test_loss,
                }
                main_writer.writerow([file_idx] + list(entry.values()))
                file_idx += 1
                print("N flips", n_flips, "Bit position", bit_position, "time", datetime.timedelta(seconds = time.time() - iteration_time), '\n')

        print("Simulation", simulation_number, KEY_CONV1, "time", datetime.timedelta(seconds = time.time() - simulation_time), '\n')

print("Total time", datetime.timedelta(seconds = time.time() - total_time), '\n')