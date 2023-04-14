import time
import datetime
import os
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from collections import OrderedDict
from typing import Tuple, List

def evaluate_model(interpreter: tf.lite.Interpreter, test_images, test_labels) -> Tuple[float, float]:
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

def prob_mass_gen(bits : int) -> Tuple[List[int], List[float]]:
    """ Discrete triangular distribution
    -
    - Receives the number of bits and calculates the probability mass function """
    values = np.arange(0, bits)
    n = len(values)
    p_max = 2/n
    m_max = p_max/(n - 1)
    # Probability calculation
    p = p_max * 1.0
    m = 2/(n - 1)*(p - 1/n)
    probabilities = [p + m*(i - values[-1]) for i in values]
    return values, probabilities

def random_bit_flipper(value : int) -> Tuple[int, int]:
    """ Random bit flipper 
    -
    Obtains a value and flips one bit at a random position according to a triangular distribution.
    - All values are in 8 bits, MSB have higher probability of getting flipped
    - It is assumed value is a signed 8 bit number """
    bits, probs = prob_mass_gen(8)
    bit_pos = np.random.choice(bits, p = probs)
    # Negative 2 Complement conversion
    if value < 0:
        value = (-value ^ 0xFF) + 1
    flip_mask = 1 << bit_pos
    flipped_value = value ^ flip_mask
    # Negative back conversion 2 Complement
    if flipped_value >= 128:
        flipped_value = -((flipped_value ^ 0xFF) + 1)
    return bit_pos, flipped_value

def random_bit_flipper_uniform(value : int) -> Tuple[int, int]:
    """ Random bit flipper with uniform distribution
    -
    Obtains a value and flips one bit at a random position according to a uniform distribution.
    - All values are in 8 bits, MSB have higher probability of getting flipped
    - It is assumed value is a signed 8 bit number """
    bit_pos = np.random.randint(8)
    # Negative 2 Complement conversion
    if value < 0:
        value = (-value ^ 0xFF) + 1
    flip_mask = 1 << bit_pos
    flipped_value = value ^ flip_mask
    # Negative back conversion 2 Complement
    if flipped_value >= 128:
        flipped_value = -((flipped_value ^ 0xFF) + 1)
    return bit_pos, flipped_value

N_SIMULATIONS_PER_LAYER = 1000
N_FLIPS = 4

OUTPUTS_DIR = "./outputs/"
MODELS_DIR = "./model/"
LOAD_PATH_MODEL =  MODELS_DIR + "model_final_01"
LOAD_PATH_Q_AWARE = MODELS_DIR + "model_q_aware_final_01"
LOAD_TFLITE_PATH = MODELS_DIR + 'tflite_final_01.tflite'

SAVE_NEW_TFLITE_PATH = MODELS_DIR + 'new_tflite_flip_01.tflite'
SAVE_DATA_PATH = OUTPUTS_DIR + 'Performance_Multiple.csv'

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

# Evaluate accuracy of both models
q_aware_test_loss, q_aware_test_acc = q_aware_model.evaluate(test_images, test_labels)
print('Q Aware model test accuracy : ', "{:0.2%}".format(q_aware_test_acc))
print('Q Aware model test loss: ', q_aware_test_loss)
q_aware_train_loss, q_aware_train_acc = q_aware_model.evaluate(train_images, train_labels)
print('Q Aware model train accuracy : ', "{:0.2%}".format(q_aware_train_acc))
print('Q Aware model train loss: ', q_aware_train_loss)
interpreter.allocate_tensors()
tflite_test_loss, tflite_test_accuracy = evaluate_model(interpreter, test_images, test_labels)
print('TFLite model test accuracy:', "{:0.2%}".format(tflite_test_accuracy))
print('TFLite model test loss: ', tflite_test_loss)
interpreter.allocate_tensors()
tflite_train_loss, tflite_train_accuracy = evaluate_model(interpreter, train_images, train_labels)
print('TFLite model train accuracy:', "{:0.2%}".format(tflite_train_accuracy))
print('TFLite model train loss: ', tflite_train_loss)

# Quantification of values
BIT_WIDTH = 8
quantized_and_dequantized = OrderedDict()
quantized = OrderedDict()
layer_index_list = []
keys_list = []
layers_shapes = []

layer : tfmot.quantization.keras.QuantizeWrapperV2
for i, layer in enumerate(q_aware_model.layers):
    quantizer : tfmot.quantization.keras.quantizers.Quantizer
    weight : tf.Variable
    if hasattr(layer, '_weight_vars'):
        for weight, quantizer, quantizer_vars in layer._weight_vars:
            min_var = quantizer_vars['min_var']
            max_var = quantizer_vars['max_var']

            key = weight.name[:-2]
            layer_index_list.append(i)
            keys_list.append(key)
            layers_shapes.append(weight.numpy().shape)
            quantized_and_dequantized[key] = quantizer(inputs = weight, training = False, weights = quantizer_vars)
            quantized[key] = np.round(quantized_and_dequantized[key] / max_var * (2**(BIT_WIDTH-1)-1))

performance_data = []
entry = {}
entry['name'] = q_aware_model.name + "-original"
entry['layer_affected'] = None
entry['kernel_index'] = None
entry['layer_affected_index'] = None
entry['position_disrupted'] = None
entry['min_var'] = None
entry['max_var'] = None
entry['original_weight_value'] = None
entry['quantized_value'] = None
entry['bit_disrupted'] = None
entry['flipped_quantized_value'] = None
entry['flipped_weight_value'] = None
entry['q_aware_test_accuracy'] = q_aware_test_acc
entry['tflite_test_accuracy'] = tflite_test_accuracy
entry['q_aware_test_acc_degradation'] = None
entry['tflite_test_acc_degradation'] = None
entry['q_aware_test_loss'] = q_aware_test_loss
entry['tflite_test_loss'] = tflite_test_loss
entry['q_aware_train_accuracy'] = q_aware_train_acc
entry['tflite_train_accuracy'] = tflite_train_accuracy
entry['q_aware_train_acc_degradation'] = None
entry['tflite_train_acc_degradation'] = None
entry['q_aware_train_loss'] = q_aware_train_loss
entry['tflite_train_loss'] = tflite_train_loss
entry['original_laplacian'] = None
entry['modified_laplacian'] = None
entry['original_int_laplacian'] = None
entry['modified_int_laplacian'] = None
entry['abs_laplacian_diff'] = None
entry['abs_int_laplacian_diff'] = None
performance_data.append(entry)

print("Keys of layers", keys_list)
print("Layer shapes", layers_shapes)
T_VARIABLES_KERNEL_INDEX = 0
total_time = time.time()
for key in keys_list:
    layer_time = time.time()
    for i in range(N_SIMULATIONS_PER_LAYER):
        iteration_time = time.time()
        q_aware_copy : tf.keras.Model
        # Load Q Aware model copy
        with tfmot.quantization.keras.quantize_scope():
            q_aware_copy = tf.keras.models.load_model(LOAD_PATH_Q_AWARE)

        kernel_idx = keys_list.index(key)
        # key = keys_list[kernel_idx]
        layer_index = layer_index_list[kernel_idx]
        print("Iteration", i, key, kernel_idx)

        # List definitions
        min_list = []
        max_list = []
        position_list = []
        kernel_position_list = []
        value_position_list = []
        bit_position_list = []
        original_weights_list = []
        quantized_weights_list = []
        flipped_quantized_list = []
        flipped_float_weight_list = []
        for j in range(N_FLIPS):
            m_vars = {variable.name: variable for i, variable in enumerate(q_aware_model.layers[layer_index].non_trainable_variables) if keys_list[kernel_idx] in variable.name}
            min_key = list(key for key in m_vars if "min" in key)[0]
            max_key = list(key for key in m_vars if "max" in key)[0]
            # Random position for weight change and max min variables identification
            if "dense" not in key:
                # It is a convolutional layer
                kernel_row = np.random.randint(0, layers_shapes[kernel_idx][0])
                kernel_column = np.random.randint(0, layers_shapes[kernel_idx][1])
                in_channel = np.random.randint(0, layers_shapes[kernel_idx][2])
                out_channel = np.random.randint(0, layers_shapes[kernel_idx][3])
                position = (kernel_row, kernel_column, in_channel, out_channel)
                kernel_position = (slice(None), slice(None), in_channel, out_channel)
                value_position = (kernel_row, kernel_column)
                # Convolutional layers max is divided per channels
                min_var = m_vars[min_key][out_channel].numpy()
                max_var = m_vars[max_key][out_channel].numpy()
            else:
                # It is a fully connected layer
                kernel_row = None
                kernel_column = None
                in_channel = np.random.randint(0, layers_shapes[kernel_idx][0])
                out_channel = np.random.randint(0, layers_shapes[kernel_idx][1])
                position = (in_channel, out_channel)
                kernel_position = (slice(None), slice(None))
                value_position = (in_channel, out_channel)
                # Fully connected layer has only 1 max value for the kernel
                min_var = m_vars[min_key].numpy()
                max_var = m_vars[max_key].numpy()

            print(key, "Random position", position)

            # Flip values calculation
            bit_position, flipped_int_kernel_value = random_bit_flipper_uniform(int(quantized[key][position]))
            flipped_float_kernel_val = flipped_int_kernel_value * max_var / (2**(BIT_WIDTH - 1) - 1)
            # New kernel creation, copy of full kernel
            full_kernel = q_aware_copy.layers[layer_index].trainable_variables[T_VARIABLES_KERNEL_INDEX].numpy()
            update_kernel = np.copy(full_kernel)
            update_kernel[position] = flipped_float_kernel_val
            q_aware_copy.layers[layer_index].trainable_variables[T_VARIABLES_KERNEL_INDEX].assign(update_kernel)
            # Laplacian calculation
            original_laplacian = sp.ndimage.laplace(full_kernel[kernel_position])
            new_laplacian = sp.ndimage.laplace(update_kernel[kernel_position])
            int_kernel = np.copy(quantized[key][kernel_position])
            original_int_laplacian = sp.ndimage.laplace(int_kernel)
            int_kernel[value_position] = flipped_int_kernel_value
            new_int_laplacian = sp.ndimage.laplace(int_kernel)

            # Appending data
            min_list.append(min_var)
            max_list.append(max_var)
            position_list.append(position)
            kernel_position_list.append(kernel_position)
            value_position_list.append(value_position)
            original_weights_list.append(full_kernel[position])
            quantized_weights_list.append(int(quantized[key][position]))
            bit_position_list.append(bit_position)
            flipped_quantized_list.append(flipped_int_kernel_value)
            flipped_float_weight_list.append(flipped_float_kernel_val)

        # Conversion of new model to TF Lite model
        new_converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_copy)
        new_converter.optimizations = [tf.lite.Optimize.DEFAULT]
        new_tflite_model = new_converter.convert()
        new_interpreter = tf.lite.Interpreter(model_content = new_tflite_model)

        # Check new accuracy
        q_copy_test_loss, q_copy_test_acc = q_aware_copy.evaluate(test_images, test_labels, verbose = 0)
        print('New Q Aware model test accuracy : ', "{:0.2%}".format(q_copy_test_acc))
        print('New Q Aware model test loss: ', q_copy_test_loss)
        q_copy_train_loss, q_copy_train_acc = q_aware_copy.evaluate(train_images, train_labels, verbose = 0)
        print('New Q Aware model train accuracy : ', "{:0.2%}".format(q_copy_train_acc))
        print('New Q Aware model train loss: ', q_copy_train_loss)
        new_interpreter.allocate_tensors()
        new_tflite_test_loss, new_tflite_test_accuracy = evaluate_model(new_interpreter, test_images, test_labels)
        print('New TFLite model test accuracy:', "{:0.2%}".format(new_tflite_test_accuracy))
        print('New TFLite model test loss: ', new_tflite_test_loss)
        new_interpreter.allocate_tensors()
        new_tflite_train_loss, new_tflite_train_accuracy = evaluate_model(new_interpreter, train_images, train_labels)
        print('New TFLite model train accuracy:', "{:0.2%}".format(new_tflite_train_accuracy))
        print('New TFLite model train loss: ', new_tflite_train_loss)
        
        entry = {}
        entry['name'] = q_aware_copy.name + "_" + str(kernel_idx) + "_" + str(i)
        entry['layer_affected'] = key
        entry['kernel_index'] = kernel_idx
        entry['layer_affected_index'] = layer_index
        entry['position_disrupted'] = position_list
        entry['min_var'] = min_list
        entry['max_var'] = max_list
        entry['original_weight_value'] = original_weights_list
        entry['quantized_value'] = quantized_weights_list
        entry['bit_disrupted'] = bit_position_list
        entry['flipped_quantized_value'] = flipped_quantized_list
        entry['flipped_weight_value'] = flipped_float_weight_list
        entry['q_aware_test_accuracy'] = q_copy_test_acc
        entry['tflite_test_accuracy'] = new_tflite_test_accuracy
        entry['q_aware_test_acc_degradation'] = q_copy_test_acc - q_aware_test_acc
        entry['tflite_test_acc_degradation'] = new_tflite_test_accuracy - tflite_test_accuracy
        entry['q_aware_test_loss'] = q_copy_test_loss
        entry['tflite_test_loss'] = new_tflite_test_loss
        entry['q_aware_train_accuracy'] = q_copy_train_acc
        entry['tflite_train_accuracy'] = new_tflite_train_accuracy
        entry['q_aware_train_acc_degradation'] = q_copy_train_acc - q_aware_train_acc
        entry['tflite_train_acc_degradation'] = new_tflite_train_accuracy - tflite_train_accuracy
        entry['q_aware_train_loss'] = q_copy_train_loss
        entry['tflite_train_loss'] = new_tflite_train_loss
        entry['original_laplacian'] = original_laplacian[value_position]
        entry['modified_laplacian'] = new_laplacian[value_position]
        entry['original_int_laplacian'] = original_int_laplacian[value_position]
        entry['modified_int_laplacian'] = new_int_laplacian[value_position]
        entry['abs_laplacian_diff'] = np.abs(entry['original_laplacian'] - entry['modified_laplacian'])
        entry['abs_int_laplacian_diff'] = np.abs(entry['original_int_laplacian'] - entry['modified_int_laplacian'])

        performance_data.append(entry)
        print("Iteration", i, "time", datetime.timedelta(seconds = time.time() - iteration_time), '\n')

    print("Layer", key, "time", datetime.timedelta(seconds = time.time() - layer_time), '\n')

data = pd.DataFrame(performance_data)
data.to_csv(SAVE_DATA_PATH)
print("Total time", datetime.timedelta(seconds = time.time() - total_time), '\n')