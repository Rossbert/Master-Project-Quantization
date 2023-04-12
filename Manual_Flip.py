import numpy as np
import scipy as sp
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from collections import OrderedDict
from typing import Tuple

def evaluate_model(interpreter: tf.lite.Interpreter) -> Tuple[float, float]:
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

def bit_flipper(value : int, bit_pos : int) -> int:
    """ Random bit flipper 
    -
    Obtains a value and bit position and flips it.
    - All values are in 8 bits, MSB have higher probability of getting flipped
    - It is assumed value is a signed 8 bit number """
    # Negative 2 Complement conversion
    if value < 0:
        value = (-value ^ 0xFF) + 1
    flip_mask = 1 << bit_pos
    flipped_value = value ^ flip_mask
    # Negative back conversion 2 Complement
    if flipped_value >= 128:
        flipped_value = -((flipped_value ^ 0xFF) + 1)
    return flipped_value

LOAD_PATH_Q_AWARE = "./model/" + "model_q_aware_final_01"
LOAD_TFLITE_PATH = "./model/" + 'tflite_final_01.tflite'
SAVE_NEW_TFLITE_PATH = "./model/" + 'new_tflite_flip_01.tflite'

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
print('Q Aware model test accuracy: ', "{:0.2%}".format(q_aware_test_acc))
print('Q Aware model test loss: ', q_aware_test_loss)
interpreter.allocate_tensors()
tflite_loss, tflite_accuracy = evaluate_model(interpreter)
print('TFLite model test accuracy: ', "{:0.2%}".format(tflite_accuracy))
print('TFLite model test loss: ', tflite_loss)

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

q_aware_copy : tf.keras.Model
# Load Q Aware model copy
with tfmot.quantization.keras.quantize_scope():
    q_aware_copy = tf.keras.models.load_model(LOAD_PATH_Q_AWARE)

# Random position for weight change
T_VARIABLES_KERNEL_INDEX = 0
kernel_idx = 1 # Any kernel layer in this model from 0 to 3 (3 conv layers and 1 dense)
key = keys_list[kernel_idx]
layer_index = layer_index_list[kernel_idx]
bit_position = 7 # 7 # 7
kernel_row = 1 # 2 # 4 # Only valid for conv layers
kernel_column = 1 # 4 # 1 # Only valid for conv layers
in_channel = 30 # 0 # 0
out_channel = 15 # 29 #14

m_vars = {variable.name: variable for i, variable in enumerate(q_aware_model.layers[layer_index].non_trainable_variables) if keys_list[kernel_idx] in variable.name}
min_key = list(key for key in m_vars if "min" in key)[0]
max_key = list(key for key in m_vars if "max" in key)[0]
# Position and Max declaration
if "dense" not in key:
    # It is a convolutional layer
    position = (kernel_row, kernel_column, in_channel, out_channel)
    kernel_position = (slice(None), slice(None), in_channel, out_channel)
    value_position = (kernel_row, kernel_column)
    # Convolutional layers max is divided per channels
    min_var = m_vars[min_key][out_channel]
    max_var = m_vars[max_key][out_channel]
else:
    # It is a fully connected layer
    position = (in_channel, out_channel)
    kernel_position = (slice(None), slice(None)) # This slice takes the whole densely connected kernel
    value_position = (in_channel, out_channel)
    # Fully connected layer has only 1 max value for the kernel
    min_var = m_vars[min_key]
    max_var = m_vars[max_key]

print(key)
print("Position", position)
print("Bit Flipped", bit_position)
print("Quantized kernel\n", quantized[key][kernel_position])
print(quantized[key][position])

# Flip values calculation
flipped_int_kernel_value = bit_flipper(int(quantized[key][position]), bit_position)
print("Flipped int value", flipped_int_kernel_value)
flipped_float_kernel_val = flipped_int_kernel_value * max_var.numpy() / (2**(BIT_WIDTH - 1) - 1)
print("Max_value", max_var.numpy())
print("Flipped float value", flipped_float_kernel_val)
# New kernel creation
# Careful this is a reference to the original value
full_kernel = q_aware_copy.layers[layer_index].trainable_variables[T_VARIABLES_KERNEL_INDEX].numpy()
print("Kernel\n", full_kernel[kernel_position])
print("Original kernel value", full_kernel[position])
update_kernel = np.copy(full_kernel)
update_kernel[position] = flipped_float_kernel_val
q_aware_copy.layers[layer_index].trainable_variables[T_VARIABLES_KERNEL_INDEX].assign(update_kernel)
print("New tensor kernel")
print(q_aware_copy.layers[layer_index].trainable_variables[T_VARIABLES_KERNEL_INDEX][kernel_position].numpy())
# Laplacian calculation
kernel = full_kernel[kernel_position]
original_laplacian = sp.ndimage.laplace(kernel)
new_laplacian = sp.ndimage.laplace(update_kernel[kernel_position])
# Important to avoid modifying the original values
int_kernel = np.copy(quantized[key][kernel_position])
original_int_laplacian = sp.ndimage.laplace(int_kernel)
int_kernel[value_position] = flipped_int_kernel_value
new_int_laplacian = sp.ndimage.laplace(int_kernel)

print("Laplacian of kernel\n", original_laplacian)
print('Original laplacian value', original_laplacian[value_position])
print("Laplacian of new kernel\n", new_laplacian)
print('New laplacian value', new_laplacian[value_position])
print("Laplacian of int kernel\n", original_int_laplacian)
print('Original laplacian int value', original_int_laplacian[value_position])
print("Laplacian of new int kernel\n", new_int_laplacian)
print('New laplacian int value', new_int_laplacian[value_position])

# Conversion of new model to TF Lite model
new_converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_copy)
new_converter.optimizations = [tf.lite.Optimize.DEFAULT]
new_tflite_model = new_converter.convert()
new_interpreter = tf.lite.Interpreter(model_content = new_tflite_model)

# Check new accuracy
q_copy_test_loss, q_copy_test_acc = q_aware_copy.evaluate(test_images, test_labels)
print('New Q Aware model test accuracy : ', "{:0.2%}".format(q_copy_test_acc))
print('New Q Aware model test loss : ', q_copy_test_loss)
new_interpreter.allocate_tensors()
new_tflite_loss, new_tflite_accuracy = evaluate_model(new_interpreter)
print('New TFLite model test accuracy: ', "{:0.2%}".format(new_tflite_accuracy))
print('New TFLite model test loss: ', new_tflite_loss)