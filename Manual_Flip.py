import numpy as np
import scipy as sp
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from collections import OrderedDict

def evaluate_model(interpreter: tf.lite.Interpreter) -> float:
    """ Evaluate TFLite Model:
    -
    Receives the interpreter and returns accuracy.
    """
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
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
        prediction_digits.append(digit)

    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)
    accuracy = (prediction_digits == test_labels).mean()
    return accuracy

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

LOAD_PATH_Q_AWARE = "./model/" + "model_qware_final_01"
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
print('Q Aware model test accuracy : ', "{:0.2%}".format(q_aware_test_acc))
interpreter.allocate_tensors()
tflite_accuracy = evaluate_model(interpreter)
print('TFLite model test accuracy:', "{:0.2%}".format(tflite_accuracy))

# Quantification of values
bit_width = 8
quantized_and_dequantized = OrderedDict()
quantized = OrderedDict()
layer_index_list = []
keys_list = []

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
            quantized_and_dequantized[key] = quantizer(inputs = weight, training = False, weights = quantizer_vars)
            quantized[key] = np.round(quantized_and_dequantized[key] / max_var * (2**(bit_width-1)-1))

q_aware_copy : tf.keras.Model
# Load Q Aware model copy
with tfmot.quantization.keras.quantize_scope():
    q_aware_copy = tf.keras.models.load_model(LOAD_PATH_Q_AWARE)

# Random position for weight change
bit_position = 6 # 7 # 7
channel = 28 # 29 #14
kernel_row = 3 # 2 # 4
kernel_column = 4 # 4 # 1
conv_idx = 0 # First conv layer
T_VARIABLES_KERNEL_INDEX = 0

key = keys_list[conv_idx]
layer_index = layer_index_list[conv_idx]
print(key)
print("Position", kernel_row, kernel_column, 0, channel)
print("Bit Flipped", bit_position)
# Print original values
print("Quantized kernel\n", quantized[key][:,:,0,channel])
print(quantized[key][kernel_row,kernel_column,0,channel])
# Careful this is a reference to the original value
full_kernel = q_aware_copy.layers[layer_index].trainable_variables[T_VARIABLES_KERNEL_INDEX]
print("Kernel\n", full_kernel[:,:,0,channel].numpy())
print("Original kernel value", full_kernel[kernel_row,kernel_column,0,channel].numpy())

# Flip values calculation
flipped_int_kernel_value = bit_flipper(int(quantized[key][kernel_row,kernel_column,0,channel]), bit_position)
print("Flipped int value", flipped_int_kernel_value)
m_vars = {variable.name: variable for i, variable in enumerate(q_aware_model.layers[layer_index].non_trainable_variables) if keys_list[conv_idx] in variable.name}
min_key = list(key for key in m_vars if "min" in key)[0]
max_key = list(key for key in m_vars if "max" in key)[0]
min_var = m_vars[min_key]
max_var = m_vars[max_key]
flipped_float_kernel_val = flipped_int_kernel_value * max_var.numpy()[channel] / (2**(bit_width - 1) - 1)
print("Max_value", max_var.numpy()[channel])
print("Flipped float value", flipped_float_kernel_val)
# New kernel creation
update_kernel = full_kernel.numpy()
update_kernel[kernel_row,kernel_column,0,channel] = flipped_float_kernel_val
q_aware_copy.layers[layer_index].trainable_variables[T_VARIABLES_KERNEL_INDEX].assign(update_kernel)
print("New tensor kernel")
print(q_aware_copy.layers[layer_index].trainable_variables[T_VARIABLES_KERNEL_INDEX][:,:,0,channel].numpy())


# Laplacian calculation
kernel = full_kernel[:,:,0,channel].numpy()         # Assigns a copy of the values
original_laplacian = sp.ndimage.laplace(kernel)
kernel[kernel_row,kernel_column] = flipped_float_kernel_val
new_laplacian = sp.ndimage.laplace(kernel)
print("Laplacian of kernel\n", original_laplacian)
print('Original laplacian value', original_laplacian[kernel_row,kernel_column])
print("Laplacian of new kernel\n", new_laplacian)
print('New laplacian value', new_laplacian[kernel_row,kernel_column])
# Important to avoid modifying the original values
int_kernel = np.copy(quantized[key][:,:,0,channel])
original_int_laplacian = sp.ndimage.laplace(int_kernel)
int_kernel[kernel_row,kernel_column] = flipped_int_kernel_value
new_int_laplacian = sp.ndimage.laplace(int_kernel)    
print("Laplacian of int kernel\n", original_int_laplacian)
print('Original laplacian int value', original_int_laplacian[kernel_row,kernel_column])
print("Laplacian of new int kernel\n", new_int_laplacian)
print('New laplacian int value', new_int_laplacian[kernel_row,kernel_column])

# Conversion of new model to TF Lite model
new_converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_copy)
new_converter.optimizations = [tf.lite.Optimize.DEFAULT]
new_tflite_model = new_converter.convert()
new_interpreter = tf.lite.Interpreter(model_content = new_tflite_model)

# Check new accuracy
q_copy_test_loss, q_copy_test_acc = q_aware_copy.evaluate(test_images, test_labels)
print('New Q Aware model test accuracy : ', "{:0.2%}".format(q_copy_test_acc))
new_interpreter.allocate_tensors()
new_tflite_accuracy = evaluate_model(new_interpreter)
print('New TFLite model test accuracy:', "{:0.2%}".format(new_tflite_accuracy))