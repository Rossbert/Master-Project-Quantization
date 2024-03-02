import tensorflow as tf
import numpy as np
import time
import datetime
import os
import csv
import numpy.typing as npt
from enum import IntEnum
from typing import List, Tuple

def evaluate_ninput_model(interpreter: tf.lite.Interpreter, dataset_inputs: dict, dataset_labels: npt.NDArray) -> Tuple[float, float, List[npt.NDArray]]:
    """ Evaluate TFLite Model:
    - Receives the interpreter and returns addition of inputs
    """
    # print(interpreter.get_input_details())
    # print(interpreter.get_tensor_details())
    dataset_size = len(dataset_inputs)
    predicted_categories = []
    outputs = []
    # For the number of inputs
    for k in range(dataset_size):
        # For the length of the dataset
        interpreter.set_tensor(
            tensor_index = interpreter.get_input_details()[0]["index"], 
            value = dataset_inputs[k, :][np.newaxis, :])
        
        idx_output = interpreter.get_output_details()[0]["index"]

        # Run inference.
        interpreter.invoke()

        # Post-processing
        # output = interpreter.tensor(idx_output)
        output = interpreter.get_tensor(idx_output)

        outputs.append(output[0])
        category = np.argmax(output[0])
        predicted_categories.append(category)

        # print(f"{output[0]} => predicted label: {digit} - real label: {dataset_labels[k]}")

    # Compare prediction results with ground truth labels to calculate accuracy.
    predicted_categories = np.array(predicted_categories)
    outputs = np.array(outputs)
    scce = tf.keras.losses.SparseCategoricalCrossentropy()(dataset_labels, outputs)

    loss = scce.numpy()
    accuracy = (predicted_categories == dataset_labels).mean()
    return loss, accuracy, outputs

class OperationMode(IntEnum):
    """ Int enum to get the operation modes """
    none = 0
    weights = 1
    convolution = 2

def get_operation_mode(operation_mode : OperationMode) -> str:
    """ Gets the operation mode in string """
    match operation_mode:
        case OperationMode.none:
            return "none"
        case OperationMode.weights:
            return "weights"
        case OperationMode.convolution:
            return "convolution"
        case _ :
            return "error"

def get_bits_size(operation_mode: OperationMode) -> int:
    """ Gets the bit number of the operation """
    match operation_mode:
        case OperationMode.convolution:
            return 32
        case OperationMode.weights:
            return 8
        case _ :
            return -1

N_SIMULATIONS = 10
N_FLIPS_LIMIT = 4
# Load paths
LAYERS = ("conv2d/", "conv2d_1/", "conv2d_2/", "last/")
TFLITE_PATH = "./model/tflite_ep5_2023-07-02_16-50-58.tflite"
DELEGATE_PATH = "./dependencies/custom_delegates.dll"
OUTPUTS_DIR = "./outputs/"

if not os.path.exists(OUTPUTS_DIR):
    os.mkdir(OUTPUTS_DIR)

# Interpreter creation
interpreter = tf.lite.Interpreter(model_path = TFLITE_PATH)

# Details of interpreter
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Database read
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = np.expand_dims(train_images, axis = 3) / np.float32(255.0)
test_images = np.expand_dims(test_images, axis = 3) / np.float32(255.0)

# Here we limit the number of operations done for the whole dataset
section = slice(0, None)
test_images = test_images[section]
test_labels = test_labels[section]

# Allocation of memory
interpreter.allocate_tensors()
# Output calculation
print(f"Output of Interpreter with default behaviour:")
evaluation_time = time.time()
original_loss, original_accuracy, original_outputs = evaluate_ninput_model(interpreter, test_images, test_labels)
print(f"Evaluation time {time.time() - evaluation_time:.3f} seconds")
print(f"Model accuracy : {original_accuracy:.2%}")
print(f"Model loss: {original_loss:.6f}")

bit_position = 31
number_flips = 0
layer_name = LAYERS[3]

delegate = tf.lite.experimental.load_delegate(
    library = DELEGATE_PATH,
    options = {"layer_name": layer_name,
            "operation_mode" : int(OperationMode.convolution),
            "bit_position": bit_position,
            "number_flips": number_flips,
            "dataset_size": test_labels.shape[0]
            })

# Interpreter creation
new_interpreter = tf.lite.Interpreter(
    model_path = TFLITE_PATH, experimental_delegates = [delegate])

# Allocation of memory
new_interpreter.allocate_tensors()
# Output calculation
print(f"Output of Interpreter with custom delegate:")
evaluation_time = time.time()
loss, accuracy, outputs = evaluate_ninput_model(new_interpreter, test_images, test_labels)
print(f"Evaluation time {time.time() - evaluation_time:.3f} seconds")
print(f"Model with delegate accuracy : {accuracy:.2%}")
print(f"Model with delegate loss: {loss:.6f}")

save_file_name = f"delegate_{TFLITE_PATH[-15:-7]}_{get_operation_mode(OperationMode.convolution)}_{layer_name[:-1]}_{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"
save_data_path = OUTPUTS_DIR + save_file_name