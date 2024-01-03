import tensorflow as tf
import numpy as np
import time
import datetime
import os
import csv
import numpy.typing as npt
from enum import IntEnum
from typing import List

def evaluate_ninput_model(interpreter: tf.lite.Interpreter, dataset_inputs: dict, dataset_labels: npt.NDArray) -> List[npt.NDArray]:
    """ Evaluate TFLite Model:
    - Receives the interpreter and returns addition of inputs
    """
    # print(interpreter.get_input_details())
    # print(interpreter.get_tensor_details())
    dataset_size = len(dataset_inputs)
    prediction_digits = []
    predictions = []
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

        predictions.append(output[0])
        digit = np.argmax(output[0])
        prediction_digits.append(digit)

        # print(f"{output[0]} => predicted label: {digit} - real label: {dataset_labels[k]}")

    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)
    predictions = np.array(predictions)
    scce = tf.keras.losses.SparseCategoricalCrossentropy()(dataset_labels, predictions)

    loss = scce.numpy()
    accuracy = (prediction_digits == dataset_labels).mean()
    return loss, accuracy

class OperationMode(IntEnum):
    none = 0
    weigths = 1
    convolution = 2

N_SIMULATIONS = 10
N_FLIPS_LIMIT = 4
# Load paths
LAYER_NAME = "conv2d/"
TFLITE_PATH = "./model/tflite_ep5_2023-07-02_16-50-58.tflite"
DELEGATE_PATH = "./dependencies/custom_delegates.dll"
OUTPUTS_DIR = "./outputs/"
SAVE_FILE_NAME = f"delegate_{TFLITE_PATH[-15:-7]}_{'weights'}_{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"
SAVE_DATA_PATH = OUTPUTS_DIR + SAVE_FILE_NAME
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

# Allocation of memory
interpreter.allocate_tensors()
# Output calculation
print(f"Output of Interpreter with default behaviour:")
original_loss, original_accuracy = evaluate_ninput_model(interpreter, test_images, test_labels)
print("")
print(f"Model accuracy : {original_accuracy:.2%}")
print(f"Model loss: {original_loss:.6f}")

total_time = time.time()
FILE_EXISTS_FLAG = os.path.exists(SAVE_DATA_PATH)
last_index = -1
with open(SAVE_DATA_PATH, 'a', newline = '') as main_file:
    main_writer = csv.writer(main_file)
    if not FILE_EXISTS_FLAG:
        main_writer.writerow([
            '',
            'reference', 
            'layer_name', 
            'n_bits_flipped', 
            'bit_disrupted', 
            'accuracy', 
            'accuracy_degradation', 
            'loss'
        ])
        main_writer.writerow([
            None,
            "original",
            None,
            None,
            None,
            original_accuracy,
            None,
            original_loss,
        ])
        main_file.flush()
        file_idx = 0
    else:
        file_idx = last_index + 1

    for simulation_number in range(N_SIMULATIONS):
        simulation_time = time.time()

        for bit_position in range(8):
            for number_flips in range(1, N_FLIPS_LIMIT + 1):
                iteration_time = time.time()
                # Using custom_delegate to affect weights or convolutions
                # This function calls the plugin_delegate_create
                delegate = tf.lite.experimental.load_delegate(
                    library = DELEGATE_PATH,
                    options = {"layer_name": LAYER_NAME,
                            "operation_mode" : int(OperationMode.convolution),
                            "bit_position": bit_position,
                            "number_flips": number_flips
                            })

                # Interpreter creation
                new_interpreter = tf.lite.Interpreter(
                    model_path = TFLITE_PATH, experimental_delegates = [delegate])

                # Allocation of memory
                new_interpreter.allocate_tensors()
                # Output calculation
                print(f"Output of Interpreter with custom delegate:")
                loss, accuracy = evaluate_ninput_model(new_interpreter, test_images, test_labels)
                print(f"Model with delegate accuracy : {accuracy:.2%}")
                print(f"Model with delegate loss: {loss:.6f}")

                main_writer.writerow([
                        file_idx,
                        f"{datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')}",
                        LAYER_NAME,
                        number_flips,
                        bit_position,
                        accuracy,
                        accuracy - original_accuracy,
                        loss,
                ])
                main_file.flush()
                file_idx += 1
                print(f"Sim={simulation_number} Model={file_idx} flips={number_flips} bit-pos={bit_position} iter-time={datetime.timedelta(seconds = time.time() - iteration_time)} time-now={datetime.timedelta(seconds = time.time() - total_time)}\n")

        print(f"Simulation={simulation_number} layer={LAYER_NAME} sim-time={datetime.timedelta(seconds = time.time() - simulation_time)}\n")

print(f"Finished total-time={datetime.timedelta(seconds = time.time() - total_time)}\n")