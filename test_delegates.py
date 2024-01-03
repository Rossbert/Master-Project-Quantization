import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
from enum import IntEnum
from typing import List
import numpy.typing as npt

def evaluate_ninput_model(interpreter: tf.lite.Interpreter, dataset_inputs: dict, dataset_labels: npt.NDArray) -> List[npt.NDArray]:
    """ Evaluate TFLite Model:
    - Receives the interpreter and returns addition of inputs
    """
    # print(interpreter.get_input_details())
    # print(interpreter.get_tensor_details())
    dataset_size = dataset_inputs[interpreter.get_input_details()[0]["name"]].shape[0]
    prediction_digits = []
    predictions = []
    # For the number of inputs
    for k in range(dataset_size):
        # For the length of the dataset
        for i in range(len(dataset_inputs)):
            interpreter.set_tensor(
                tensor_index = interpreter.get_input_details()[i]["index"], 
                value = dataset_inputs[interpreter.get_input_details()[i]["name"]][k, :][np.newaxis, :])
        
        idx_output = interpreter.get_output_details()[0]["index"]

        # Run inference.
        interpreter.invoke()

        # Post-processing
        # output = interpreter.tensor(idx_output)
        output = interpreter.get_tensor(idx_output)

        predictions.append(output[0])
        digit = np.argmax(output[0])
        prediction_digits.append(digit)

        print(f"{output[0]} => predicted label: {digit} - real label: {dataset_labels[k]}")

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

# Load paths
LAYER_NAME = "conv1"
DATABASE_PATH = "./model/database2.npy"
TFLITE_PATH = "./model/add2_test.tflite"
DELEGATE_PATH = "./dependencies/custom_delegates.dll"
index = 1 # 1 8 10 541 892

# Interpreter creation
interpreter = tf.lite.Interpreter(model_path = TFLITE_PATH)

# Details of interpreter
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
tensors_type = np.float32
# Database read
database = np.load(DATABASE_PATH, allow_pickle = True)
y : npt.NDArray[np.int32] = database['y']
x1 : npt.NDArray[np.float32] = database['x1']
x2 : npt.NDArray[np.float32] = database['x2']

M = 10
index_range = slice(0, M)
dataset_inputs = {}
for i in range(len(input_details)):
    print(f"Input {i} name={input_details[i]['name']} shape={input_details[i]['shape']} dtype={input_details[i]['dtype']}")
    dataset_inputs[input_details[i]['name']] = database[f'x{i + 1}'][index_range]
dataset_labels = database['y'][index_range]
print("Output Shape: ", output_details[0]['shape'])

# Allocation of memory
interpreter.allocate_tensors()
# Output calculation
print(f"Output of Interpreter with default behaviour:")
loss, accuracy = evaluate_ninput_model(interpreter, dataset_inputs, dataset_labels)
print("")
print(f"Model accuracy : {accuracy:.2%}")
print(f"Model loss: {loss:.6f}")

N_SIMULATIONS = 2

for i in range(N_SIMULATIONS):
    # Using custom_delegate to affect weights or convolutions
    # This function calls the plugin_delegate_create
    delegate = tf.lite.experimental.load_delegate(
        library = DELEGATE_PATH,
        options = {"name": "MyDelegateSETWeights",
                "layer_name": LAYER_NAME,
                "operation_mode" : int(OperationMode.weigths),
                "bit_position": 7})

    # Interpreter creation
    new_interpreter = tf.lite.Interpreter(
        model_path = TFLITE_PATH, experimental_delegates = [delegate])

    # Allocation of memory
    new_interpreter.allocate_tensors()
    # Output calculation
    print(f"Output of Interpreter with custom delegate:")
    loss, accuracy = evaluate_ninput_model(new_interpreter, dataset_inputs, dataset_labels)
    print("")
    print(f"Model with delegate accuracy : {accuracy:.2%}")
    print(f"Model with delegate loss: {loss:.6f}")

    # for i in range(len(outputs)):
    #     print(f"{outputs[i]} = {y[list(range(*index_range.indices(M)))[i]]}")
    # print("")
    # valid_tensor_indexes = [*list(range(21)), *list(range(27,30))]
    # for i in range(len(tensors)):
    #     print(f"Tensor {valid_tensor_indexes[i]}")
    #     print(tensors[i])
    #     print(f"New tensor {valid_tensor_indexes[i]}")
    #     print(new_tensors[i])
