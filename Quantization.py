import tensorflow as tf
import tensorflow_model_optimization as tfmot
from keras.engine.functional import Functional
import numpy as np
import numpy.typing as npt
from typing import Tuple, List
from collections import OrderedDict
import os
import gc
from enum import Enum

class ModelEvaluationMode(Enum):
    _2_quantized : int = 0
    no_input_saturation : int = 1
    manual_saturation : int = 2

class QuantizedModelInfo():
    """ Quantized Model Info Class
    -
    - Receives a quantized_model as an input parameter
    - Contains all relevant information for calculations related to the quantized model 
    """
    TVAR_KERNEL : int = 0
    TVAR_BIAS : int = 1
    def __init__(self, quantized_model : Functional | tf.keras.Sequential, bit_width : int = 8) -> None:
        self.quantized_model = quantized_model
        self.generate_indexes_keys()
        self.generate_model_scales(bit_width = bit_width)
        self.generate_quantized_weights_and_biases(bit_width = bit_width)
        self.generate_layers_shapes()

    def generate_indexes_keys(self) -> None:
        """ Generate the model indexes and lists
        - First created value is a list of every index of each layer that contains important parameters
        - Second created value is a list of all the keys for every layer that will be used later
        """
        self.layers_indexes : List[int] = []
        self.keys : List[str] = []
        for i, layer in enumerate(self.quantized_model.layers):
            layer_flag = False
            for nt_variable in layer.non_trainable_variables:
                if ("quantize_layer" in nt_variable.name or "kernel" in nt_variable.name) and not layer_flag:
                    self.layers_indexes.append(i)
                    self.keys.append(layer.name)
                    layer_flag = True

    def generate_model_scales(self, bit_width : int = 8) -> None:
        """ Generate scales for all the layers of the quantized model
        - kernel_max_vars : max values for each convolutional layer
        - kernel_min_vars : min values for each convolutional layer
        - input_scales : input quantizing scale to each layer
        - kernel_scales : scales for the kernel of each convolutional and dense layer
        - bias_scales : scales for all the biases in each convolutional and dense layer
        - output_scales : scales for the output of each layer
        """
        self.input_scales : OrderedDict[str, float] = OrderedDict()
        self.input_max : OrderedDict[str, float] = OrderedDict()
        self.input_min : OrderedDict[str, float] = OrderedDict()
        self.output_scales : OrderedDict[str, float] = OrderedDict()
        self.output_max : OrderedDict[str, float] = OrderedDict()
        self.output_min : OrderedDict[str, float] = OrderedDict()
        self.quantized_output_max : OrderedDict[str, float] = OrderedDict()
        self.quantized_output_min : OrderedDict[str, float] = OrderedDict()
        self.input_zeros : OrderedDict[str, int] = OrderedDict()
        self.output_zeros : OrderedDict[str, int] = OrderedDict()
        self.kernel_max_vars : OrderedDict[str, npt.NDArray[np.float32]] = OrderedDict()
        self.kernel_min_vars : OrderedDict[str, npt.NDArray[np.float32]] = OrderedDict()
        self.kernel_scales : OrderedDict[str, npt.NDArray[np.float32]] = OrderedDict()
        self.bias_scales : OrderedDict[str, npt.NDArray[np.float32]] = OrderedDict()

        for i, index in enumerate(self.layers_indexes):
            key = self.keys[i]
            for nt_variable in self.quantized_model.layers[index].non_trainable_variables:
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
                self.input_scales[key] = self.output_scales[self.keys[i - 1]]
                self.input_max[key] = self.output_max[self.keys[i - 1]]
                self.input_min[key] = self.output_min[self.keys[i - 1]]
                self.input_zeros[key] = self.output_zeros[self.keys[i - 1]]
                self.kernel_max_vars[key] = kernel_max_var.numpy()
                self.kernel_min_vars[key] = kernel_min_var.numpy()
                self.kernel_scales[key] = (kernel_max_var - kernel_min_var).numpy()/(2**bit_width - 2)
                self.bias_scales[key] = self.input_scales[key]*self.kernel_scales[key]
            self.output_scales[key] = ((max_var - min_var).numpy()/(2**bit_width - 1)).astype(np.float32)
            self.output_max[key] = max_var.numpy()
            self.output_min[key] = min_var.numpy()
            self.quantized_output_max[key] = self.output_scales[key]*np.round(max_var.numpy()/self.output_scales[key])
            self.quantized_output_min[key] = self.output_scales[key]*np.round(min_var.numpy()/self.output_scales[key])
            self.output_zeros[key] = -(2**(bit_width - 1) + np.round(min_var.numpy()/self.output_scales[key]).astype(int))

    def generate_quantized_weights_and_biases(self, bit_width : int = 8) -> None:
        """ Generates the quantized int values of the kernels weights and biases
        - quantized_weights : The quantized weights of every kernel of all the convolutional and dense layers
        - quantized_bias : The quantized biases of every convolutional and dense layers
        """
        self.quantized_bias : OrderedDict[str, npt.NDArray[np.int32]] = OrderedDict()
        self.quantized_and_dequantized_weights : OrderedDict[str, npt.NDArray[np.int32]] = OrderedDict()
        self.quantized_weights : OrderedDict[str, npt.NDArray[np.int32]] = OrderedDict()

        for i, index in enumerate(self.layers_indexes):
            key = self.keys[i]
            if len(self.quantized_model.layers[index].trainable_variables) != 0:
                self.quantized_bias[key] = np.round(self.quantized_model.layers[index].trainable_variables[self.TVAR_BIAS].numpy()/self.bias_scales[key]).astype(int)
                if "conv2d" in key:
                    self.quantized_and_dequantized_weights[key] = tf.quantization.fake_quant_with_min_max_vars_per_channel(self.quantized_model.layers[index].trainable_variables[self.TVAR_KERNEL], self.kernel_min_vars[key], self.kernel_max_vars[key], bit_width, narrow_range = True).numpy()
                elif "dense" in key:
                    self.quantized_and_dequantized_weights[key] = tf.quantization.fake_quant_with_min_max_vars(self.quantized_model.layers[index].trainable_variables[self.TVAR_KERNEL], self.kernel_min_vars[key], self.kernel_max_vars[key], bit_width, narrow_range = True).numpy()
                self.quantized_weights[key] = np.round(self.quantized_and_dequantized_weights[key] / self.kernel_scales[key]).astype(int)

    def generate_layers_shapes(self) -> None:
        """ Generates the layer shapes
        - input_shapes
        - kernel_shapes
        - bias_shapes
        - output_shapes
        """
        self.input_shapes : OrderedDict[str, Tuple[None | int]] = OrderedDict()
        self.kernel_shapes : OrderedDict[str, Tuple[None | int]] = OrderedDict()
        self.bias_shapes : OrderedDict[str, Tuple[None | int]] = OrderedDict()
        self.output_shapes : OrderedDict[str, Tuple[None | int]] = OrderedDict()
        for i, layer in enumerate(self.quantized_model.layers):
            layer_flag = False
            for nt_variable in layer.non_trainable_variables:
                if "kernel" in nt_variable.name and not layer_flag:
                    self.input_shapes[layer.name] = layer.input_shape
                    self.kernel_shapes[layer.name] = layer.trainable_variables[self.TVAR_KERNEL].numpy().shape
                    self.bias_shapes[layer.name] = layer.trainable_variables[self.TVAR_BIAS].numpy().shape
                    self.output_shapes[layer.name] = layer.output_shape
                    layer_flag = True

# Deprecated
def evaluate_separation_error(q_aware_model : tf.keras.Model, test_images: np.ndarray, 
out_q_aware: np.ndarray, dequantized_out_part1: np.ndarray, out_part2: np.ndarray) -> None:
    """ Test the errors generated by separating the model into 2 parts
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

# Log and other relevant functions
def garbage_collection():
    """ Manual garbage collection
    - Absolute necessary to free RAM during execution
    """
    # Garbage collection
    tf.keras.backend.clear_session()
    gc.collect()
    if tf.config.list_physical_devices('GPU'):
        tf.config.experimental.reset_memory_stats('GPU:0')

def evaluate_model(interpreter: tf.lite.Interpreter, test_images : np.ndarray, test_labels : np.ndarray) -> Tuple[float, float]:
    """ Evaluate TFLite Model:
    - Receives the interpreter and returns a tuple of loss and accuracy.
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

def mix_split_models_generator(q_aware_model : Functional | tf.keras.Model, q_model_info : QuantizedModelInfo) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """ Mixed model generation
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
    key = q_model_info.keys[INDEX_FIRST_LAYER_KEY_LIST]
    nq_model_part1.layers[INDEX_FIRST_CONV_PT1_MODEL].set_weights([q_model_info.quantized_weights[key]])
    
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

def recover_file_index(file_exists_flag: bool, save_data_path : str, file_delimiter: str = ",") -> int:
    """ Verifies if file already exists
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

def model_partial_predict(q_aware_model: Functional | tf.keras.Model, data_input: npt.NDArray[np.float32], layer_index_start: int) -> npt.NDArray[np.float32]:
    """ Predicts one output from a keras model assuming the prediction starts from a given layer
    - It is assumed that the shapes coincide
    - Need to add assertion of shape compatibility in the future
    """
    LIMIT_INDEX = len(q_aware_model.layers)
    for i in range(layer_index_start, LIMIT_INDEX):
        if i == layer_index_start:
            partial_output : npt.NDArray[np.float32] = q_aware_model.layers[i](data_input)
        else:
            partial_output = q_aware_model.layers[i](partial_output)

    return partial_output.numpy()

def model_partial_evaluate(q_aware_model: Functional | tf.keras.Model, layer_index_start: int, data_input : np.ndarray, test_labels : npt.NDArray[np.uint8]) -> Tuple[float, float]:
    """ Evaluate Keras Model from partial input manually:
    - Receives a keras model and returns a tuple of loss and accuracy.
    """
    # Run predictions on every input element of the set
    prediction_digits = []
    predictions = []
    
    output = model_partial_predict(q_aware_model, data_input, layer_index_start)

    # entropy = []
    for i, out in enumerate(output):
        digit = np.argmax(out)
        predictions.append(out)
        prediction_digits.append(digit)
        # entropy.append(-np.log(out[test_labels[i]]/np.sum(out)))

    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)
    predictions = np.array(predictions)
    scce = tf.keras.losses.SparseCategoricalCrossentropy()(test_labels, predictions)
    # entropy = np.array(entropy)
    # print(np.average(entropy))

    loss = scce.numpy()
    accuracy = (prediction_digits == test_labels).mean()
    return loss, accuracy

def prediction_by_batches(data_input : npt.NDArray[np.int32], 
test_labels : npt.NDArray[np.uint8],
n_partitions : int, 
q_model_info : QuantizedModelInfo,
scales_key : str, 
model_1 : tf.keras.Model | None = None,
model_2 : tf.keras.Model | None = None,
evaluation_mode : ModelEvaluationMode = ModelEvaluationMode._2_quantized,
start_index : int = 0) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """ Evaluates the model and deletes memory unused
    """
    # Batch analysis for testing and avoid memory head overload
    # model.predict(), tf.nn.bias_add(), tf.nn.relu(), and all eager tensor operations may exceed memory allocation
    BATCH_SIZE = test_labels.shape[0]//n_partitions

    # Calculation of the dequantized output of part 1
    # Must be done in batches
    quant_conv1 : List[npt.NDArray[np.float32]] | npt.NDArray[np.int32] = []
    dequant_activ1 : List[npt.NDArray[np.float32]] | npt.NDArray[np.float32] = []
    batch_quant_conv1 : npt.NDArray[np.float32] | npt.NDArray[np.int32]
    for i in range(n_partitions):
        # output type is float32, there is no conflict as the results are convolution sumations and multiplication of 8 bit numbers
        # No rounding problem as the maximum int value is as big as 18 bits
        # Bigger values than 24 bits will produce rounding error when using tf.float32 number values
        if model_1 is not None:
            batch_quant_conv1 = model_1.predict(data_input[i*BATCH_SIZE: (i + 1)*BATCH_SIZE]) # np.float32
            quant_conv1.append(batch_quant_conv1)
        else:
            batch_quant_conv1 = data_input[i*BATCH_SIZE:(i + 1)*BATCH_SIZE]
        # Multiplication of an eager tensor by a numpy array will yield the same output dtype as of the eager tensor regardless of the numpy array dtype
        # On the other hand multiplying 2 np.arrays() will preserve the highest memory requirement
        # Use numpy() because the values the multiplication of np.array() with eager tensor will preserve tensor dtype.
        pre_dequant_activ1 : npt.NDArray[np.float32] = (q_model_info.bias_scales[scales_key] * tf.nn.relu(tf.nn.bias_add(batch_quant_conv1, q_model_info.quantized_bias[scales_key])).numpy()).astype(np.float32)
        dequant_activ1.append(q_model_info.output_scales[scales_key] * np.round(pre_dequant_activ1 / q_model_info.output_scales[scales_key]))

    dequant_activ1 = np.concatenate(dequant_activ1).astype(np.float32) # 703.12515 MBi after conversion, 88 bytes befores pointer because it is a list
    if quant_conv1:
        quant_conv1 = np.concatenate(quant_conv1).astype(np.int32) # 703.12515 MBi after conversion, 88 bytes before conversion. Important it must be int32 for flipping values later and avoiding rounding error when using float32
    
    # Accuracy
    match(evaluation_mode):
        case ModelEvaluationMode._2_quantized:
            test_loss, test_accuracy = model_2.evaluate(dequant_activ1, test_labels, verbose = 0)
        case ModelEvaluationMode.no_input_saturation:
            test_loss, test_accuracy = model_partial_evaluate(model_2, layer_index_start = start_index, data_input = dequant_activ1, test_labels = test_labels)
        case ModelEvaluationMode.manual_saturation:
            idx = q_model_info.layers_indexes.index(start_index - 1)
            key = q_model_info.keys[idx]
            dequant_activ1[dequant_activ1 >= q_model_info.quantized_output_max[key]] = q_model_info.quantized_output_max[key]
            dequant_activ1[dequant_activ1 <= q_model_info.quantized_output_min[key]] = q_model_info.quantized_output_min[key]
            test_loss, test_accuracy = model_partial_evaluate(model_2, layer_index_start = start_index, data_input = dequant_activ1, test_labels = test_labels)
        case _:
            test_loss, test_accuracy = model_2.evaluate(dequant_activ1, test_labels, verbose = 0)

    # Garbage collection
    del batch_quant_conv1 # 351.5626 MBi Numpy array
    del pre_dequant_activ1 # 351.5626 MBi Numpy array
    del dequant_activ1 # 703.125 MBi Numpy array
    garbage_collection()

    return quant_conv1, test_loss, test_accuracy



