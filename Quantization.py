import tensorflow as tf
import tensorflow_model_optimization as tfmot
from keras.engine.functional import Functional
import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Dict, Any
from collections import OrderedDict
import os
import gc
import copy
from enum import Enum

class ModelEvaluationMode(Enum):
    """ Model Evaluation Mode
    - The original model will be split in 2 parts:
        * The first one will have quantized inputs
        * The second part will follow a behavior according to the Enum case
    - Modify the enum value to indicate the mode of evaluation of data:
        * m2_quantized: The second part will operate with an input quantizing layer with floating point weights.
        * no_input_saturation: the second part will operate with floating point weights without an input quantizing layer.
        * manual_saturation: the second part will operate with floating point weights but their values are previously manually saturated.
        * multi_relu: applying an integer manual multichannel relu activation function.
    """
    m2_quantized : int = 0
    no_input_saturation : int = 1
    manual_saturation : int = 2
    multi_relu : int = 3

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
        self.dequantized_output_max : OrderedDict[str, float] = OrderedDict()
        self.dequantized_output_min : OrderedDict[str, float] = OrderedDict()
        self.quantized_post_activ_max : OrderedDict[str, npt.NDArray[np.int32]] = OrderedDict()
        self.quantized_post_activ_min : OrderedDict[str, npt.NDArray[np.int32]] = OrderedDict()
        self.input_zeros : OrderedDict[str, int] = OrderedDict()
        self.output_zeros : OrderedDict[str, int] = OrderedDict()
        self.kernel_max_vars : OrderedDict[str, npt.NDArray[np.float32]] = OrderedDict()
        self.kernel_min_vars : OrderedDict[str, npt.NDArray[np.float32]] = OrderedDict()
        self.kernel_scales : OrderedDict[str, npt.NDArray[np.float32]] = OrderedDict()
        self.bias_scales : OrderedDict[str, npt.NDArray[np.float64]] = OrderedDict()

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
            self.output_scales[key] = ((max_var - min_var).numpy()/(2**bit_width - 1)).astype(np.float32)
            self.dequantized_output_max[key] = self.output_scales[key]*np.round(max_var.numpy()/self.output_scales[key])
            self.dequantized_output_min[key] = self.output_scales[key]*np.round(min_var.numpy()/self.output_scales[key])
            if i != 0:
                self.input_scales[key] = self.output_scales[self.keys[i - 1]]
                self.input_max[key] = self.output_max[self.keys[i - 1]]
                self.input_min[key] = self.output_min[self.keys[i - 1]]
                self.input_zeros[key] = self.output_zeros[self.keys[i - 1]]
                self.kernel_max_vars[key] = kernel_max_var.numpy()
                self.kernel_min_vars[key] = kernel_min_var.numpy()
                self.kernel_scales[key] = (kernel_max_var - kernel_min_var).numpy()/(2**bit_width - 2)
                self.bias_scales[key] = self.input_scales[key].astype(np.float64)*self.kernel_scales[key].astype(np.float)
                self.quantized_post_activ_max[key] = np.round(self.dequantized_output_max[key]/self.bias_scales[key]).astype(int)
                self.quantized_post_activ_min[key] = np.round(self.dequantized_output_min[key]/self.bias_scales[key]).astype(int)
            self.output_max[key] = max_var.numpy()
            self.output_min[key] = min_var.numpy()
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

# Relevant functions
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

def unwrapp_layers(layers_list : List[Dict[str, int | str | Dict | List]]) -> List[Dict[str, int | str | Dict | List]]:
    """ Unwrapps a quantized model
    - Receives a list of layers
    """
    quantize_index : None | int = None
    layers_list = copy.deepcopy(layers_list)

    for i, layer_config in enumerate(layers_list):
        if 'QuantizeLayer' in layer_config['class_name']:
            quantize_index = i
        if 'QuantizeWrapper' in layer_config['class_name']:
            layer_config['class_name'] = layer_config['config']['layer']['class_name']
            layer_config['name'] = layer_config['config']['layer']['config']['name']
            layer_config['config'] = layer_config['config']['layer']['config']
            if i >= 1:
                layer_config['inbound_nodes'][0][0][0] = layers_list[i - 1]['name']
            else:
                layer_config['inbound_nodes'] = []
            if 'Conv' in layer_config['class_name'] or 'Dense' in layer_config['class_name']:
                layer_config['config']['activation'] = layer_config['config']['activation']['config']['activation']

    if quantize_index is not None and quantize_index < len(layers_list) - 1:
        layers_list[quantize_index + 1]['inbound_nodes'][0][0][0] = layers_list[quantize_index - 1]['name']
        del layers_list[quantize_index]
    
    return layers_list

def split_model(model : Functional | tf.keras.Model, start_index : int = 3) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """ Divides the model in 2 at the selected index
    """
    configuration : Dict[str, str | bool | List] = model.get_config()
    layers_part_1 : List[Dict[str, int | str | Dict | List]] = copy.deepcopy(configuration['layers'][0: start_index])
    layers_part_2 : List[Dict[str, int | str | Dict | List]] = copy.deepcopy(configuration['layers'][start_index:])

    m1_configuration = {
        'name' : 'model_part_1',
        'trainable' : True,
        'layers' : layers_part_1,
        'input_layers' : [[layers_part_1[0]['name'], 0, 0]],
        'output_layers' : [[layers_part_1[-1]['name'], 0, 0]],
    }

    with tfmot.quantization.keras.quantize_scope():
        # m1 = tf.keras.models.model_from_config(configuration)
        m1 : Functional | tf.keras.Sequential = tf.keras.models.Model.from_config(m1_configuration)
    
    # Add input layer to the second model
    input_config = copy.deepcopy(layers_part_1[0])
    input_config['config']['batch_input_shape'] = m1.output_shape 
    layers_part_2[0]['inbound_nodes'][0][0][0] = input_config['name']
    layers_part_2.insert(0, input_config)

    m2_configuration = {
        'name' : 'model_part_2',
        'trainable' : True,
        'layers' : layers_part_2,
        'input_layers' : [[layers_part_2[0]['name'], 0, 0]],
        'output_layers' : [[layers_part_2[-1]['name'], 0, 0]],
    }

    with tfmot.quantization.keras.quantize_scope():
        m2 : Functional | tf.keras.Sequential = tf.keras.models.Model.from_config(m2_configuration)

    weights = [model.layers[idx].get_weights() for idx in range(len(model.layers))]

    for idx in range(len(m1.layers)):
        m1.layers[idx].set_weights(weights[idx])

    for i, idx in enumerate(range(start_index, len(model.layers))):
        m2.layers[i + 1].set_weights(weights[idx])

    return m1, m2

def split_model_mixed(q_aware_model : Functional | tf.keras.Model, q_model_info : QuantizedModelInfo, start_index : int = 3, first_quantized : bool = False) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """ Divides the model in 2
    - First part either quantized or non-quantized
    - Second part quantized model with dequantized weights
    - Assumes selected index is always the next index of a convolutional layer
    """
    configuration : Dict[str, str | bool | List] = q_aware_model.get_config()
    layers_part_1 : List[Dict[str, int | str | Dict | List]] = copy.deepcopy(configuration['layers'][0: start_index])
    layers_part_2 : List[Dict[str, int | str | Dict | List]] = copy.deepcopy(configuration['layers'][start_index:])

    if first_quantized:
        layers_part_1 = unwrapp_layers(layers_part_1)
        layers_part_1[-1]['config']['activation'] = 'linear'
        layers_part_1[-1]['config']['use_bias'] = False
    else:
        layers_part_1[-1]['config']['layer']['config']['activation']['config']['activation'] = 'linear'
        layers_part_1[-1]['config']['layer']['config']['use_bias'] = False

    m1_configuration = {
        'name' : 'model_part_1',
        'trainable' : True,
        'layers' : layers_part_1,
        'input_layers' : [[layers_part_1[0]['name'], 0, 0]],
        'output_layers' : [[layers_part_1[-1]['name'], 0, 0]],
    }

    with tfmot.quantization.keras.quantize_scope():
        # m1 = tf.keras.models.model_from_config(configuration)
        m1 : Functional | tf.keras.Sequential = tf.keras.models.Model.from_config(m1_configuration)
    
    # Add input layer to the second model
    input_config = copy.deepcopy(layers_part_1[0])
    input_config['config']['batch_input_shape'] = m1.output_shape 
    layers_part_2[0]['inbound_nodes'][0][0][0] = input_config['name']
    layers_part_2.insert(0, input_config)

    m2_configuration = {
        'name' : 'model_part_2',
        'trainable' : True,
        'layers' : layers_part_2,
        'input_layers' : [[layers_part_2[0]['name'], 0, 0]],
        'output_layers' : [[layers_part_2[-1]['name'], 0, 0]],
    }

    with tfmot.quantization.keras.quantize_scope():
        m2 : Functional | tf.keras.Sequential = tf.keras.models.Model.from_config(m2_configuration)

    weights = [q_aware_model.layers[idx].get_weights() for idx in range(len(q_aware_model.layers))]

    new_weights = copy.deepcopy(weights[:start_index])

    if first_quantized:
        i = 0
        idx = 0
        for j in range(len(new_weights)):
            idx = q_model_info.layers_indexes[i]
            key = q_model_info.keys[i]
            if j == idx:
                if 'quantize_layer' not in key:
                    if idx < start_index - 1:
                        new_weights[j] = [q_model_info.quantized_weights[key], q_model_info.quantized_bias[key]]
                    else:
                        new_weights[j] = [q_model_info.quantized_weights[key]]
                i += 1
            else:
                new_weights[j] = []

        i = 0
        quantize_index : int | None = None
        while quantize_index is None and i < len(configuration['layers']):
            if 'QuantizeLayer' in configuration['layers'][i]['class_name']:
                quantize_index = i
            i += 1
        del new_weights[quantize_index]
    else:
        # Position occupied by the bias values
        del new_weights[-1][1]

    for idx in range(len(m1.layers)):
        m1.layers[idx].set_weights(new_weights[idx])

    for i, idx in enumerate(range(start_index, len(q_aware_model.layers))):
        m2.layers[i + 1].set_weights(weights[idx])

    m2.compile(optimizer = 'adam', 
        loss = 'sparse_categorical_crossentropy', 
        metrics = ['accuracy'])

    return m1, m2

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
                print(f"Last index read: {last_index}")
            else:
                file.seek(0, os.SEEK_SET)
                last_index = '-1'
            file.close()
    else:
        last_index = 0
    return int(last_index)

def model_end_predict(q_aware_model: Functional | tf.keras.Model, data_input: npt.NDArray[np.float32], layer_index_start: int) -> npt.NDArray[np.float32]:
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

def model_begin_predict(model: Functional | tf.keras.Model, data_input: npt.NDArray[np.float32], layer_index_stop: int) -> npt.NDArray[np.float32]:
    """ Predicts one output from a keras model assuming the prediction stops at a given layer
    - It is assumed that the shapes coincide
    - Need to add assertion of shape compatibility in the future
    """
    for i in range(layer_index_stop):
        if i == 0:
            partial_output : npt.NDArray[np.float32] = model.layers[i](data_input)
        else:
            partial_output = model.layers[i](partial_output)
    garbage_collection()
    return partial_output.numpy()

def model_partial_evaluate(q_aware_model: Functional | tf.keras.Model, layer_index_start: int, data_input : np.ndarray, test_labels : npt.NDArray[np.uint8]) -> Tuple[float, float]:
    """ Evaluate Keras Model from partial input manually:
    - Receives a keras model and returns a tuple of loss and accuracy.
    """
    # Run predictions on every input element of the set
    prediction_digits = []
    predictions = []
    
    output = model_end_predict(q_aware_model, data_input, layer_index_start)

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
model_1 : tf.keras.Model | None = None,
model_2 : tf.keras.Model | None = None,
evaluation_mode : ModelEvaluationMode = ModelEvaluationMode.manual_saturation,
start_index : int = 3) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """ Evaluates the model and deletes memory unused
    """
    # Batch analysis for testing and avoid memory head overload
    BATCH_SIZE = test_labels.shape[0]//n_partitions
    idx = q_model_info.layers_indexes.index(start_index - 1)
    key = q_model_info.keys[idx]
    # Must be done in batches
    # model.predict(), tf.nn.bias_add(), tf.nn.relu(), and all eager tensor operations may exceed memory allocation
    quant_conv : List[npt.NDArray[np.float32]] | npt.NDArray[np.int32] = []
    quant_postactiv : List[npt.NDArray[np.int32]] | npt.NDArray[np.int32] = []
    for i in range(n_partitions):
        # Output type of .predict() method is float32, there is no conflict as the results are convolution sumations and multiplication of 8 bit numbers
        # No rounding problem as the maximum int value is as big as 18 bits
        # Bigger values than 24 bits will produce rounding error when using tf.float32 number values
        if model_1 is not None:
            batch_quant_conv : npt.NDArray[np.float32] = model_1.predict(data_input[i*BATCH_SIZE: (i + 1)*BATCH_SIZE]) # np.float32
            quant_conv.append(batch_quant_conv)
        else:
            batch_quant_conv : npt.NDArray[np.int32] = data_input[i*BATCH_SIZE: (i + 1)*BATCH_SIZE]
        # Multiplication of an eager tensor by a numpy array will yield the same output dtype as of the eager tensor regardless of the numpy array dtype
        # On the other hand multiplying 2 np.arrays() will preserve the highest memory requirement
        # Use .numpy() method to convert tensor to numpy array because the multiplication preserves highest dtype.
        batch_quant_postactiv : npt.NDArray[np.int32] = tf.nn.relu(tf.nn.bias_add(batch_quant_conv, q_model_info.quantized_bias[key])).numpy().astype(int)
        if model_1 is None and evaluation_mode == ModelEvaluationMode.multi_relu:
            for channel in range(batch_quant_postactiv.shape[-1]):
                batch_quant_postactiv[:,:,:, channel][batch_quant_postactiv[:,:,:, channel] >= q_model_info.quantized_post_activ_max[key][channel]] = q_model_info.quantized_post_activ_max[key][channel]
                batch_quant_postactiv[:,:,:, channel][batch_quant_postactiv[:,:,:, channel] <= q_model_info.quantized_post_activ_min[key][channel]] = q_model_info.quantized_post_activ_min[key][channel]
        quant_postactiv.append(batch_quant_postactiv)

    # Garbage collection
    del batch_quant_conv # 351.5626 MBi Numpy array
    del batch_quant_postactiv # 351.5626 MBi Numpy array
    garbage_collection()

    if quant_conv:
        quant_conv = np.concatenate(quant_conv).astype(np.int32) # 703.12515 MBi Numpy array. Important it must be int32 for flipping values later and avoiding rounding error when using float32
    quant_postactiv = np.concatenate(quant_postactiv) # 703.125 MBi Numpy array

    dequant_postactiv : npt.NDArray[np.float32] = (q_model_info.bias_scales[key] * quant_postactiv).astype(np.float32)
    rescaled_postactiv = (q_model_info.output_scales[key] * np.round(dequant_postactiv / q_model_info.output_scales[key])).astype(np.float32)
    
    # Garbage collection
    del quant_postactiv # 703.125 MBi Numpy array
    del dequant_postactiv # 703.125 MBi Numpy array
    garbage_collection()

    # Identifying the case of the evaluation mode
    match(evaluation_mode):
        case ModelEvaluationMode.manual_saturation:
            rescaled_postactiv[rescaled_postactiv >= q_model_info.dequantized_output_max[key]] = q_model_info.dequantized_output_max[key]
            rescaled_postactiv[rescaled_postactiv <= q_model_info.dequantized_output_min[key]] = q_model_info.dequantized_output_min[key]
        case ModelEvaluationMode.m2_quantized:
            pass
        case ModelEvaluationMode.no_input_saturation:
            pass
        case ModelEvaluationMode.multi_relu:
            pass
        case _:
            pass
    # test_loss, test_accuracy = model_partial_evaluate(model_2, layer_index_start = start_index, data_input = rescaled_postactiv, test_labels = test_labels)
    test_loss, test_accuracy = model_2.evaluate(rescaled_postactiv, test_labels, verbose = 0)

    # Garbage collection
    del rescaled_postactiv # 703.125 MBi Numpy array
    garbage_collection()

    return quant_conv, test_loss, test_accuracy
