import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from typing import List
import Quantization
from keras.engine.functional import Functional
import sys

def by_batches_non_quantized(data_input : npt.NDArray[np.int32] | npt.NDArray[np.float32], 
test_labels : npt.NDArray[np.uint8],
n_partitions : int, 
q_aware_model : tf.keras.Model | None = None,
start_index : int = 3, 
model : tf.keras.Model | None = None) -> None:
    """ Evaluates the model and deletes memory unused
    """
    # Batch analysis for testing and avoid memory head overload
    # model.predict(), tf.nn.bias_add(), tf.nn.relu(), and all eager tensor operations may exceed memory allocation
    BATCH_SIZE = test_labels.shape[0]//n_partitions
    # Calculation of the dequantized output of part 1
    # Must be done in batches
    conv : List[npt.NDArray[np.float32]] | npt.NDArray[np.float32] = []
    conv_bias : List[npt.NDArray[np.int32]] | npt.NDArray[np.float32] = []
    postactiv : List[npt.NDArray[np.int32]] | npt.NDArray[np.float32] = []
    
    biases : npt.NDArray[np.float32] = q_aware_model.layers[start_index - 1].get_weights()[1]

    for i in range(n_partitions):
        batch_conv : npt.NDArray[np.float32] = model.predict(data_input[i*BATCH_SIZE: (i + 1)*BATCH_SIZE]) # np.float32
        batch_conv_bias : npt.NDArray[np.float32] = tf.nn.bias_add(batch_conv, biases).numpy()
        batch_postactiv : npt.NDArray[np.float32] = tf.nn.relu(batch_conv_bias).numpy()
        
        conv.append(batch_conv)
        conv_bias.append(batch_conv_bias)
        postactiv.append(batch_postactiv)

    # Garbage collection
    del batch_conv # 351.5626 MBi Numpy array
    del batch_conv_bias # 351.5626 MBi Numpy array
    del batch_postactiv # 351.5626 MBi Numpy array
    Quantization.garbage_collection()
    
    conv = np.concatenate(conv) # 703.12515 MBi after conversion, 88 bytes before conversion. Important it must be int32 for flipping values later and avoiding rounding error when using float32
    
    print(f"Bias shape: {biases.shape}")
    print(f"Convolution shape: {conv.shape}")
    print(f"Convolution float output max: {np.max(conv)}")
    print(f"Convolution float output min: {np.min(conv)}")
    # Garbage collection
    del conv # 703.12515 MBi
    Quantization.garbage_collection()

    conv_bias = np.concatenate(conv_bias)

    print(f"Convolution + bias float output max: {np.max(conv_bias)}")
    print(f"Convolution + bias float output min: {np.min(conv_bias)}")
    # Garbage collection
    del conv_bias # 703.12515 MBi
    Quantization.garbage_collection()

    postactiv = np.concatenate(postactiv)

    print(f"Activation float output max: {np.max(postactiv)}")
    print(f"Activation float output min: {np.min(postactiv)}")
    # Garbage collection
    del postactiv # 703.12515 MBi
    Quantization.garbage_collection()

def by_batches_quantized(data_input : npt.NDArray[np.int32], 
test_labels : npt.NDArray[np.uint8],
n_partitions : int, 
q_model_info : Quantization.QuantizedModelInfo,
start_index : int = 3, 
model : tf.keras.Model | None = None) -> None:
    """ Evaluates the model and deletes memory unused
    """
    # Batch analysis for testing and avoid memory head overload
    # model.predict(), tf.nn.bias_add(), tf.nn.relu(), and all eager tensor operations may exceed memory allocation
    BATCH_SIZE = test_labels.shape[0]//n_partitions
    i = q_model_info.layers_indexes.index(start_index - 1)
    key = q_model_info.keys[i]
    # Calculation of the dequantized output of part 1
    # Must be done in batches
    quant_conv : List[npt.NDArray[np.float32]] | npt.NDArray[np.int32] = []
    quant_conv_bias : List[npt.NDArray[np.int32]] | npt.NDArray[np.int32] = []
    quant_postactiv : List[npt.NDArray[np.int32]] | npt.NDArray[np.int32] = []
    # rescaled_dequant_postactiv1 : List[npt.NDArray[np.float32]] | npt.NDArray[np.float32] = []
    for i in range(n_partitions):
        batch_quant_conv : npt.NDArray[np.float32] = model.predict(data_input[i*BATCH_SIZE: (i + 1)*BATCH_SIZE]) # np.float32
        batch_quant_conv_bias : npt.NDArray[np.int32] = tf.nn.bias_add(batch_quant_conv, q_model_info.quantized_bias[key]).numpy().astype(int)
        batch_quant_postactiv : npt.NDArray[np.int32] = tf.nn.relu(batch_quant_conv_bias).numpy()
        
        quant_conv.append(batch_quant_conv)
        quant_conv_bias.append(batch_quant_conv_bias)
        quant_postactiv.append(batch_quant_postactiv)

    # Garbage collection
    del batch_quant_conv # 351.5626525878906 MBi Numpy array
    del batch_quant_conv_bias # 351.5626 MBi Numpy array
    del batch_quant_postactiv # 351.5626 MBi Numpy array
    Quantization.garbage_collection()
    
    quant_conv = np.concatenate(quant_conv).astype(np.int32) # 703.12515 MBi after conversion, 88 bytes before conversion. Important it must be int32 for flipping values later and avoiding rounding error when using float32
    quant_conv_bias = np.concatenate(quant_conv_bias)
    quant_postactiv = np.concatenate(quant_postactiv)

    print(f"Bias shape: {q_model_info.bias_scales[key].shape}")
    print(f"Convolution shape: {quant_conv.shape}")
    print(f"Convolution integer output max: {np.max(quant_conv)}")
    print(f"Convolution integer output min: {np.min(quant_conv)}")
    print(f"Convolution + bias integer output max: {np.max(quant_conv_bias)}")
    print(f"Convolution + bias integer output min: {np.min(quant_conv_bias)}")
    print(f"Activation integer output max: {np.max(quant_postactiv)}")
    print(f"Activation integer output min: {np.min(quant_postactiv)}")

    dequant_conv = (q_model_info.bias_scales[key] * quant_conv).astype(np.float32)
    dequant_conv_bias = (q_model_info.bias_scales[key] * quant_conv_bias).astype(np.float32)
    dequant_postactiv = (q_model_info.bias_scales[key] * quant_postactiv).astype(np.float32)

    # Garbage collection
    del quant_conv # 703.1251525878906 MBi
    del quant_conv_bias # 703.12515 MBi
    del quant_postactiv # 703.12515 MBi
    Quantization.garbage_collection()
    print(f"Dequantized convolution float output max: {np.max(dequant_conv)}")
    print(f"Dequantized convolution float output min: {np.min(dequant_conv)}")
    print(f"Dequantized convolution + bias float output max: {np.max(dequant_conv_bias)}")
    print(f"Dequantized convolution + bias float output min: {np.min(dequant_conv_bias)}")
    print(f"Dequantized post activation float output max: {np.max(dequant_postactiv)}")
    print(f"Dequantized post activation float output min: {np.min(dequant_postactiv)}")

    quant_rescaled_postactiv = np.round(dequant_postactiv / q_model_info.output_scales[key]).astype(np.int16)
    rescaled_postactiv = (q_model_info.output_scales[key] * quant_rescaled_postactiv).astype(np.float32)

    # Garbage collection
    del dequant_conv # 703.12515 MBi
    del dequant_conv_bias # 703.12515 MBi
    del dequant_postactiv # 703.12515 MBi
    Quantization.garbage_collection()
    print(f"Quantized activation rescaled integer output max: {np.max(quant_rescaled_postactiv)}")
    print(f"Quantized activation rescaled integer output min: {np.min(quant_rescaled_postactiv)}")
    print(f"Activation rescaled float output max: {np.max(rescaled_postactiv)}")
    print(f"Activation rescaled float output min: {np.min(rescaled_postactiv)}")

    # Garbage collection
    del quant_rescaled_postactiv # 351.5626525878906 MBi Numpy array
    del rescaled_postactiv # 703.12515 MBi Numpy array
    Quantization.garbage_collection()

def by_batches_alternative_comparison(quantized_data_input : npt.NDArray[np.int32], 
non_quantized_data_input : npt.NDArray[np.float32], 
test_labels : npt.NDArray[np.uint8],
n_partitions : int, 
q_model_info : Quantization.QuantizedModelInfo,
q_aware_model : tf.keras.Model | None = None,
start_index : int = 3, 
model : tf.keras.Model | None = None) -> None:
    """ Evaluates the model and deletes memory unused
    """
    # Batch analysis for testing and avoid memory head overload
    # model.predict(), tf.nn.bias_add(), tf.nn.relu(), and all eager tensor operations may exceed memory allocation
    BATCH_SIZE = test_labels.shape[0]//n_partitions
    i = q_model_info.layers_indexes.index(start_index - 1)
    key = q_model_info.keys[i]
    # Calculation of the dequantized output of part 1
    # Must be done in batches
    quant_conv : List[npt.NDArray[np.float32]] | npt.NDArray[np.int32] = []
    quant_conv_bias : List[npt.NDArray[np.int32]] | npt.NDArray[np.int32] = []
    dequant_conv_bias = []
    quant_rescaled_conv_bias = []
    quant_rescaled_postactiv : List[npt.NDArray[np.int32]] | npt.NDArray[np.int32] = []
    # rescaled_dequant_postactiv1 : List[npt.NDArray[np.float32]] | npt.NDArray[np.float32] = []
    for i in range(n_partitions):
        batch_quant_conv : npt.NDArray[np.float32] = model.predict(quantized_data_input[i*BATCH_SIZE: (i + 1)*BATCH_SIZE]) # np.float32
        batch_quant_conv_bias : npt.NDArray[np.int32] = tf.nn.bias_add(batch_quant_conv, q_model_info.quantized_bias[key]).numpy().astype(int)
        batch_dequant_conv_bias : npt.NDArray[np.float32] = (q_model_info.bias_scales[key] * batch_quant_conv_bias).astype(np.float32)
        batch_quant_rescaled_conv_bias : npt.NDArray[np.int32] = np.round(batch_dequant_conv_bias / q_model_info.output_scales[key]).astype(np.int32)
        batch_quant_rescaled_postactiv : npt.NDArray[np.int32] = tf.nn.relu(batch_quant_rescaled_conv_bias).numpy()
        
        quant_conv.append(batch_quant_conv)
        quant_conv_bias.append(batch_quant_conv_bias)
        dequant_conv_bias.append(batch_dequant_conv_bias)
        quant_rescaled_conv_bias.append(batch_quant_rescaled_conv_bias)
        quant_rescaled_postactiv.append(batch_quant_rescaled_postactiv)

    # Garbage collection
    del batch_quant_conv # 351.5626525878906 MBi Numpy array
    del batch_quant_conv_bias # 351.5626 MBi Numpy array
    del batch_dequant_conv_bias # 351.5626 MBi Numpy array
    del batch_quant_rescaled_conv_bias
    del batch_quant_rescaled_postactiv
    Quantization.garbage_collection()
    
    quant_conv = np.concatenate(quant_conv).astype(np.int32) # 703.12515 MBi after conversion, 88 bytes before conversion. Important it must be int32 for flipping values later and avoiding rounding error when using float32
    quant_conv_bias = np.concatenate(quant_conv_bias)
    dequant_conv_bias = np.concatenate(dequant_conv_bias)
    quant_rescaled_conv_bias = np.concatenate(quant_rescaled_conv_bias)
    quant_rescaled_postactiv = np.concatenate(quant_rescaled_postactiv)

    print(f"Bias shape: {q_model_info.bias_scales[key].shape}")
    print(f"Convolution shape: {quant_conv.shape}")
    print(f"Convolution integer output max: {np.max(quant_conv)}")
    print(f"Convolution integer output min: {np.min(quant_conv)}")
    
    dequant_conv = (q_model_info.bias_scales[key] * quant_conv).astype(np.float32)

    print(f"Dequantized convolution float output max: {np.max(dequant_conv)}")
    print(f"Dequantized convolution float output min: {np.min(dequant_conv)}")
    print(f"Convolution + bias integer output max: {np.max(quant_conv_bias)}")
    print(f"Convolution + bias integer output min: {np.min(quant_conv_bias)}")
    print(f"Dequantized convolution + bias float output max: {np.max(dequant_conv_bias)}")
    print(f"Dequantized convolution + bias float output min: {np.min(dequant_conv_bias)}")
    print(f"Rescaled convolution + bias integer output max: {np.max(quant_rescaled_conv_bias)}")
    print(f"Rescaled convolution + bias integer output min: {np.min(quant_rescaled_conv_bias)}")
    print(f"Quantized activation rescaled integer output max: {np.max(quant_rescaled_postactiv)}")
    print(f"Quantized activation rescaled integer output min: {np.min(quant_rescaled_postactiv)}")

    # Garbage collection
    del quant_conv # 703.1251525878906 MBi
    del quant_conv_bias # 703.12515 MBi
    del dequant_conv # 703.12515 MBi
    del dequant_conv_bias # 703.12515 MBi
    Quantization.garbage_collection()

    rescaled_conv_bias = (q_model_info.output_scales[key] * quant_rescaled_conv_bias).astype(np.float32)
    rescaled_postactiv = (q_model_info.output_scales[key] * quant_rescaled_postactiv).astype(np.float32)

    q_model_output = Quantization.model_begin_predict(q_aware_model, non_quantized_data_input, layer_index_stop = start_index)
    difference = q_model_output - rescaled_postactiv
    difference_values, counts =  np.unique(difference, return_counts = True)

    # Garbage collection
    print(f"Rescaled convolution + bias float output max: {np.max(rescaled_conv_bias)}")
    print(f"Rescaled convolution + bias float output min: {np.min(rescaled_conv_bias)}")
    print(f"Activation rescaled float output max: {np.max(rescaled_postactiv)}")
    print(f"Activation rescaled float output min: {np.min(rescaled_postactiv)}")
    print(f"Q model float output max: {np.max(q_model_output)}")
    print(f"Q model float output min: {np.min(q_model_output)}")
    print(f"Unique differences: {difference_values} and respective counts: {counts}")
    # Garbage collection
    del rescaled_conv_bias
    del quant_rescaled_conv_bias
    del quant_rescaled_postactiv # 703.12515 MBi Numpy array
    del rescaled_postactiv # 703.12515 MBi Numpy array
    del q_model_output # 703.12515 MBi Numpy array
    del difference # 703.12515 MBi Numpy array
    Quantization.garbage_collection()

def by_batches_comparison(quantized_data_input : npt.NDArray[np.int32], 
non_quantized_data_input : npt.NDArray[np.float32], 
test_labels : npt.NDArray[np.uint8],
n_partitions : int, 
q_model_info : Quantization.QuantizedModelInfo,
q_aware_model : tf.keras.Model | None = None,
start_index : int = 3, 
model : tf.keras.Model | None = None) -> None:
    """ Evaluates the model and deletes memory unused
    """
    # Batch analysis for testing and avoid memory head overload
    # model.predict(), tf.nn.bias_add(), tf.nn.relu(), and all eager tensor operations may exceed memory allocation
    BATCH_SIZE = test_labels.shape[0]//n_partitions
    i = q_model_info.layers_indexes.index(start_index - 1)
    key = q_model_info.keys[i]
    # Calculation of the dequantized output of part 1
    # Must be done in batches
    quant_postactiv : List[npt.NDArray[np.int32]] | npt.NDArray[np.int32] = []
    # rescaled_dequant_postactiv1 : List[npt.NDArray[np.float32]] | npt.NDArray[np.float32] = []
    for i in range(n_partitions):
        batch_quant_conv : npt.NDArray[np.float32] = model.predict(quantized_data_input[i*BATCH_SIZE: (i + 1)*BATCH_SIZE]) # np.float32
        batch_quant_conv_bias : npt.NDArray[np.int32] = tf.nn.bias_add(batch_quant_conv, q_model_info.quantized_bias[key]).numpy().astype(int)
        batch_quant_postactiv : npt.NDArray[np.int32] = tf.nn.relu(batch_quant_conv_bias).numpy()
        
        quant_postactiv.append(batch_quant_postactiv)

    # Garbage collection
    del batch_quant_conv # 351.5626 MBi Numpy array
    del batch_quant_conv_bias # 351.5626 MBi Numpy array
    del batch_quant_postactiv # 351.5626 MBi Numpy array
    Quantization.garbage_collection()

    quant_postactiv = np.concatenate(quant_postactiv)
    dequant_postactiv = (q_model_info.bias_scales[key] * quant_postactiv).astype(np.float32)

    # Garbage collection
    del quant_postactiv # 703.12515 MBi
    Quantization.garbage_collection()

    quant_rescaled_postactiv = np.round(dequant_postactiv / q_model_info.output_scales[key]).astype(np.int16)

    # Garbage collection
    del dequant_postactiv # 703.12515 MBi
    Quantization.garbage_collection()
    
    rescaled_postactiv = (q_model_info.output_scales[key] * quant_rescaled_postactiv).astype(np.float32)

    # Garbage collection
    del quant_rescaled_postactiv # 351.5626 MBi Numpy array
    Quantization.garbage_collection()
    
    q_model_output = Quantization.model_begin_predict(q_aware_model, non_quantized_data_input, layer_index_stop = start_index)
    difference = q_model_output - rescaled_postactiv
    difference_values, counts =  np.unique(difference, return_counts = True)

    print(f"Manual quantized output shape: {rescaled_postactiv.shape}")
    print(f"Q model output shape: {q_model_output.shape}")
    print(f"Manual activation rescaled float output max: {np.max(rescaled_postactiv)}")
    print(f"Manual activation rescaled float output min: {np.min(rescaled_postactiv)}")
    print(f"Q model float output max: {np.max(q_model_output)}")
    print(f"Q model float output min: {np.min(q_model_output)}")
    print(f"Unique differences: {difference_values} and respective counts: {counts}")
    # Garbage collection
    del rescaled_postactiv # 703.12515 MBi Numpy array
    del q_model_output # 703.12515 MBi Numpy array
    del difference # 703.12515 MBi Numpy array
    Quantization.garbage_collection()

OPERATION_MODE = 3                                      # Modification of operation mode
# Number of partitions for batch analysis
N_PARTITIONS = 2
# First index of q_aware_model
SPLIT_INDEX = 3
FIRST_QUANTIZED = True

LOAD_PATH_Q_AWARE = "./model/model_q_aware_final_01"

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# Load Q Aware model
with tfmot.quantization.keras.quantize_scope():
    q_aware_model : Functional = tf.keras.models.load_model(LOAD_PATH_Q_AWARE)

q_model_info = Quantization.QuantizedModelInfo(q_aware_model)

# Preparing the quantized test set
quantized_test_images = np.round(test_images[:,:,:,np.newaxis]/q_model_info.output_scales[q_model_info.keys[0]]).astype(int)

# Generating the split models
m1, m2 = Quantization.split_model_mixed(q_aware_model, q_model_info, start_index = SPLIT_INDEX, first_quantized = FIRST_QUANTIZED)

if FIRST_QUANTIZED:
    by_batches_quantized(
        data_input = quantized_test_images,
        test_labels = test_labels,
        n_partitions = N_PARTITIONS,
        q_model_info = q_model_info,
        start_index = SPLIT_INDEX,
        model = m1,
        )
    # by_batches_comparison(
    #     quantized_data_input = quantized_test_images,
    #     non_quantized_data_input = test_images[:,:,:,np.newaxis],
    #     test_labels = test_labels,
    #     n_partitions = N_PARTITIONS,
    #     q_model_info = q_model_info,
    #     q_aware_model = q_aware_model,
    #     start_index = SPLIT_INDEX,
    #     model = m1,
    # )
    by_batches_alternative_comparison(
        quantized_data_input = quantized_test_images,
        non_quantized_data_input = test_images[:,:,:,np.newaxis],
        test_labels = test_labels,
        n_partitions = N_PARTITIONS,
        q_model_info = q_model_info,
        q_aware_model = q_aware_model,
        start_index = SPLIT_INDEX,
        model = m1,
    )
else:
    by_batches_non_quantized(
        data_input = test_images[:,:,:,np.newaxis],
        test_labels = test_labels,
        n_partitions = N_PARTITIONS,
        q_aware_model = q_aware_model,
        start_index = SPLIT_INDEX,
        model = m1,
    )

# Deletion of unsused variable to diminish RAM usage, highest memory value so far
del train_images # 358.8868 MBi
del test_images # 59.8145 MBi
del quantized_test_images # 29.9073 MBi
del m1 # 48 bytes
del m2 # 48 bytes
Quantization.garbage_collection()